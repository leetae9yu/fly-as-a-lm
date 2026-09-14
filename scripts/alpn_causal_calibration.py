"""Read-only paired calibration from restored sources and explicit old training."""

from dataclasses import dataclass
from typing import Annotated, Final

import numpy as np
from pydantic import Field

from flyrl.ar_model import ConnectomeLM
from flyrl.connectome import Graph
from flyrl.language_data import IntVector
from flyrl.regional_probe import SATURATION_THRESHOLD, extract_features
from flyrl.regional_probe_types import ExtractionConfig
from flyrl.story_data import StorySplit
from scripts.alpn_causal_groups import FloatMatrix, assemble_coordinates

OLD_TRAINING_TARGETS: Final = 229_745
DISTANCE_CAP: Final = 3


class CalibrationConfig(ExtractionConfig):
    """Story-safe extraction with an explicit expected old-training target count."""

    expected_targets: Annotated[int, Field(ge=1)] = OLD_TRAINING_TARGETS


@dataclass(frozen=True, slots=True)
class TrainingActivity:
    """Raw signed mean, population std and saturation, in explicit neuron order."""

    indices: tuple[int, ...]
    positions: int
    values: FloatMatrix


def original_graph_coordinates(graph: Graph) -> FloatMatrix:
    """Return raw in/out degree then in/out absolute strength in original node order."""
    nodes = len(graph.node_ids)
    strength = np.abs(graph.weight)
    return np.asarray(
        (
            np.bincount(graph.target, minlength=nodes),
            np.bincount(graph.source, minlength=nodes),
            np.bincount(graph.target, weights=strength, minlength=nodes),
            np.bincount(graph.source, weights=strength, minlength=nodes),
        ),
        dtype=np.float64,
    ).T


def outgoing_weight_l2(model: ConnectomeLM) -> FloatMatrix:
    """Reduce learned edge-weight squares by source, before parallel-edge coalescing.

    Both endpoints and weights use parameter order, not canonical sparse order.
    Shuffled parallel edges remain separate learned outgoing signals.
    """
    source = np.asarray(model.edges[1].detach().cpu().numpy(), dtype=np.int64)
    weight = np.asarray(model.weight.detach().cpu().numpy(), dtype=np.float64)
    return np.sqrt(np.bincount(source, weights=weight * weight, minlength=model.nodes))


def _distances(
    source: IntVector, target: IntVector, ports: IntVector, nodes: int
) -> FloatMatrix:
    distances = np.full(nodes, DISTANCE_CAP, dtype=np.float64)
    distances[ports] = 0
    for depth in range(1, DISTANCE_CAP):
        reached = target[distances[source] == depth - 1]
        distances[reached] = np.minimum(distances[reached], depth)
    return distances


def directed_distances(model: ConnectomeLM) -> FloatMatrix:
    """Return distance from sensory and to readout using this wiring's actual edges.

    Reverse traversal starts at readout ports. Ports have distance zero; paths
    of length >=3 and unreachable nodes all have distance three. Zero learned
    weights do not delete anatomical edges.
    """
    source = np.asarray(model.edges[1].detach().cpu().numpy(), dtype=np.int64)
    target = np.asarray(model.edges[0].detach().cpu().numpy(), dtype=np.int64)
    sensory = np.asarray(model.sensory.detach().cpu().numpy(), dtype=np.int64)
    readout = np.asarray(model.ports.detach().cpu().numpy(), dtype=np.int64)
    return np.asarray(
        (
            _distances(source, target, sensory, model.nodes),
            _distances(target, source, readout, model.nodes),
        ),
        dtype=np.float64,
    ).T


def training_activity(
    model: ConnectomeLM,
    tokens: IntVector,
    split: StorySplit,
    indices: tuple[int, ...],
    *,
    config: CalibrationConfig | None = None,
) -> TrainingActivity:
    """Summarize every old-training target, never padding or cross-story positions.

    The explicit stream/split must be authenticated as old training by the caller;
    no corpus loader, fresh split, RNG or model mode is accessed here. The default
    count enforces the frozen training extent; override only for other/test corpora.
    Reduction uses float64 one column at a time without a full float64 state copy.
    """
    options = config or CalibrationConfig()
    positions = sum(
        max(0, end - start - 1)
        for start, end in zip(split.offsets[:-1], split.offsets[1:], strict=True)
    )
    if positions != options.expected_targets:
        message = "Old-training target count differs from expected targets"
        raise ValueError(message)
    cache = extract_features(model, tokens, split, indices, config=options)
    values = np.empty((len(indices), 3), dtype=np.float64)
    for column in range(len(indices)):
        states = np.asarray(cache.features[:, column], dtype=np.float64)
        values[column] = (
            states.mean(),
            states.std(ddof=0),
            (np.abs(states) >= SATURATION_THRESHOLD).mean(dtype=np.float64),
        )
    values.setflags(write=False)
    return TrainingActivity(indices, positions, values)


def paired_coordinates(
    graph: Graph,
    real: ConnectomeLM,
    shuffled: ConnectomeLM,
    activity: tuple[TrainingActivity, TrainingActivity],
) -> FloatMatrix:
    """Assemble the documented 16 coordinates, preserving activity.indices row order.

    Activity is supplied real first, shuffled second. The caller owns paired source
    identity and graph provenance. All original-node rows can be requested for the
    matcher, or an explicit ordered subset for calibration/extraction batches.
    """
    first, second = activity
    indices = first.indices
    nodes = len(graph.node_ids)
    if (
        real.nodes != nodes
        or shuffled.nodes != nodes
        or not indices
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= nodes
        or indices != second.indices
        or first.positions != second.positions
        or first.positions <= 0
        or any(item.values.shape != (len(indices), 3) for item in activity)
    ):
        message = "Paired activity requires matching valid indices, counts and shapes"
        raise ValueError(message)
    propagation = np.empty((nodes, 6), dtype=np.float64)
    for wiring, model in enumerate((real, shuffled)):
        propagation[:, wiring] = outgoing_weight_l2(model)
        propagation[:, 2 + wiring : 6 : 2] = directed_distances(model)
    paired_activity = np.empty((len(indices), 6), dtype=np.float64)
    paired_activity[:, ::2], paired_activity[:, 1::2] = first.values, second.values
    return assemble_coordinates(
        original_graph_coordinates(graph)[list(indices)],
        propagation[list(indices)],
        paired_activity,
    )
