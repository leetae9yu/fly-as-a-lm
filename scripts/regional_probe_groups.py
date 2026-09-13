"""Deterministic equal-width neuron groups for frozen regional probes."""

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import TypeAdapter

from flyrl.connectome import Graph, GraphError

ProbeGroupName: TypeAlias = Literal[
    "alpn",
    "kenyon",
    "mbon",
    "degree_matched",
    "centrality",
    "trained_readout",
]
FloatMatrix: TypeAlias = NDArray[np.float64]
IntVector: TypeAlias = NDArray[np.int64]
SELECTION_SEED: Final = 20260913


@dataclass(frozen=True, slots=True)
class ProbeGroup:
    """One exact ordered 97-neuron probe input."""

    name: ProbeGroupName
    draw: int
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RegionalSelection:
    """Anatomical classes and functional ports for one source checkpoint seed."""

    alpn: tuple[int, ...]
    mbon: tuple[int, ...]
    kenyon: tuple[int, ...]
    sensory: tuple[int, ...]
    trained_readout: tuple[int, ...]
    seed: int


@dataclass(frozen=True, slots=True)
class _SamplePlan:
    """Private deterministic subset draw request."""

    name: Literal["alpn", "kenyon"]
    candidates: tuple[int, ...]
    seed: int
    group_id: int
    group_size: int
    draws: int


def _sample_groups(plan: _SamplePlan) -> tuple[ProbeGroup, ...]:
    if len(plan.candidates) < plan.group_size:
        message = f"Probe group {plan.name} has too few eligible neurons"
        raise GraphError(message)
    values: IntVector = np.asarray(plan.candidates, dtype=np.int64)
    groups: list[ProbeGroup] = []
    for draw in range(plan.draws):
        rng = np.random.default_rng(
            np.random.SeedSequence([SELECTION_SEED, plan.seed, plan.group_id, draw])
        )
        selected = values.copy()
        rng.shuffle(selected)
        selected_tuple = TypeAdapter(tuple[int, ...]).validate_python(
            selected[: plan.group_size].tolist()
        )
        indices = tuple(sorted(selected_tuple))
        groups.append(ProbeGroup(plan.name, draw, indices))
    return tuple(groups)


def _structural_features(graph: Graph) -> FloatMatrix:
    nodes = len(graph.node_ids)
    strength = np.abs(graph.weight)
    columns = (
        np.bincount(graph.source, minlength=nodes),
        np.bincount(graph.target, minlength=nodes),
        np.bincount(graph.source, weights=strength, minlength=nodes),
        np.bincount(graph.target, weights=strength, minlength=nodes),
    )
    features: FloatMatrix = np.asarray(columns, dtype=np.float64).T
    return np.log1p(features)


def _degree_matched(
    graph: Graph,
    mbon: tuple[int, ...],
    candidates: tuple[int, ...],
) -> tuple[int, ...]:
    features = _structural_features(graph)
    scale = features.std(axis=0)
    scale[scale == 0] = 1
    normalized = (features - features.mean(axis=0)) / scale
    candidate_array: IntVector = np.asarray(candidates, dtype=np.int64)
    mbon_array: IntVector = np.asarray(mbon, dtype=np.int64)
    difference = (
        normalized[mbon_array, None, :] - normalized[candidate_array][None, :, :]
    )
    costs: FloatMatrix = np.square(difference).sum(axis=2)
    minimum_cost: NDArray[np.float64] = costs.min(axis=1)
    priority = sorted(
        range(mbon_array.size),
        key=lambda row: (
            -float(minimum_cost.item(row)),
            int(mbon_array.item(row)),
        ),
    )
    available = np.ones(candidate_array.size, dtype=np.bool_)
    selected: list[int] = []
    for row in priority:
        match = min(
            (
                float(cast("np.float64", costs[row, column])),
                column,
            )
            for column in range(candidate_array.size)
            if bool(available.item(column))
        )[1]
        available[match] = False
        selected.append(candidates[match])
    return tuple(sorted(selected))


def regional_probe_groups(
    graph: Graph,
    selection: RegionalSelection,
    *,
    group_size: int = 97,
    draws: int = 5,
) -> tuple[ProbeGroup, ...]:
    """Build paired anatomy and topology controls without observing metrics."""
    nodes = len(graph.node_ids)
    anatomy = set(selection.alpn) | set(selection.mbon) | set(selection.kenyon)
    blocked = set(selection.sensory) | set(selection.trained_readout)
    if (
        group_size < 1
        or draws < 1
        or len(selection.mbon) != group_size
        or len(selection.trained_readout) != group_size
        or len(anatomy)
        != len(selection.alpn) + len(selection.mbon) + len(selection.kenyon)
        or any(index < 0 or index >= nodes for index in anatomy | blocked)
    ):
        message = "Regional probe classes, ports or capacities are invalid"
        raise GraphError(message)
    alpn_candidates = tuple(sorted(set(selection.alpn) - blocked))
    kenyon_candidates = tuple(sorted(set(selection.kenyon) - blocked))
    other_candidates = tuple(
        index for index in range(nodes) if index not in anatomy | blocked
    )
    if len(other_candidates) < group_size:
        message = "Regional probe requires enough unannotated internal controls"
        raise GraphError(message)
    strength = np.bincount(
        graph.source,
        weights=np.abs(graph.weight),
        minlength=nodes,
    ) + np.bincount(
        graph.target,
        weights=np.abs(graph.weight),
        minlength=nodes,
    )
    centrality_order = sorted(
        other_candidates,
        key=lambda index: (-float(strength.item(index)), index),
    )
    centrality = TypeAdapter(tuple[int, ...]).validate_python(
        sorted(centrality_order[:group_size])
    )
    return (
        *_sample_groups(
            _SamplePlan(
                "alpn",
                alpn_candidates,
                selection.seed,
                0,
                group_size,
                draws,
            )
        ),
        *_sample_groups(
            _SamplePlan(
                "kenyon",
                kenyon_candidates,
                selection.seed,
                1,
                group_size,
                draws,
            )
        ),
        ProbeGroup("mbon", 0, tuple(sorted(selection.mbon))),
        ProbeGroup(
            "degree_matched",
            0,
            _degree_matched(graph, selection.mbon, other_candidates),
        ),
        ProbeGroup("centrality", 0, centrality),
        ProbeGroup(
            "trained_readout",
            0,
            tuple(sorted(selection.trained_readout)),
        ),
    )
