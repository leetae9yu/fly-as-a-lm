"""Strict structural checks for sealed central-replication checkpoints."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from pydantic import TypeAdapter

from flyrl.activations import ActivationMetadata
from flyrl.bpe_data import BPECorpus
from flyrl.connectome import Graph
from flyrl.story_pilot import PilotReport
from scripts.central_replication_schema import Protocol

if TYPE_CHECKING:
    from numpy import generic


@dataclass(frozen=True, slots=True)
class ActivationEvidence:
    """Inputs needed to bind an activation export to sealed training evidence."""

    output: Path
    activation: ActivationMetadata
    report: PilotReport
    protocol: Protocol
    graph: Graph
    corpus: BPECorpus
    parameter_fingerprint: str


def validate_checkpoint_tensors(
    data: "NpzFile[generic]",
    model_shapes: Mapping[str, tuple[int, ...]],
    updates: int,
) -> None:
    """Require every model, AdamW and RNG array with exact shape and dtype."""
    expected = {"metadata", "rng"}
    for name in model_shapes:
        expected.add(f"model.{name}")
        expected.update(
            {
                f"adam.{name}.exp_avg",
                f"adam.{name}.exp_avg_sq",
                f"adam.{name}.step",
            }
        )
    if set(data.files) != expected:
        message = "Checkpoint tensor key set is incomplete or unexpected"
        raise ValueError(message)
    rng = data["rng"]
    if rng.dtype != np.uint8 or rng.ndim != 1 or rng.size == 0:
        message = "Checkpoint RNG invariant violation"
        raise ValueError(message)
    _ = torch.Generator().set_state(torch.tensor(np.asarray(rng, dtype=np.uint8)))
    for name, shape in model_shapes.items():
        parameter = data[f"model.{name}"]
        if (
            parameter.shape != shape
            or parameter.dtype != np.float32
            or not np.isfinite(parameter).all()
        ):
            message = f"Checkpoint parameter invariant violation: {name}"
            raise ValueError(message)
        for moment in ("exp_avg", "exp_avg_sq"):
            value = data[f"adam.{name}.{moment}"]
            if (
                value.shape != shape
                or value.dtype != np.float32
                or not np.isfinite(value).all()
                or (
                    moment == "exp_avg_sq"
                    and bool((np.asarray(value, dtype=np.float32) < 0).any())
                )
            ):
                message = f"Checkpoint AdamW invariant violation: {name}/{moment}"
                raise ValueError(message)
        step = data[f"adam.{name}.step"]
        if (
            step.shape != ()
            or step.dtype != np.float32
            or not np.isfinite(step).all()
            or TypeAdapter(float).validate_python(step.item()) != updates
        ):
            message = f"Checkpoint AdamW step violation: {name}"
            raise ValueError(message)


def checkpoint_parameter_fingerprint(
    data: "NpzFile[generic]",
    config_json: str,
    parameter_order: tuple[str, ...],
) -> str:
    """Recompute the exporter fingerprint from immutable checkpoint arrays."""
    digest = hashlib.sha256(config_json.encode())
    for name in parameter_order:
        digest.update(name.encode())
        digest.update(data[f"model.{name}"].tobytes())
    return digest.hexdigest()


def validate_activation_export(evidence: ActivationEvidence) -> None:
    """Bind every activation field and file to its report and checkpoint."""
    output = evidence.output
    activation = evidence.activation
    report = evidence.report
    protocol = evidence.protocol
    graph = evidence.graph
    corpus = evidence.corpus
    prompt_ids = tuple(corpus.encode(protocol.prompt))
    generated = activation.generated_ids
    if (
        activation.config != report.config
        or activation.updates != protocol.updates
        or activation.graph_fingerprint != report.graph_fingerprint
        or activation.corpus_fingerprint != report.corpus_fingerprint
        or activation.parameter_fingerprint != evidence.parameter_fingerprint
        or activation.prompt != protocol.prompt
        or activation.prompt_ids != prompt_ids
        or len(generated) != protocol.sample_length
        or len(activation.contexts) != protocol.sample_length
        or len(activation.token_labels) != protocol.sample_length
        or activation.generated_text != corpus.decode([*prompt_ids, *generated])
        or not activation.greedy
        or activation.neurons_total != protocol.graph_nodes
        or activation.arrays != "activations.npz"
    ):
        message = "Activation metadata identity or completeness mismatch"
        raise ValueError(message)
    expected_figures = tuple(
        f"activations-{page:03}.{extension}"
        for page in range(1, (protocol.sample_length + 31) // 32 + 1)
        for extension in ("png", "svg")
    )
    if activation.figures != expected_figures:
        message = "Activation figure manifest mismatch"
        raise ValueError(message)
    with (
        (output / activation.arrays).open("rb") as stream,
        NpzFile(stream, allow_pickle=False) as data,
    ):
        if set(data.files) != {
            "states",
            "generated_ids",
            "probabilities",
            "node_ids",
            "selected_indices",
        }:
            message = "Activation array key set mismatch"
            raise ValueError(message)
        states = np.asarray(data["states"], dtype=np.float32)
        probabilities = np.asarray(data["probabilities"], dtype=np.float32)
        generated_array = data["generated_ids"]
        selected_array = data["selected_indices"]
        node_ids = TypeAdapter(tuple[str, ...]).validate_python(
            data["node_ids"].tolist()
        )
        selected = TypeAdapter(tuple[int, ...]).validate_python(selected_array.tolist())
        if (
            data["states"].dtype != np.float32
            or states.shape != (protocol.sample_length, protocol.graph_nodes)
            or not np.isfinite(states).all()
            or bool((np.abs(states) > 1).any())
            or generated_array.dtype != np.int64
            or generated_array.shape != (protocol.sample_length,)
            or TypeAdapter(tuple[int, ...]).validate_python(generated_array.tolist())
            != generated
            or data["probabilities"].dtype != np.float32
            or probabilities.shape != (protocol.sample_length,)
            or not np.isfinite(probabilities).all()
            or bool(((probabilities < 0) | (probabilities > 1)).any())
            or node_ids != tuple(graph.node_ids)
            or selected_array.dtype != np.int64
            or selected_array.shape != (protocol.activation_neurons,)
            or len(set(selected)) != protocol.activation_neurons
            or any(index < 0 or index >= protocol.graph_nodes for index in selected)
        ):
            message = "Activation array identity or numeric invariant mismatch"
            raise ValueError(message)
        scores = TypeAdapter(tuple[float, ...]).validate_python(
            np.mean(np.abs(states), axis=0).tolist()
        )
        expected_selected = tuple(
            sorted(range(protocol.graph_nodes), key=lambda index: -scores[index])[
                : protocol.activation_neurons
            ]
        )
        if selected != expected_selected:
            message = "Activation selected-neuron ranking mismatch"
            raise ValueError(message)
    rng = np.random.default_rng(report.config.seed)
    order = np.arange(protocol.graph_nodes, dtype=np.int64)
    rng.shuffle(order)
    order_indices = TypeAdapter(tuple[int, ...]).validate_python(order.tolist())
    sensory = tuple(graph.node_ids[index] for index in order_indices[:192])
    readout = tuple(
        graph.node_ids[index] for index in order_indices[-protocol.readout_neurons :]
    )
    if (
        activation.selected_node_ids
        != tuple(graph.node_ids[index] for index in selected)
        or activation.sensory_node_ids != sensory
        or activation.readout_node_ids != readout
        or activation.token_labels
        != tuple(corpus.vocabulary[token] for token in generated)
        or activation.generated_ids != report.generated_token_ids["greedy"]
        or any(
            context != prompt_ids + generated[:index]
            for index, context in enumerate(activation.contexts)
        )
    ):
        message = "Activation token, port or causal-context mismatch"
        raise ValueError(message)
    for figure in activation.figures:
        path = output / figure
        signature = path.read_bytes()[:8] if path.is_file() else b""
        if (figure.endswith(".png") and signature != b"\x89PNG\r\n\x1a\n") or (
            figure.endswith(".svg") and not signature.lstrip().startswith(b"<?xml")
        ):
            message = f"Activation figure missing or invalid: {figure}"
            raise ValueError(message)
