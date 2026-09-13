"""Reconstruct activation port IDs for legacy and explicit configurations."""

from typing import Protocol, assert_never

import numpy as np
from pydantic import TypeAdapter

from flyrl.connectome import Graph
from flyrl.story_pilot import PilotReport


class GraphProtocol(Protocol):
    """Protocol fields needed to reconstruct legacy random ports."""

    graph_nodes: int


def activation_port_ids(
    report: PilotReport,
    protocol: GraphProtocol,
    graph: Graph,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return exact sensory and readout stable IDs for one recovered report."""
    match report.config.port_policy:
        case "legacy_random":
            rng = np.random.default_rng(report.config.seed)
            order = np.arange(protocol.graph_nodes, dtype=np.int64)
            rng.shuffle(order)
            order_indices = TypeAdapter(tuple[int, ...]).validate_python(order.tolist())
            match report.config.tokenization:
                case "character":
                    sensory_budget = report.config.alphabet_size * 4
                case "bpe":
                    sensory_budget = 192
                case _:
                    assert_never(report.config.tokenization)
            sensory_count = min(protocol.graph_nodes // 2, sensory_budget)
            readout_count = min(
                report.config.readout_neurons,
                protocol.graph_nodes - sensory_count,
            )
            return (
                tuple(graph.node_ids[index] for index in order_indices[:sensory_count]),
                tuple(
                    graph.node_ids[index] for index in order_indices[-readout_count:]
                ),
            )
        case "alpn_mbon" | "alpn_random" | "random_mbon" | "random_random":
            if (
                report.config.sensory_indices is None
                or report.config.readout_indices is None
            ):
                message = "Explicit activation ports are missing from config"
                raise ValueError(message)
            return (
                tuple(graph.node_ids[index] for index in report.config.sensory_indices),
                tuple(graph.node_ids[index] for index in report.config.readout_indices),
            )
        case _:
            assert_never(report.config.port_policy)
