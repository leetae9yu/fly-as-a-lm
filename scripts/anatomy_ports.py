"""Exact anatomical port classes and directed connectivity summaries."""

from dataclasses import dataclass

import numpy as np

from flyrl.connectome import GraphError
from scripts.connectome_arrays import Anatomy, IntArray


@dataclass(frozen=True, slots=True)
class ConnectionCount:
    """Directed pair and anatomical contact totals between two classes."""

    directed_pairs: int
    contacts: int


@dataclass(frozen=True, slots=True)
class AnatomyPortAudit:
    """Aligned ALPN, Kenyon-cell and MBON indices with pathway evidence."""

    alpn_indices: IntArray
    mbon_indices: IntArray
    kenyon_indices: IntArray
    alpn_to_kenyon: ConnectionCount
    kenyon_to_mbon: ConnectionCount
    alpn_to_mbon: ConnectionCount
    reachable_mbons: int


def _connection_count(
    graph: Anatomy,
    source_indices: IntArray,
    target_indices: IntArray,
) -> ConnectionCount:
    source_mask = np.zeros(graph.ids.size, dtype=np.bool_)
    target_mask = np.zeros(graph.ids.size, dtype=np.bool_)
    source_mask[source_indices] = True
    target_mask[target_indices] = True
    selected = source_mask[graph.edges.source] & target_mask[graph.edges.target]
    return ConnectionCount(
        directed_pairs=int(selected.sum(dtype=np.int64)),
        contacts=int(graph.edges.count[selected].sum(dtype=np.int64)),
    )


def anatomy_port_audit(
    graph: Anatomy,
    classes: tuple[str, ...],
) -> AnatomyPortAudit:
    """Select exact publisher classes and audit directed paths inside one graph."""
    if len(classes) != graph.ids.size:
        message = "Class labels must align with graph IDs"
        raise GraphError(message)
    alpn = np.asarray(
        [index for index, label in enumerate(classes) if label == "ALPN"],
        dtype=np.int64,
    )
    mbon = np.asarray(
        [index for index, label in enumerate(classes) if label == "MBON"],
        dtype=np.int64,
    )
    kenyon = np.asarray(
        [index for index, label in enumerate(classes) if label == "Kenyon_Cell"],
        dtype=np.int64,
    )
    if not alpn.size or not mbon.size or not kenyon.size:
        message = "Anatomy ports require ALPN, Kenyon_Cell and MBON classes"
        raise GraphError(message)
    reachable = np.zeros(graph.ids.size, dtype=np.bool_)
    reachable[alpn] = True
    while True:
        expanded = reachable.copy()
        expanded[graph.edges.target[reachable[graph.edges.source]]] = True
        if expanded.tobytes() == reachable.tobytes():
            break
        reachable = expanded
    return AnatomyPortAudit(
        alpn_indices=alpn,
        mbon_indices=mbon,
        kenyon_indices=kenyon,
        alpn_to_kenyon=_connection_count(graph, alpn, kenyon),
        kenyon_to_mbon=_connection_count(graph, kenyon, mbon),
        alpn_to_mbon=_connection_count(graph, alpn, mbon),
        reachable_mbons=int(reachable[mbon].sum(dtype=np.int64)),
    )
