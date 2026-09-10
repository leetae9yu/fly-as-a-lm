"""Primitive-array connectome transformations."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from numpy.lib.format import write_array

from flyrl.connectome import Graph, GraphError, save_graph

IntArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int64]]
IdArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint64]]
NumericArray: TypeAlias = np.ndarray[
    tuple[int], np.dtype[np.int64 | np.uint64 | np.float64]
]


@dataclass(frozen=True, slots=True)
class Edges:
    """Directed pre/post columns with anatomical contact counts, never signs."""

    source: IntArray
    target: IntArray
    count: IntArray


@dataclass(frozen=True, slots=True)
class Anatomy:
    """Sorted exact body IDs and canonical, unique directed edge indices."""

    ids: IdArray
    edges: Edges


def exact_ids(values: NumericArray) -> IdArray:
    """Parse integer source IDs without a floating-point intermediate."""
    if values.ndim != 1 or values.dtype.kind not in "iu" or (values < 0).any():
        message = "Body IDs must be a one-dimensional nonnegative integer array"
        raise GraphError(message)
    return values.astype(np.uint64, copy=False)


def index_edges(ids: IdArray, raw: Edges) -> Edges:
    """Filter endpoints to retained IDs while preserving pre -> post direction."""
    if ids.size == 0 or (ids[1:] <= ids[:-1]).any():
        message = "Retained body IDs must be sorted and unique"
        raise GraphError(message)
    pre, post = exact_ids(raw.source), exact_ids(raw.target)
    count = raw.count
    if (
        count.ndim != 1
        or count.dtype.kind not in "iu"
        or (count <= 0).any()
        or (count > np.iinfo(np.uint32).max).any()
        or not (pre.size == post.size == count.size)
    ):
        message = "Synapse counts must be aligned positive uint32-compatible integers"
        raise GraphError(message)
    source: IntArray = ids.searchsorted(pre).astype(np.int64)
    target: IntArray = ids.searchsorted(post).astype(np.int64)
    keep: np.ndarray[tuple[int], np.dtype[np.bool_]] = (source < ids.size) & (
        target < ids.size
    )
    keep = np.logical_and(keep, np.equal(ids[np.minimum(source, ids.size - 1)], pre))
    keep = np.logical_and(keep, np.equal(ids[np.minimum(target, ids.size - 1)], post))
    return Edges(
        source[keep].astype(np.int64),
        target[keep].astype(np.int64),
        count[keep].astype(np.int64),
    )


def aggregate_edges(ids: IdArray, edges: Edges) -> Anatomy:
    """Sum duplicate pairs using primitive arrays; order by (pre, post)."""
    keys = edges.source * ids.size + edges.target
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    first = np.ones(sorted_keys.size, dtype=np.bool_)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    starts = first.nonzero()[0]
    counts = np.add.reduceat(edges.count[order], starts)
    pairs = sorted_keys[starts]
    return Anatomy(ids, Edges(pairs // ids.size, pairs % ids.size, counts))


def induced_subgraph(graph: Anatomy, selected: IntArray) -> Anatomy:
    """Retain every internal directed edge, with canonical original-ID order."""
    nodes = np.sort(selected)
    mapping = np.full(graph.ids.size, -1, dtype=np.int64)
    mapping[nodes] = np.arange(nodes.size, dtype=np.int64)
    source, target = mapping[graph.edges.source], mapping[graph.edges.target]
    keep = (source >= 0) & (target >= 0)
    return Anatomy(
        graph.ids[nodes], Edges(source[keep], target[keep], graph.edges.count[keep])
    )


def incident_contacts(graph: Anatomy) -> IntArray:
    """Count incident contacts exactly, counting autapses in both directions."""
    strength = np.zeros(graph.ids.size, dtype=np.int64)
    np.add.at(strength, graph.edges.source, graph.edges.count)
    np.add.at(strength, graph.edges.target, graph.edges.count)
    return strength


def connected_order(graph: Anatomy, size: int) -> IntArray:
    """BFS from the strongest hub; rank each expansion by strength then body ID.

    Every prefix is weakly connected, not necessarily strongly connected.
    This samples a high-contact hub neighborhood, not an unbiased brain region.
    Only the incoming edge permutation is O(E); no Python objects per edge.
    """
    if not 1 <= size <= graph.ids.size:
        message = "Selection size must be within the retained neuron count"
        raise GraphError(message)
    edges = graph.edges
    strength = incident_contacts(graph)
    root = int(np.argmax(strength))
    incoming_order = np.argsort(edges.target, kind="stable")
    outgoing_ptr = np.zeros(graph.ids.size + 1, dtype=np.int64)
    incoming_ptr = np.zeros(graph.ids.size + 1, dtype=np.int64)
    outgoing_ptr[1:] = np.bincount(edges.source, minlength=graph.ids.size).cumsum()
    incoming_ptr[1:] = np.bincount(edges.target, minlength=graph.ids.size).cumsum()
    queue: IntArray = np.empty(size, dtype=np.int64)
    queue[0] = root
    seen = np.zeros(graph.ids.size, dtype=np.bool_)
    seen[root] = True
    head, tail = 0, 1
    while head < tail < size:
        node = queue.item(head)
        outgoing = edges.target[outgoing_ptr[node] : outgoing_ptr[node + 1]]
        incoming = edges.source[
            incoming_order[incoming_ptr[node] : incoming_ptr[node + 1]]
        ]
        neighbors = np.unique(np.concatenate((outgoing, incoming)))
        fresh = neighbors[~seen[neighbors]]
        ranked = fresh[np.lexsort((graph.ids[fresh], -strength[fresh]))]
        added = ranked[: size - tail]
        queue[tail : tail + added.size] = added
        seen[added] = True
        tail += added.size
        head += 1
    if tail != size:
        message = "Hub component too small; refusing to pad with disconnected neurons"
        raise GraphError(message)
    return queue


def export_graph(graph: Anatomy, path: Path, provenance: str) -> None:
    """Write Graph-compatible NPZ plus exact int64 anatomical synapse_count."""
    modeled = Graph(
        tuple(f"malecns:v1.0:{int(body)}" for body in graph.ids),
        graph.edges.source,
        graph.edges.target,
        np.log1p(graph.edges.count.astype(np.float64)),
        provenance,
    )
    save_graph(modeled, path)
    with (
        ZipFile(path, "a", compression=ZIP_DEFLATED) as archive,
        archive.open("synapse_count.npy", "w") as stream,
    ):
        write_array(stream, graph.edges.count, allow_pickle=False)
