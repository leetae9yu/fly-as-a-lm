"""Sparse, directed connectome input and degree-preserving control graphs."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from pydantic import TypeAdapter
from typing_extensions import override

if TYPE_CHECKING:
    from numpy import generic


@dataclass(frozen=True, slots=True)
class Graph:
    """A directed graph with stable node identities and signed initial weights."""

    node_ids: tuple[str, ...]
    source: np.ndarray[tuple[int], np.dtype[np.int64]]
    target: np.ndarray[tuple[int], np.dtype[np.int64]]
    weight: np.ndarray[tuple[int], np.dtype[np.float64]]
    provenance: str

    def __post_init__(self) -> None:
        """Reject invalid sparse graphs before their first simulation step."""
        count = len(self.node_ids)
        if count <= 1 or len(set(self.node_ids)) != count:
            message = "At least two uniquely identified neurons are required"
            raise GraphError(message)
        if any(array.ndim != 1 for array in (self.source, self.target, self.weight)):
            message = "Edge arrays must be one-dimensional"
            raise GraphError(message)
        if not (self.source.size == self.target.size == self.weight.size > 0):
            message = "Edge arrays must have the same nonzero length"
            raise GraphError(message)
        if any(
            array.dtype != np.dtype(np.int64) for array in (self.source, self.target)
        ):
            message = "Edge indices must be int64"
            raise GraphError(message)
        if any(
            ((array < 0) | (array >= count)).any()
            for array in (self.source, self.target)
        ):
            message = "Edge index outside the node array"
            raise GraphError(message)
        if not np.isfinite(self.weight).all() or not self.weight.all():
            message = "Weights must be finite and nonzero"
            raise GraphError(message)
        pairs = np.empty((self.source.size, 2), dtype=np.int64)
        pairs[:, 0], pairs[:, 1] = self.source, self.target
        records = pairs.view(
            dtype=[("source", np.int64), ("target", np.int64)]
        ).reshape(-1)
        records.sort(order=("source", "target"))
        duplicates = np.equal(pairs[1:, 0], pairs[:-1, 0])
        duplicates &= np.equal(pairs[1:, 1], pairs[:-1, 1])
        if duplicates.any():
            message = "Parallel edges must be aggregated before import"
            raise GraphError(message)
        for array in (self.source, self.target, self.weight):
            array.flags.writeable = False


@dataclass(frozen=True, slots=True)
class GraphError(ValueError):
    """An invalid graph at the file or adjacency input boundary."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


def from_adjacency(
    adjacency: NDArray[np.float64], node_ids: tuple[str, ...], provenance: str
) -> Graph:
    """Convert nonnegative synapse counts to log-scaled sparse weights."""
    if adjacency.shape != (len(node_ids), len(node_ids)):
        message = "Adjacency must be square and match the node identities"
        raise GraphError(message)
    if not np.isfinite(adjacency).all() or (adjacency < 0).any():
        message = "Synapse counts must be finite and nonnegative"
        raise GraphError(message)
    source, target = np.nonzero(adjacency)
    return Graph(
        node_ids,
        source.astype(np.int64),
        target.astype(np.int64),
        np.log1p(adjacency[source, target]),
        provenance,
    )


def save_graph(graph: Graph, path: Path) -> None:
    """Save a graph without pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = (
        ("node_ids", np.asarray(graph.node_ids)),
        ("source", graph.source),
        ("target", graph.target),
        ("weight", graph.weight),
        ("provenance", np.asarray(graph.provenance)),
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, array in arrays:
            with archive.open(f"{name}.npy", "w") as stream:
                write_array(stream, array, allow_pickle=False)


def load_graph(path: Path) -> Graph:
    """Parse a saved graph and reject malformed data."""
    with path.open("rb") as stream:
        archive: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with archive as data:
            if data["source"].dtype != np.int64 or data["target"].dtype != np.int64:
                message = "Edge indices must be int64"
                raise GraphError(message)
            node_ids = TypeAdapter(tuple[str, ...]).validate_python(
                data["node_ids"].tolist()
            )
            source = np.asarray(data["source"], dtype=np.int64)
            target = np.asarray(data["target"], dtype=np.int64)
            weight = np.asarray(data["weight"], dtype=np.float64)
            provenance = TypeAdapter(str).validate_python(
                data["provenance"].item(), strict=True
            )
        return Graph(node_ids, source, target, weight, provenance)


def shuffled_graph(graph: Graph, seed: int) -> Graph:
    """Rewire directed edges while preserving each node's in/out degree."""
    rng = np.random.default_rng(seed)
    source = TypeAdapter(list[int]).validate_python(graph.source.tolist(), strict=True)
    target = TypeAdapter(list[int]).validate_python(graph.target.tolist(), strict=True)
    weight = TypeAdapter(list[float]).validate_python(
        graph.weight.tolist(), strict=True
    )
    edges = set(zip(source, target, strict=True))
    accepted = 0
    # Fixed proposal budget, not a convergence or uniform-mixing guarantee.
    for _ in range(20 * len(target)):
        first, second = int(rng.integers(len(target))), int(rng.integers(len(target)))
        a, b = source[first], target[first]
        c, d = source[second], target[second]
        distinct_nodes = {a, b, c, d}
        if (
            len(distinct_nodes) != len((a, b, c, d))
            or (a, d) in edges
            or (c, b) in edges
            or (weight[first] > 0) != (weight[second] > 0)
        ):
            continue
        edges.remove((a, b))
        edges.remove((c, d))
        edges.update(((a, d), (c, b)))
        target[first], target[second] = d, b
        accepted += 1
    return Graph(
        graph.node_ids,
        graph.source.copy(),
        np.asarray(target, dtype=np.int64),
        graph.weight.copy(),
        f"{graph.provenance}; shuffle_seed={seed}; accepted_swaps={accepted}",
    )
