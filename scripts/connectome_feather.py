"""Batchwise parsing of the pinned MaleCNS Feather release (optional pyarrow)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import ipc
from pydantic import TypeAdapter

from flyrl.connectome import GraphError
from scripts.connectome_arrays import (
    Anatomy,
    Edges,
    IdArray,
    IntArray,
    aggregate_edges,
    exact_ids,
    index_edges,
)


@dataclass(frozen=True, slots=True)
class Neurons:
    """Retained sorted neurons and source annotation accounting."""

    ids: IdArray
    superclasses: tuple[str, ...]
    annotation_rows: int
    glia_rows: int
    unresolved_rows: int


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Anatomy and loss accounting across every released edge row."""

    anatomy: Anatomy
    neurons: Neurons
    source_edge_rows: int
    source_synaptic_contacts: int
    retained_edge_rows: int


def read_neurons(path: Path) -> Neurons:
    """Retain assigned neuronal superclasses, including tbc; exclude explicit Glia."""
    with pa.memory_map(str(path), "r") as mapped:
        table = ipc.open_file(mapped).read_all()
    ids = exact_ids(np.asarray(table.column("bodyId").to_numpy()))
    if np.unique(ids).size != ids.size:
        message = "Duplicate body IDs in source annotations"
        raise GraphError(message)
    strings = TypeAdapter(list[str | None])
    classes = strings.validate_python(table.column("superclass").to_pylist())
    statuses = strings.validate_python(table.column("status").to_pylist())
    keep = np.array(
        [
            bool(label) and status != "Glia"
            for label, status in zip(classes, statuses, strict=True)
        ]
    )
    indices: IntArray = keep.nonzero()[0].astype(np.int64)
    order: IntArray = indices[ids[indices].argsort(kind="stable")]
    glia = statuses.count("Glia")
    return Neurons(
        ids[order],
        tuple(str(classes[int(index)]) for index in order),
        ids.size,
        glia,
        ids.size - int(np.count_nonzero(keep)) - glia,
    )


def read_connectome(raw: Path) -> ImportResult:
    """Stream 65K-row Arrow batches; retain O(E) numeric arrays, not edge objects."""
    pa.set_cpu_count(4)
    neurons = read_neurons(raw / "annotations.feather")
    sources: list[IntArray] = []
    targets: list[IntArray] = []
    counts: list[IntArray] = []
    source_rows, source_contacts, retained_rows = 0, 0, 0
    with pa.memory_map(str(raw / "edges.feather"), "r") as mapped:
        reader = ipc.open_file(mapped)
        for number in range(reader.num_record_batches):
            batch = reader.get_batch(number)
            edges = Edges(
                np.asarray(batch.column("body_pre").to_numpy(), dtype=np.int64),
                np.asarray(batch.column("body_post").to_numpy(), dtype=np.int64),
                np.asarray(batch.column("weight").to_numpy(), dtype=np.int64),
            )
            retained = index_edges(neurons.ids, edges)
            source_rows += edges.count.size
            source_contacts += int(edges.count.sum(dtype=np.int64))
            retained_rows += retained.count.size
            sources.append(retained.source)
            targets.append(retained.target)
            counts.append(retained.count)
    indexed = Edges(
        np.concatenate(sources), np.concatenate(targets), np.concatenate(counts)
    )
    del sources, targets, counts
    graph = aggregate_edges(neurons.ids, indexed)
    return ImportResult(graph, neurons, source_rows, source_contacts, retained_rows)
