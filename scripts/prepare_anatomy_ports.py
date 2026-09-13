# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2,<3", "pydantic>=2.10,<3", "typer>=0.15,<1", "pyarrow==21.0.0",
# ]
# ///
# Run from the project root:
# uv run --no-sync --with pyarrow==21.0.0
#   python -m scripts.prepare_anatomy_ports
"""Freeze exact ALPN input and MBON readout indices for central N5,600."""

import sys
from pathlib import Path
from typing import Annotated, Final

import numpy as np
import pyarrow as pa
import typer
from numpy.lib.npyio import NpzFile
from pyarrow import ipc
from pydantic import TypeAdapter

from flyrl.connectome import GraphError
from scripts.anatomy_port_artifacts import (
    AnatomyPortManifest,
    PortArtifactPlan,
    align_classes,
    write_port_artifacts,
)
from scripts.anatomy_ports import anatomy_port_audit
from scripts.connectome_arrays import Anatomy, Edges, IdArray, exact_ids
from scripts.connectome_source import PINS, file_digest, verify_source

DEFAULT_GRAPH: Final = Path("data/central_connectome/malecns_v1_cb_intrinsic_n5600.npz")
DEFAULT_ANNOTATIONS: Final = Path("data/large_connectome/raw/annotations.feather")
DEFAULT_OUTPUT: Final = Path("data/central_connectome")
GRAPH_SHA256: Final = "738d23289b7345dcf49a25e9305d106e7349fca87b93f97df8698db2753ce935"
EXPECTED_COUNTS: Final = (313, 97, 4064)


def load_anatomy(path: Path) -> Anatomy:
    """Load exact stable IDs, directed pairs and contact counts from a graph NPZ."""
    with path.open("rb") as stream, NpzFile(stream, allow_pickle=False) as data:
        node_ids = TypeAdapter(tuple[str, ...]).validate_python(
            data["node_ids"].tolist()
        )
        ids = np.fromiter(
            (int(node_id.rsplit(":", 1)[1]) for node_id in node_ids),
            dtype=np.uint64,
            count=len(node_ids),
        )
        edges = Edges(
            np.asarray(data["source"], dtype=np.int64),
            np.asarray(data["target"], dtype=np.int64),
            np.asarray(data["synapse_count"], dtype=np.int64),
        )
    return Anatomy(ids, edges)


def load_annotation_classes(path: Path) -> tuple[IdArray, tuple[str, ...]]:
    """Read exact body IDs and publisher `class` labels from the pinned Feather."""
    with pa.memory_map(str(path), "r") as mapped:
        table = ipc.open_file(mapped).read_all()
    ids = exact_ids(np.asarray(table.column("bodyId").to_numpy()))
    values = TypeAdapter(list[str | None]).validate_python(
        table.column("class").to_pylist()
    )
    return ids, tuple("" if value is None else value for value in values)


def prepare(graph_path: Path, annotations: Path, output: Path) -> AnatomyPortManifest:
    """Verify source identities, audit pathways and freeze exact port arrays."""
    if file_digest(graph_path) != GRAPH_SHA256:
        message = "Central N5600 graph identity mismatch"
        raise GraphError(message)
    verify_source(annotations, PINS["annotations.feather"])
    graph = load_anatomy(graph_path)
    annotation_ids, annotation_classes = load_annotation_classes(annotations)
    classes = align_classes(graph.ids, annotation_ids, annotation_classes)
    audit = anatomy_port_audit(graph, classes)
    counts = (
        audit.alpn_indices.size,
        audit.mbon_indices.size,
        audit.kenyon_indices.size,
    )
    if counts != EXPECTED_COUNTS or audit.reachable_mbons != audit.mbon_indices.size:
        message = "Central N5600 port classes or directed reachability changed"
        raise GraphError(message)
    return write_port_artifacts(
        graph,
        audit,
        PortArtifactPlan(
            output=output,
            graph_sha256=GRAPH_SHA256,
            annotations_sha256=PINS["annotations.feather"].sha256,
        ),
    )


def main(
    graph: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Central N5600 graph.")
    ] = DEFAULT_GRAPH,
    annotations: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Pinned annotations.")
    ] = DEFAULT_ANNOTATIONS,
    output: Annotated[
        Path, typer.Option(file_okay=False, help="Port artifact directory.")
    ] = DEFAULT_OUTPUT,
) -> None:
    """Prepare the anatomy-aware port manifest without training."""
    result = prepare(graph, annotations, output)
    _ = sys.stdout.write(result.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    typer.run(main)
