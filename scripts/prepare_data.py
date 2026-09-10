# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "pydantic>=2.10,<3"]
# ///
# Run from the project root: uv run python -m scripts.prepare_data
"""Prepare the checksum-pinned larval mushroom-body graph."""

import hashlib
from collections import Counter
from pathlib import Path
from typing import ClassVar, Final

import numpy as np
from pydantic import BaseModel, ConfigDict

from flyrl.connectome import Graph, GraphError, from_adjacency, save_graph

COMMIT: Final = "ccf1458bd73b25cb1e7a779c098a54943647bfa1"
SOURCE: Final = (
    f"https://raw.githubusercontent.com/graspologic-org/graspologic/{COMMIT}"
    "/graspologic/datasets/drosophila"
)
CHECKSUMS: Final = {
    "left_adjacency.csv": (
        "f23ba630f10dff95164ddbbdccf7d4fe1ac01a9a8ed77db26069af4a6248a1cf"
    ),
    "left_cell_labels.csv": (
        "a733652a33d5dde5c7cbdc9d8c0e679680cd000dc11edf3ea762a4da43b764c9"
    ),
}


class Provenance(BaseModel):
    """Data origin and explicit modeling assumptions shipped with the graph."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    dataset: str = "Left Drosophila larval mushroom body, Eichler et al. (2017)"
    paper: str = "https://doi.org/10.1038/nature23455"
    source: str = SOURCE
    commit: str = COMMIT
    checksums: dict[str, str]
    neurons: int
    directed_edges: int
    cell_classes: dict[str, int]
    assumptions: tuple[str, ...] = (
        "Rows are presynaptic and columns postsynaptic.",
        "All nonzero entries, including observed self-connections, are retained.",
        "Weights are log1p(synapse count), not measured synaptic efficacy.",
        "No transmitter annotations: all positive signs are a modeling assumption.",
        "Node IDs identify source matrix rows, not original anatomical skeleton IDs.",
        "This is a larval subcircuit, not an adult or whole-brain simulation.",
    )


def prepare(root: Path) -> Graph:
    """Verify raw data and create the graph and provenance artifacts."""
    raw = root / "data" / "raw"
    for name, expected in CHECKSUMS.items():
        digest = hashlib.sha256((raw / name).read_bytes()).hexdigest()
        if digest != expected:
            message = f"Source checksum mismatch: {name}"
            raise GraphError(message)
    labels = (raw / "left_cell_labels.csv").read_text().split()
    rows = [
        [float(value) for value in line.split()]
        for line in (raw / "left_adjacency.csv").read_text().splitlines()
        if line.strip()
    ]
    graph = from_adjacency(
        np.asarray(rows, dtype=np.float64),
        tuple(f"left:{index}:{label}" for index, label in enumerate(labels)),
        f"Eichler2017-left-larva-MB; graspologic_commit={COMMIT}; all-positive-assumed",
    )
    save_graph(graph, root / "data" / "larva_left_mb.npz")
    metadata = Provenance(
        checksums=CHECKSUMS,
        neurons=len(graph.node_ids),
        directed_edges=graph.weight.size,
        cell_classes=dict(Counter(labels)),
    )
    _ = (root / "data" / "provenance.json").write_text(
        metadata.model_dump_json(indent=2) + "\n"
    )
    return graph


if __name__ == "__main__":
    _ = prepare(Path(__file__).resolve().parents[1])
