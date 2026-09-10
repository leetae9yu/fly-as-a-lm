# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2,<3", "pydantic>=2.10,<3", "typer>=0.15,<1", "pyarrow==21.0.0",
# ]
# ///
# From project root, without changing the shared environment:
# uv run --no-sync --with pyarrow==21.0.0 python -m scripts.prepare_large_connectome
"""Prepare verified full retained MaleCNS anatomy and nested connected subsets."""

from pathlib import Path
from typing import Annotated, Final

import numpy as np
import typer

from flyrl.connectome import GraphError
from scripts.connectome_arrays import connected_order, export_graph, induced_subgraph
from scripts.connectome_feather import ImportResult, read_connectome
from scripts.connectome_manifest import Artifact, Manifest, describe_artifact
from scripts.connectome_source import PINS, file_digest, obtain_sources

DEFAULT_OUTPUT: Final = Path("data/large_connectome")
SIZES: Final = (256, 1024, 4096, 16384)


def verify_release_totals(imported: ImportResult) -> None:
    """Independently reproduce the reference import's raw and retained totals."""
    actual = (
        imported.neurons.annotation_rows,
        imported.neurons.glia_rows,
        imported.neurons.unresolved_rows,
        imported.anatomy.ids.size,
        imported.source_edge_rows,
        imported.source_synaptic_contacts,
        imported.retained_edge_rows,
        imported.anatomy.edges.count.size,
        int(imported.anatomy.edges.count.sum()),
    )
    expected = (
        211577,
        11864,
        33013,
        166700,
        151856684,
        311833243,
        25582938,
        25582938,
        124177617,
    )
    if actual != expected:
        message = f"Pinned MaleCNS release totals differ: {actual} != {expected}"
        raise GraphError(message)


def prepare(output: Path, *, download: bool = True) -> Manifest:
    """Verify source hashes, parse all batches, and write reproducible artifacts."""
    obtain_sources(output / "raw", download=download)
    typer.echo("Source SHA256 pins verified; parsing all released edge batches.")
    imported = read_connectome(output / "raw")
    verify_release_totals(imported)
    full = imported.anatomy
    typer.echo(
        f"Retained {full.ids.size} neurons / {full.edges.count.size} directed pairs."
    )
    order = connected_order(full, max(SIZES))
    np.save(output / "selection_order.npy", full.ids[order], allow_pickle=False)
    provenance = (
        "MaleCNS-v1.0; brain-and-ventral-nerve-cord; "
        f"source_edges_sha256={PINS['edges.feather'].sha256}; "
        "body_pre->body_post; synapse_count=anatomical-contacts; "
        "weight=log1p(contacts)-all-positive-model-assumption; CC-BY-4.0"
    )
    artifacts: list[Artifact] = []
    for size in SIZES:
        subset = induced_subgraph(full, order[:size])
        path = output / f"malecns_v1_n{size}.npz"
        export_graph(subset, path, provenance + f"; induced-hub-BFS-n={size}")
        artifact = describe_artifact(path, subset, imported)
        artifacts.append(artifact)
        typer.echo(artifact.model_dump_json())
    full_path = output / "malecns_v1_full.npz"
    export_graph(full, full_path, provenance + "; full-retained-neuronal-graph")
    artifacts.append(describe_artifact(full_path, full, imported))
    manifest = Manifest(
        source_annotation_rows=imported.neurons.annotation_rows,
        excluded_glia=imported.neurons.glia_rows,
        excluded_unresolved_objects=imported.neurons.unresolved_rows,
        retained_annotation_fraction=full.ids.size / imported.neurons.annotation_rows,
        source_edge_rows=imported.source_edge_rows,
        source_synaptic_contacts=imported.source_synaptic_contacts,
        retained_edge_rows=imported.retained_edge_rows,
        duplicate_retained_rows_aggregated=imported.retained_edge_rows
        - full.edges.count.size,
        excluded_edge_rows=imported.source_edge_rows - imported.retained_edge_rows,
        excluded_synaptic_contacts=imported.source_synaptic_contacts
        - int(full.edges.count.sum()),
        root_body_id=str(int(full.ids[order[0]])),
        selection_order_sha256=file_digest(output / "selection_order.npy"),
        artifacts=artifacts,
    )
    _ = (output / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def main(
    output: Annotated[
        Path, typer.Option(help="Source cache and graph output directory.")
    ] = DEFAULT_OUTPUT,
    *,
    download: Annotated[
        bool, typer.Option(help="Download missing pinned source files.")
    ] = True,
) -> None:
    """Import the public MaleCNS v1.0 release; no training runs on this host."""
    manifest = prepare(output, download=download)
    typer.echo(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    typer.run(main)
