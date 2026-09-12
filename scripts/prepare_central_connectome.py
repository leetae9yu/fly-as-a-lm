# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2,<3", "pydantic>=2.10,<3", "typer>=0.15,<1", "pyarrow==21.0.0",
# ]
# ///
# From project root, without changing the shared environment:
# uv run --no-sync --with pyarrow==21.0.0 python -m scripts.prepare_central_connectome
"""Prepare induced cb_intrinsic MaleCNS graphs for matched comparisons."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal

import numpy as np
import typer
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from pydantic import BaseModel, ConfigDict, TypeAdapter

from flyrl.connectome import GraphError
from scripts.connectome_arrays import (
    Anatomy,
    Edges,
    IntArray,
    connected_order,
    export_graph,
    induced_subgraph,
)
from scripts.connectome_feather import Neurons, read_neurons
from scripts.connectome_manifest import Manifest
from scripts.connectome_source import PINS, file_digest, verify_source

DEFAULT_SOURCE: Final = Path("data/large_connectome")
DEFAULT_OUTPUT: Final = Path("data/central_connectome")
SIZES: Final = (5600, 16384)
FULL_GRAPH: Final = "malecns_v1_full.npz"
COMPARISON_GRAPH: Final = "malecns_v1_n16384.npz"


@dataclass(frozen=True, slots=True)
class CentralPlan:
    """Output identity and comparison budget for one deterministic preparation."""

    output: Path
    sizes: tuple[int, ...]
    comparison_edges: int
    source_graph_sha256: str
    annotations_sha256: str


@dataclass(frozen=True, slots=True)
class CentralSelection:
    """Connected cb_intrinsic prefixes and their shared discovery order."""

    pool: Anatomy
    order: IntArray
    graphs: tuple[Anatomy, ...]


class CentralArtifact(BaseModel):
    """One selected graph's portable identity and anatomical counts."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    path: str
    bytes: int
    sha256: str
    neurons: int
    directed_edges: int
    synaptic_contacts: int
    edge_delta: int


class CentralManifest(BaseModel):
    """Machine-readable selection policy, source identity and output evidence."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    format: Literal["flyrl-central-connectome-v1"] = "flyrl-central-connectome-v1"
    region_superclass: Literal["cb_intrinsic"] = "cb_intrinsic"
    source_graph_sha256: str
    annotations_sha256: str
    comparison_edges: int
    pool_nodes: int
    pool_edges: int
    pool_contacts: int
    root_body_id: str
    selection_order_path: str = "selection_order.npy"
    selection_order_sha256: str
    artifacts: tuple[CentralArtifact, ...]
    assumptions: tuple[str, ...] = (
        "Selection uses only the induced cb_intrinsic pool.",
        "Connected BFS ranks fresh neighbors by pool incident contacts then bodyId.",
        (
            "Each artifact retains every internal directed pair and original "
            "contact count."
        ),
        (
            "The 5600-node artifact matches edge count; the 16384-node artifact "
            "matches nodes."
        ),
        "Boundary edges and every non-cb_intrinsic neuron are excluded.",
    )


def central_intrinsic_subgraphs(
    graph: Anatomy,
    superclasses: tuple[str, ...],
    sizes: tuple[int, ...],
) -> CentralSelection:
    """Select nested connected prefixes inside the induced cb_intrinsic graph."""
    if len(superclasses) != graph.ids.size:
        message = "Superclass labels must align with graph IDs"
        raise GraphError(message)
    if not sizes or sizes != tuple(sorted(set(sizes))):
        message = "Central subset sizes must be nonempty, unique and ascending"
        raise GraphError(message)
    selected = np.asarray(
        [index for index, label in enumerate(superclasses) if label == "cb_intrinsic"],
        dtype=np.int64,
    )
    pool = induced_subgraph(graph, selected)
    order = connected_order(pool, sizes[-1])
    return CentralSelection(
        pool,
        order,
        tuple(induced_subgraph(pool, order[:size]) for size in sizes),
    )


def load_full_anatomy(path: Path, neurons: Neurons) -> Anatomy:
    """Parse exact counts from the verified full NPZ and align source annotations."""
    with path.open("rb") as stream:
        archive = NpzFile(stream, allow_pickle=False)
        with archive as data:
            node_ids = TypeAdapter(tuple[str, ...]).validate_python(
                data["node_ids"].tolist()
            )
            ids = np.fromiter(
                (int(node_id.rsplit(":", 1)[1]) for node_id in node_ids),
                dtype=np.uint64,
                count=len(node_ids),
            )
            source = np.asarray(data["source"], dtype=np.int64)
            target = np.asarray(data["target"], dtype=np.int64)
            counts = np.asarray(data["synapse_count"], dtype=np.int64)
            weights = np.asarray(data["weight"], dtype=np.float64)
    if ids.size != neurons.ids.size or ids.tobytes() != neurons.ids.tobytes():
        message = "Full graph IDs do not align with pinned annotations"
        raise GraphError(message)
    expected_weights = np.log1p(counts.astype(np.float64))
    if not (
        source.size == target.size == counts.size == weights.size
        and (counts > 0).all()
        and weights.tobytes() == expected_weights.tobytes()
    ):
        message = "Full graph edges or anatomical counts are inconsistent"
        raise GraphError(message)
    return Anatomy(ids, Edges(source, target, counts))


def prepare_central_graphs(
    graph: Anatomy,
    superclasses: tuple[str, ...],
    plan: CentralPlan,
) -> CentralManifest:
    """Write nested induced central graphs and their measured comparison manifest."""
    selection = central_intrinsic_subgraphs(graph, superclasses, plan.sizes)
    plan.output.mkdir(parents=True, exist_ok=True)
    order_path = plan.output / "selection_order.npy"
    with order_path.open("wb") as stream:
        write_array(stream, selection.pool.ids[selection.order], allow_pickle=False)
    root = str(int(selection.pool.ids[selection.order[0]]))
    artifacts: list[CentralArtifact] = []
    for size, subset in zip(plan.sizes, selection.graphs, strict=True):
        path = plan.output / f"malecns_v1_cb_intrinsic_n{size}.npz"
        provenance = (
            "MaleCNS-v1.0; region=cb_intrinsic; induced-region-pool; "
            "selection=connected-contact-ranked-BFS; "
            f"root_body_id={root}; n={size}; all-internal-directed-edges; "
            "boundary-edges-cut; synapse_count=anatomical-contacts; "
            "weight=log1p(contacts)-all-positive-model-assumption; "
            f"source_full_sha256={plan.source_graph_sha256}; CC-BY-4.0"
        )
        export_graph(subset, path, provenance)
        artifacts.append(
            CentralArtifact(
                path=path.name,
                bytes=path.stat().st_size,
                sha256=file_digest(path),
                neurons=subset.ids.size,
                directed_edges=subset.edges.count.size,
                synaptic_contacts=sum(memoryview(subset.edges.count)),
                edge_delta=subset.edges.count.size - plan.comparison_edges,
            )
        )
    manifest = CentralManifest(
        source_graph_sha256=plan.source_graph_sha256,
        annotations_sha256=plan.annotations_sha256,
        comparison_edges=plan.comparison_edges,
        pool_nodes=selection.pool.ids.size,
        pool_edges=selection.pool.edges.count.size,
        pool_contacts=sum(memoryview(selection.pool.edges.count)),
        root_body_id=root,
        selection_order_sha256=file_digest(order_path),
        artifacts=tuple(artifacts),
    )
    _ = (plan.output / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n"
    )
    return manifest


def prepare(source: Path, output: Path) -> CentralManifest:
    """Verify retained source identities, then prepare both fixed comparisons."""
    manifest = Manifest.model_validate_json((source / "manifest.json").read_text())
    full_artifact = next(
        artifact for artifact in manifest.artifacts if artifact.path == FULL_GRAPH
    )
    comparison = next(
        artifact for artifact in manifest.artifacts if artifact.path == COMPARISON_GRAPH
    )
    full_path = source / full_artifact.path
    if file_digest(full_path) != full_artifact.sha256:
        message = f"Full graph integrity mismatch: {full_path}"
        raise GraphError(message)
    annotation_path = source / "raw" / "annotations.feather"
    verify_source(annotation_path, PINS["annotations.feather"])
    neurons = read_neurons(annotation_path)
    graph = load_full_anatomy(full_path, neurons)
    if (
        graph.ids.size != full_artifact.neurons
        or graph.edges.count.size != full_artifact.directed_edges
    ):
        message = "Full graph counts differ from its verified manifest"
        raise GraphError(message)
    return prepare_central_graphs(
        graph,
        neurons.superclasses,
        CentralPlan(
            output=output,
            sizes=SIZES,
            comparison_edges=comparison.directed_edges,
            source_graph_sha256=full_artifact.sha256,
            annotations_sha256=PINS["annotations.feather"].sha256,
        ),
    )


def main(
    source: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Verified MaleCNS directory."),
    ] = DEFAULT_SOURCE,
    output: Annotated[
        Path, typer.Option(file_okay=False, help="Central graph output directory.")
    ] = DEFAULT_OUTPUT,
) -> None:
    """Create central-brain graph artifacts without training a model."""
    typer.echo(prepare(source, output).model_dump_json(indent=2))


if __name__ == "__main__":
    typer.run(main)
