"""Scientific provenance and loss-accounting schema for MaleCNS artifacts."""

from collections import Counter
from pathlib import Path
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from scripts.connectome_arrays import Anatomy, IntArray, incident_contacts
from scripts.connectome_feather import ImportResult
from scripts.connectome_source import COMMIT, PINS, SourcePin, file_digest


class Artifact(BaseModel):
    """An exported graph's identity, anatomical size, and retention fractions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    path: str
    bytes: int
    sha256: str
    neurons: int
    directed_edges: int
    synaptic_contacts: int
    isolated_neurons: int
    self_edges: int
    retained_neuron_fraction: float
    retained_edge_fraction: float
    retained_contact_fraction: float
    source_edge_row_fraction: float
    superclass_counts: dict[str, int]


class Manifest(BaseModel):
    """Release identity is shared by full graph and explicitly biased subsets."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    dataset: str = "MaleCNS v1.0"
    coverage: str = "brain_and_ventral_nerve_cord"
    whole_retained_graph_feasible: bool = True
    source: str = "https://male-cns.janelia.org/download/"
    paper: str = "https://doi.org/10.1016/j.cell.2026.08.015"
    attribution: str = "MaleCNS Consortium / HHMI Janelia Research Campus, MaleCNS v1.0"
    license: str = "CC-BY-4.0"
    license_url: str = "https://creativecommons.org/licenses/by/4.0/"
    reference_repository: str = "https://github.com/nftechie/doomfly"
    reference_commit: str = COMMIT
    sources: dict[str, SourcePin] = PINS
    source_annotation_rows: int
    excluded_glia: int
    excluded_unresolved_objects: int
    retained_annotation_fraction: float
    source_edge_rows: int
    source_synaptic_contacts: int
    retained_edge_rows: int
    duplicate_retained_rows_aggregated: int
    excluded_edge_rows: int
    excluded_synaptic_contacts: int
    root_body_id: str
    selection_order_path: str = "selection_order.npy"
    selection_order_sha256: str
    artifacts: list[Artifact]
    assumptions: tuple[str, ...] = (
        (
            "Full means all annotated neuronal candidates under the explicit "
            "node policy, "
            "not all segmentation objects and not a complete living fly simulation."
        ),
        (
            "Retain nonempty assigned superclass including tbc; exclude status Glia. "
            "No tracing-quality, cell-type, neuropil, or additional edge threshold."
        ),
        (
            "Released pre/post confidence threshold is 0.5. Preserve autapses and "
            "weight-one edges. Aggregate duplicate (body_pre, body_post) "
            "by integer sum."
        ),
        (
            "body_pre is source; body_post is target; weight in source Feather is "
            "anatomical synaptic contact count, not measured functional efficacy."
        ),
        (
            "NPZ synapse_count is exact int64 anatomy. NPZ weight is float64 "
            "log1p(synapse_count), an all-positive model initialization, not biology."
        ),
        (
            "No neurotransmitter signs inferred; neurotransmitter file is not needed "
            "or downloaded. These artifacts make no excitatory/inhibitory claims."
        ),
        (
            "IDs are malecns:v1.0:<exact decimal bodyId>, ascending numeric bodyId. "
            "Edge arrays are source/target int64, sorted by directed pair."
        ),
        (
            "Subsets are induced BFS prefixes on undirected adjacency rooted "
            "at maximum "
            "in+out contact count. Each expansion ranks neighbors by descending global "
            "contact count then ascending bodyId. Root ties choose smallest bodyId."
        ),
        (
            "Every subset is weakly connected; strong connectivity is not guaranteed. "
            "Selection favors a high-contact hub neighborhood and is not an unbiased "
            "sample, anatomical named region, or whole brain. Boundary edges are cut."
        ),
        (
            "Subset retained fractions use full retained graph as denominator; "
            "source_edge_row_fraction instead uses all released segmentation edge rows."
        ),
    )


def describe_artifact(path: Path, graph: Anatomy, imported: ImportResult) -> Artifact:
    """Compute evidence from the actual artifact and anatomical arrays."""
    full = imported.anatomy
    indices: IntArray = full.ids.searchsorted(graph.ids).astype(np.int64)
    classes = imported.neurons.superclasses
    return Artifact(
        path=path.name,
        bytes=path.stat().st_size,
        sha256=file_digest(path),
        neurons=graph.ids.size,
        directed_edges=graph.edges.count.size,
        synaptic_contacts=int(graph.edges.count.sum()),
        isolated_neurons=int(np.count_nonzero(np.equal(incident_contacts(graph), 0))),
        self_edges=int(
            np.count_nonzero(np.equal(graph.edges.source, graph.edges.target))
        ),
        retained_neuron_fraction=graph.ids.size / full.ids.size,
        retained_edge_fraction=graph.edges.count.size / full.edges.count.size,
        retained_contact_fraction=int(graph.edges.count.sum())
        / int(full.edges.count.sum()),
        source_edge_row_fraction=graph.edges.count.size / imported.source_edge_rows,
        superclass_counts=dict(Counter(classes[int(index)] for index in indices)),
    )
