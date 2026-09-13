"""Checksum-bound anatomy port artifacts and factorial condition selection."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, assert_never
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from pydantic import BaseModel, ConfigDict, TypeAdapter

from flyrl.ar_config import PortPolicy
from flyrl.connectome import GraphError
from scripts.anatomy_ports import AnatomyPortAudit, ConnectionCount
from scripts.connectome_arrays import Anatomy, IdArray


@dataclass(frozen=True, slots=True)
class PortArtifactPlan:
    """Output location and immutable source identities for one port artifact."""

    output: Path
    graph_sha256: str
    annotations_sha256: str


class AnatomyPortManifest(BaseModel):
    """Portable class, connectivity and array identities for explicit ports."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    format: Literal["flyrl-anatomy-ports-v1"] = "flyrl-anatomy-ports-v1"
    graph_sha256: str
    annotations_sha256: str
    graph_nodes: int
    graph_edges: int
    graph_contacts: int
    class_column: Literal["class"] = "class"
    alpn_count: int
    mbon_count: int
    kenyon_count: int
    alpn_to_kenyon: ConnectionCount
    kenyon_to_mbon: ConnectionCount
    alpn_to_mbon: ConnectionCount
    reachable_mbons: int
    arrays: str = "anatomy_ports.npz"
    arrays_sha256: str


@dataclass(frozen=True, slots=True)
class PortSelection:
    """One paired condition's exact graph-index input and readout ports."""

    policy: PortPolicy
    sensory_indices: tuple[int, ...]
    readout_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ConfiguredPorts:
    """Manifest-bound explicit ports ready for checkpoint configuration."""

    policy: PortPolicy
    manifest_sha256: str
    graph_sha256: str
    sensory_indices: tuple[int, ...]
    readout_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AnatomyIndices:
    """All checksum-bound publisher classes in graph-index order."""

    alpn: tuple[int, ...]
    mbon: tuple[int, ...]
    kenyon: tuple[int, ...]


def align_classes(
    graph_ids: IdArray,
    annotation_ids: IdArray,
    annotation_classes: tuple[str, ...],
) -> tuple[str, ...]:
    """Join exact body-ID classes into graph-index order."""
    if annotation_ids.size != len(annotation_classes):
        message = "Annotation classes must align with annotation IDs"
        raise GraphError(message)
    adapter = TypeAdapter(tuple[int, ...])
    annotation_tuple = adapter.validate_python(annotation_ids.tolist())
    graph_tuple = adapter.validate_python(graph_ids.tolist())
    if len(set(annotation_tuple)) != len(annotation_tuple):
        message = "Annotation IDs must be unique"
        raise GraphError(message)
    classes_by_id = dict(zip(annotation_tuple, annotation_classes, strict=True))
    if any(body_id not in classes_by_id for body_id in graph_tuple):
        message = "Graph contains body IDs absent from annotations"
        raise GraphError(message)
    return tuple(classes_by_id[body_id] for body_id in graph_tuple)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_port_artifacts(
    graph: Anatomy,
    audit: AnatomyPortAudit,
    plan: PortArtifactPlan,
) -> AnatomyPortManifest:
    """Freeze exact port indices and audited pathway evidence."""
    plan.output.mkdir(parents=True, exist_ok=True)
    arrays = plan.output / "anatomy_ports.npz"
    with ZipFile(arrays, "w", compression=ZIP_DEFLATED) as archive:
        for name, values in (
            ("alpn_indices", audit.alpn_indices),
            ("mbon_indices", audit.mbon_indices),
            ("kenyon_indices", audit.kenyon_indices),
        ):
            with archive.open(f"{name}.npy", "w") as member:
                write_array(member, values, allow_pickle=False)
    manifest = AnatomyPortManifest(
        graph_sha256=plan.graph_sha256,
        annotations_sha256=plan.annotations_sha256,
        graph_nodes=graph.ids.size,
        graph_edges=graph.edges.count.size,
        graph_contacts=int(graph.edges.count.sum(dtype=np.int64)),
        alpn_count=audit.alpn_indices.size,
        mbon_count=audit.mbon_indices.size,
        kenyon_count=audit.kenyon_indices.size,
        alpn_to_kenyon=audit.alpn_to_kenyon,
        kenyon_to_mbon=audit.kenyon_to_mbon,
        alpn_to_mbon=audit.alpn_to_mbon,
        reachable_mbons=audit.reachable_mbons,
        arrays_sha256=_file_digest(arrays),
    )
    _ = (plan.output / "anatomy_ports.json").write_text(
        manifest.model_dump_json(indent=2) + "\n"
    )
    return manifest


def load_port_indices(
    path: Path,
    manifest: AnatomyPortManifest,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Load checksum-bound sensory/readout indices from a trusted manifest."""
    anatomy = load_anatomy_indices(path, manifest)
    return anatomy.alpn, anatomy.mbon


def load_anatomy_indices(
    path: Path,
    manifest: AnatomyPortManifest,
) -> AnatomyIndices:
    """Load and validate all exact publisher-class index arrays."""
    if _file_digest(path) != manifest.arrays_sha256:
        message = "Anatomy port array identity mismatch"
        raise GraphError(message)
    with path.open("rb") as stream, NpzFile(stream, allow_pickle=False) as data:
        if set(data.files) != {"alpn_indices", "mbon_indices", "kenyon_indices"}:
            message = "Anatomy port array key mismatch"
            raise GraphError(message)
        alpn = np.asarray(data["alpn_indices"], dtype=np.int64)
        mbon = np.asarray(data["mbon_indices"], dtype=np.int64)
        kenyon = np.asarray(data["kenyon_indices"], dtype=np.int64)
        adapter = TypeAdapter(tuple[int, ...])
        alpn_tuple = adapter.validate_python(alpn.tolist())
        mbon_tuple = adapter.validate_python(mbon.tolist())
        kenyon_tuple = adapter.validate_python(kenyon.tolist())
        if (
            data["alpn_indices"].dtype != np.int64
            or data["mbon_indices"].dtype != np.int64
            or data["kenyon_indices"].dtype != np.int64
            or alpn.shape != (manifest.alpn_count,)
            or mbon.shape != (manifest.mbon_count,)
            or kenyon.shape != (manifest.kenyon_count,)
            or len(set(alpn_tuple)) != len(alpn_tuple)
            or len(set(mbon_tuple)) != len(mbon_tuple)
            or len(set(kenyon_tuple)) != len(kenyon_tuple)
            or len(set(alpn_tuple) | set(mbon_tuple) | set(kenyon_tuple))
            != manifest.alpn_count + manifest.mbon_count + manifest.kenyon_count
            or bool((alpn < 0).any())
            or bool((mbon < 0).any())
            or bool((kenyon < 0).any())
            or bool((alpn >= manifest.graph_nodes).any())
            or bool((mbon >= manifest.graph_nodes).any())
            or bool((kenyon >= manifest.graph_nodes).any())
        ):
            message = "Anatomy class index invariant violation"
            raise GraphError(message)
    return AnatomyIndices(alpn_tuple, mbon_tuple, kenyon_tuple)


def condition_port_indices(
    manifest: AnatomyPortManifest,
    directory: Path,
    policy: PortPolicy,
    seed: int,
) -> PortSelection:
    """Resolve one anatomy/random factorial policy with a private seeded draw."""
    alpn, mbon = load_port_indices(directory / manifest.arrays, manifest)
    excluded = set(alpn) | set(mbon)
    candidates = np.asarray(
        [index for index in range(manifest.graph_nodes) if index not in excluded],
        dtype=np.int64,
    )
    needed = manifest.alpn_count + manifest.mbon_count
    if candidates.size < needed:
        message = "Graph has too few non-anatomy nodes for random matched ports"
        raise GraphError(message)
    np.random.default_rng(seed + 314159).shuffle(candidates)
    adapter = TypeAdapter(tuple[int, ...])
    random_sensory = tuple(
        sorted(adapter.validate_python(candidates[: manifest.alpn_count].tolist()))
    )
    random_readout = tuple(
        sorted(
            adapter.validate_python(candidates[manifest.alpn_count : needed].tolist())
        )
    )
    match policy:
        case "alpn_mbon":
            sensory, readout = alpn, mbon
        case "alpn_random":
            sensory, readout = alpn, random_readout
        case "random_mbon":
            sensory, readout = random_sensory, mbon
        case "random_random":
            sensory, readout = random_sensory, random_readout
        case "legacy_random":
            message = "Factorial port selection excludes the legacy random policy"
            raise GraphError(message)
        case _:
            assert_never(policy)
    return PortSelection(policy, sensory, readout)


def load_condition_ports(
    manifest_path: Path,
    policy: PortPolicy,
    seed: int,
) -> ConfiguredPorts:
    """Parse one manifest and bind a seeded condition to its exact file identity."""
    manifest = AnatomyPortManifest.model_validate_json(manifest_path.read_text())
    selection = condition_port_indices(manifest, manifest_path.parent, policy, seed)
    return ConfiguredPorts(
        policy=selection.policy,
        manifest_sha256=_file_digest(manifest_path),
        graph_sha256=manifest.graph_sha256,
        sensory_indices=selection.sensory_indices,
        readout_indices=selection.readout_indices,
    )
