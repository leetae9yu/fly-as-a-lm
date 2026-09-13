"""Frozen source identities and execution settings for regional probes."""

from typing import ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from flyrl.regional_probe_types import ExtractionConfig, ProbeConfig

Wiring: TypeAlias = Literal["real", "shuffled"]


class ProbeSourceCheckpoint(BaseModel):
    """One exact complete source-condition archive."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    seed: int
    wiring: Wiring
    filename: str
    archive_sha256: str
    archive_bytes: int
    checkpoint_sha256: str
    report_sha256: str
    runtime_sha256: str
    parameter_fingerprint: str


class ProbeSourceManifest(BaseModel):
    """Validated source checkpoints before the new probe worker exists."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    format: Literal["flyrl-regional-probe-sources-v1"] = (
        "flyrl-regional-probe-sources-v1"
    )
    graph_sha256: str
    corpus_fingerprint: str
    sources: tuple[ProbeSourceCheckpoint, ...]


class RegionalProbeProtocol(BaseModel):
    """Complete source, fitting and decision contract sealed before execution."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    format: Literal["flyrl-regional-probe-v1"] = "flyrl-regional-probe-v1"
    graph: str
    graph_sha256: str
    graph_nodes: int
    graph_edges: int
    corpus: str
    corpus_sha256: str
    corpus_fingerprint: str
    port_manifest: str
    port_manifest_sha256: str
    port_arrays: str
    port_arrays_sha256: str
    group_manifest: str
    group_manifest_sha256: str
    source_sha256: str
    worker_sha256: str
    checkpoints: tuple[ProbeSourceCheckpoint, ...]
    extraction: ExtractionConfig
    probe: ProbeConfig
    group_size: int
    draws: int
    heads_per_checkpoint: int
    practical_threshold: float
    heldout_policy: str
