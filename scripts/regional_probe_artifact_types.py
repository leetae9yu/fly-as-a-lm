"""Strict compact schemas for independently recoverable regional-probe artifacts."""

from typing import Literal

from flyrl.language_models import Settings
from flyrl.regional_probe_types import (
    ExtractionConfig,
    FeatureStats,
    ProbeConfig,
    ProbeMetrics,
    ProbeStep,
)
from scripts.regional_probe_manifest import ProbeGroupRecord
from scripts.regional_probe_protocol import ProbeSourceCheckpoint
from scripts.regional_probe_schema import ProbeBaselines


class CompactProbeResult(Settings):
    """Probe evidence excluding the large tuple-encoded parameter payload."""

    config: ProbeConfig
    parameter_count: int
    trace: tuple[ProbeStep, ...]
    features: tuple[FeatureStats, FeatureStats, FeatureStats]
    valid: ProbeMetrics
    test: ProbeMetrics


class HeadArtifact(Settings):
    """One exact source/group fit with separately checksummed float32 parameters."""

    format: Literal["flyrl-regional-probe-head-v1"] = "flyrl-regional-probe-head-v1"
    source: ProbeSourceCheckpoint
    group: ProbeGroupRecord
    extraction: ExtractionConfig
    result: CompactProbeResult
    parameters_file: str
    parameters_sha256: str


class ProbeRuntime(Settings):
    """Actual source extraction and probe fitting runtime identity."""

    gpu: str
    python: str
    torch: str
    cuda: str | None
    device: str
    seconds: float
    peak_allocated_bytes: int
    threads: int
    deterministic: bool
    tf32: bool


class SourceArtifact(Settings):
    """Complete independently recoverable source output; heads follow manifest order."""

    format: Literal["flyrl-regional-probe-source-v1"] = "flyrl-regional-probe-source-v1"
    source: ProbeSourceCheckpoint
    protocol_sha256: str
    source_parameter_count: int
    runtime: ProbeRuntime
    union_indices: tuple[int, ...]
    heldout_file: str
    heldout_sha256: str
    heads: tuple[str, ...]
    baselines: ProbeBaselines
