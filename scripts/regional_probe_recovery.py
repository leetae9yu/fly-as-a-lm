"""Strict one-source recovery for sealed regional probe artifacts."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from zipfile import ZipFile

import numpy as np

from flyrl.language_data import IntVector
from flyrl.regional_probe import feature_stats
from flyrl.regional_probe_types import ProbeMetrics
from flyrl.story_pilot import PilotReport
from scripts.anatomy_factorial_recovery_schema import WorkerRuntime
from scripts.connectome_source import file_digest
from scripts.regional_probe_artifact_types import SourceArtifact
from scripts.regional_probe_artifacts import (
    load_head,
    load_heldout,
)
from scripts.regional_probe_manifest import SeedGroupPlan
from scripts.regional_probe_protocol import (
    ProbeSourceCheckpoint,
    RegionalProbeProtocol,
)
from scripts.regional_probe_recovery_types import RecoveredProbeSource
from scripts.regional_probe_schema import ProbeBaselines, ProbeResult, ProbeScore
from scripts.regional_probe_source_worker import slice_cache

STATE_BOUND: Final = 1.000001


@dataclass(frozen=True, slots=True)
class SourceRecoveryInputs:
    """Files and identities needed to validate one source result."""

    directory: Path
    archive: Path
    source_archive: Path
    source: ProbeSourceCheckpoint
    plan: SeedGroupPlan
    protocol: RegionalProbeProtocol
    protocol_sha256: str
    valid_labels: IntVector
    test_labels: IntVector


def _score(metrics: ProbeMetrics) -> ProbeScore:
    return ProbeScore.model_validate(metrics.model_dump())


def recover_source(
    inputs: SourceRecoveryInputs,
) -> tuple[
    tuple[ProbeResult, ...],
    ProbeBaselines,
    RecoveredProbeSource,
]:
    """Validate one source's runtime, union, heads, traces and original report."""
    artifact = SourceArtifact.model_validate_json(
        (inputs.directory / "source.json").read_bytes()
    )
    expected_heads = tuple(
        f"{group.name}-{group.draw}.json" for group in inputs.plan.groups
    )
    runtime = artifact.runtime
    if (
        artifact.source != inputs.source
        or artifact.protocol_sha256 != inputs.protocol_sha256
        or artifact.heads != expected_heads
        or artifact.heldout_file != "heldout.npz"
        or runtime.gpu != "Tesla T4"
        or runtime.cuda is None
        or not runtime.device.startswith("cuda")
        or runtime.seconds <= 0
        or runtime.peak_allocated_bytes < 0
        or runtime.threads != 1
        or runtime.deterministic
        or runtime.tf32
    ):
        message = "Regional probe source artifact or T4 runtime differs"
        raise ValueError(message)
    valid, test = load_heldout(
        inputs.directory / artifact.heldout_file,
        artifact.union_indices,
        artifact.heldout_sha256,
    )
    if (
        valid.labels.tobytes() != inputs.valid_labels.tobytes()
        or test.labels.tobytes() != inputs.test_labels.tobytes()
        or bool((np.abs(valid.features) > STATE_BOUND).any())
        or bool((np.abs(test.features) > STATE_BOUND).any())
    ):
        message = "Regional probe heldout labels or state bounds differ"
        raise ValueError(message)
    expected_config = inputs.protocol.probe.model_copy(
        update={"seed": inputs.protocol.probe.seed + inputs.source.seed}
    )
    results: list[ProbeResult] = []
    parameter_hashes: list[str] = []
    for group, name in zip(inputs.plan.groups, artifact.heads, strict=True):
        head, result = load_head(inputs.directory / name)
        if (
            head.source != inputs.source
            or head.group != group
            or head.extraction != inputs.protocol.extraction
            or result.config != expected_config
            or result.valid.tokens != inputs.valid_labels.size
            or result.test.tokens != inputs.test_labels.size
            or feature_stats(slice_cache(valid, artifact.union_indices, group.indices))
            != result.features[1]
            or feature_stats(slice_cache(test, artifact.union_indices, group.indices))
            != result.features[2]
            or any(
                not (
                    math.isfinite(step.nll)
                    and math.isfinite(step.gradient_norm)
                    and math.isfinite(step.learning_rate)
                )
                for step in result.trace
            )
        ):
            message = "Regional probe head identity, features or trace differs"
            raise ValueError(message)
        results.append(
            ProbeResult(
                seed=inputs.source.seed,
                wiring=inputs.source.wiring,
                group=group.name,
                draw=group.draw,
                valid=_score(result.valid),
                test=_score(result.test),
            )
        )
        parameter_hashes.append(head.parameters_sha256)
    with ZipFile(inputs.source_archive) as archive:
        report = PilotReport.model_validate_json(archive.read("report.json"))
        source_runtime = WorkerRuntime.model_validate_json(archive.read("runtime.json"))
    if (
        file_digest(inputs.source_archive) != inputs.source.archive_sha256
        or report.final.test.nll != artifact.baselines.original_head_nll
        or report.trainable_parameters != artifact.source_parameter_count
        or source_runtime.torch != runtime.torch
        or source_runtime.cuda != runtime.cuda
    ):
        message = "Regional probe source checkpoint provenance differs"
        raise ValueError(message)
    recovery = RecoveredProbeSource(
        source=inputs.source,
        result_archive_sha256=file_digest(inputs.archive),
        runtime=runtime,
        heldout_sha256=artifact.heldout_sha256,
        head_parameter_sha256=tuple(parameter_hashes),
    )
    return tuple(results), artifact.baselines, recovery
