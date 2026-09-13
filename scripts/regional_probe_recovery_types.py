"""Portable evidence emitted after strict regional-probe recovery."""

from flyrl.language_models import Settings
from scripts.regional_probe_artifact_types import ProbeRuntime
from scripts.regional_probe_protocol import ProbeSourceCheckpoint
from scripts.regional_probe_schema import RegionalProbeSummary


class RecoveredProbeSource(Settings):
    """One CRC-checked source result and all validated head hashes."""

    source: ProbeSourceCheckpoint
    result_archive_sha256: str
    runtime: ProbeRuntime
    heldout_sha256: str
    head_parameter_sha256: tuple[str, ...]


class RegionalProbeRecoveryReport(Settings):
    """Independent local decision and complete twelve-source evidence."""

    summary: RegionalProbeSummary
    sources: tuple[RecoveredProbeSource, ...]
