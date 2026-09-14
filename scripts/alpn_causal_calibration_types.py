"""Strict metric-sealed calibration evidence; no fresh corpus or scoring schema."""

from hashlib import sha256
from math import isfinite
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from scripts.alpn_causal_groups import FloatMatrix, MatchingEvidence, Overlap
from scripts.regional_probe_protocol import ProbeSourceCheckpoint

SEEDS = tuple(range(7, 13))
Status = Literal["balanced", "insufficient_common_support"]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Seed = Annotated[int, Field(ge=7, le=12)]


class Record(BaseModel):
    """Reject coercion, unknown keys and nonfinite JSON throughout the evidence."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class Matrix(Record):
    """Explicit float64 C-order rows and their raw-byte SHA256."""

    rows: tuple[tuple[float, ...], ...]
    sha256: Digest

    @classmethod
    def capture(cls, values: FloatMatrix) -> Self:
        """Retain exact binary64 values through JSON roundtrips."""
        rows = TypeAdapter(tuple[tuple[float, ...], ...]).validate_python(
            values.tolist()
        )
        return cls(rows=rows, sha256=sha256(values.tobytes(order="C")).hexdigest())

    def array(self, rows: int, columns: int) -> FloatMatrix:
        """Authenticate dimensions, dtype and every coordinate byte."""
        values = np.asarray(self.rows, dtype=np.float64)
        if (
            values.shape != (rows, columns)
            or not bool(np.isfinite(values).all())
            or sha256(values.tobytes()).hexdigest() != self.sha256
        ):
            message = "Calibration matrix shape or hash differs"
            raise ValueError(message)
        return values


class CoordinateBalance(Record):
    """Unequal constants encode infinite SMD explicitly, never as JSON null."""

    coordinate: str
    smd: float | Literal["inf", "-inf"]
    ks: float = Field(ge=0, le=1)


class Control(Record):
    """Exact target-to-control correspondence and selected assignment costs."""

    comparator: Literal["S", "M"]
    draw: int = Field(ge=0, le=4)
    target_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    pair_costs: tuple[float, ...]
    total_cost: float = Field(ge=0)
    balance: tuple[CoordinateBalance, ...]


class MatchingRecord(Record):
    """Lossless typed view of the existing deterministic matcher."""

    eligible: tuple[int, ...]
    excluded: tuple[int, ...]
    required: int = Field(gt=0)
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    controls: tuple[Control, ...]
    overlaps: tuple[Overlap, ...]
    status: Status

    @classmethod
    def capture(cls, evidence: MatchingEvidence) -> Self:
        """Convert only non-JSON infinite SMDs; preserve all finite evidence."""
        return cls(
            eligible=evidence.eligible,
            excluded=evidence.excluded,
            required=evidence.required,
            mean=evidence.mean,
            scale=evidence.scale,
            overlaps=evidence.overlaps,
            status=evidence.status,
            controls=tuple(
                Control(
                    comparator=item.comparator,
                    draw=item.draw,
                    target_indices=item.target_indices,
                    control_indices=item.control_indices,
                    pair_costs=item.pair_costs,
                    total_cost=item.total_cost,
                    balance=tuple(
                        CoordinateBalance(
                            coordinate=b.coordinate,
                            ks=b.ks,
                            smd=b.smd
                            if isfinite(b.smd)
                            else "inf"
                            if b.smd > 0
                            else "-inf",
                        )
                        for b in item.balance
                    ),
                )
                for item in evidence.controls
            ),
        )


class FileIdentity(Record):
    """One pre-execution sealed relative path; content is not learned on recovery."""

    path: str
    sha256: Digest
    size: int = Field(ge=0)


class CalibrationSeal(Record):
    """Trusted local input seal supplied separately from the untrusted result ZIP."""

    format: Literal["flyrl-alpn-calibration-seal-v1"] = "flyrl-alpn-calibration-seal-v1"
    protocol_sha256: Digest
    files: tuple[FileIdentity, ...]
    training_tokens_sha256: Digest
    training_offsets_sha256: Digest


class Runtime(Record):
    """The frozen source runtime, not a substitute CPU calibration."""

    gpu: Literal["Tesla T4"]
    device: str = Field(pattern=r"^cuda(?::\d+)?$")
    dtype: Literal["torch.float32"]
    python: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    cuda: str = Field(min_length=1)
    threads: int = Field(ge=1, le=1)
    deterministic: bool
    tf32: bool

    @model_validator(mode="after")
    def flags(self) -> Self:
        """Require the frozen flags without integer/boolean literal coercion."""
        if self.deterministic or self.tf32:
            message = "Calibration runtime flags differ"
            raise ValueError(message)
        return self


class Snapshot(Record):
    """Parameters, all buffers/topology, optimizer, progress and RNG immutability."""

    parameters: Digest
    buffers: Digest
    optimizer: Digest
    progress: Digest
    rng: Digest


class SourceIdentity(ProbeSourceCheckpoint, Record):
    """Strict source checkpoint identity using the exact regional fields."""


class SourceEvidence(Record):
    """Both snapshots must agree; the parameter fingerprint must be the source's."""

    source: SourceIdentity
    runtime: Runtime
    before: Snapshot
    after: Snapshot

    @model_validator(mode="after")
    def unchanged(self) -> Self:
        """Do not serialize a merely asserted successful immutability check."""
        if (
            self.before != self.after
            or self.before.parameters != self.source.parameter_fingerprint
        ):
            message = "Calibration source immutability differs"
            raise ValueError(message)
        return self


class SeedCalibration(Record):
    """Selected-neuron activity and all sixteen paired coordinates for one seed."""

    seed: Seed
    indices: tuple[int, ...]
    positions: Annotated[int, Field(ge=229745, le=229745)]
    activity: tuple[Matrix, Matrix]
    coordinates: Matrix
    matching: MatchingRecord
    sources: tuple[SourceEvidence, SourceEvidence]
    seconds: float = Field(gt=0)


class CalibrationResult(Record):
    """Only an exact ordered six-seed matrix can have a terminal calibration status."""

    format: Literal["flyrl-alpn-calibration-result-v1"] = (
        "flyrl-alpn-calibration-result-v1"
    )
    seal: CalibrationSeal
    seeds: tuple[SeedCalibration, ...]
    status: Status

    @model_validator(mode="after")
    def complete(self) -> Self:
        """Reject missing, duplicated or reordered seeds and false verdicts."""
        status = (
            "balanced"
            if all(s.matching.status == "balanced" for s in self.seeds)
            else "insufficient_common_support"
        )
        if tuple(s.seed for s in self.seeds) != SEEDS or self.status != status:
            message = "Calibration completeness or aggregate status differs"
            raise ValueError(message)
        return self
