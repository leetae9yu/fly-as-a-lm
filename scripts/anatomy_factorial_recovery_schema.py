"""Portable recovery evidence for the sealed anatomy-port factorial."""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from flyrl.story_metrics import HeldoutMetrics
from scripts.anatomy_factorial_schema import (
    ConditionName,
    FactorialPortPolicy,
    FactorialSummary,
    Wiring,
)


class WorkerRuntime(BaseModel):
    """Exact runtime fields written beside one completed remote condition."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    seed: int
    condition: ConditionName
    wiring: Wiring
    port_policy: FactorialPortPolicy
    gpu: Literal["Tesla T4"]
    python: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    cuda: str = Field(min_length=1)
    seconds: float = Field(gt=0, allow_inf_nan=False)
    peak_allocated_bytes: int = Field(ge=0)
    resumed: bool


class CheckpointRuntime(BaseModel):
    """Execution flags that make an AR checkpoint exactly continuable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    torch: str = Field(min_length=1)
    device: str = Field(pattern=r"^cuda(?::\d+)?$")
    name: Literal["Tesla T4"]
    threads: Literal[1]
    deterministic: Literal[False]
    tf32: Literal[False]


class FactorialRecoveryEntry(BaseModel):
    """Verified checkpoint and report evidence for one condition."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    seed: int
    condition: ConditionName
    wiring: Wiring
    port_policy: FactorialPortPolicy
    initial: HeldoutMetrics
    final: HeldoutMetrics
    runtime: WorkerRuntime
    checkpoint_sha256: str


class FactorialRecoveryReport(BaseModel):
    """Complete local recovery evidence and independent factorial decision."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    summary: FactorialSummary
    entries: tuple[FactorialRecoveryEntry, ...]
