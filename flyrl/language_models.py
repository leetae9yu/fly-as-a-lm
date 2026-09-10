"""Validated character experiment settings and report values."""

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override


class Settings(BaseModel):
    """Immutable, strict experiment boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False
    )


class LanguageConfig(Settings):
    """One reproducible learning stream; update budget is deliberately external."""

    alphabet_size: Annotated[int, Field(ge=2, le=48)]
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 0
    device: str = "cpu"
    context: Annotated[int, Field(ge=1)] = 16
    batch_size: Annotated[int, Field(ge=1)] = 64
    learning_rate: Annotated[float, Field(gt=0, le=1)] = 0.1
    initial_gain: Annotated[float, Field(gt=0, le=4)] = 1.0
    max_weight: Annotated[float, Field(ge=4)] = 4.0
    baseline_rate: Annotated[float, Field(gt=0, le=1)] = 0.05
    frozen: bool = False
    eval_windows: Annotated[int, Field(ge=1)] = 512
    sample_length: Annotated[int, Field(ge=0)] = 120


@dataclass(frozen=True, slots=True)
class LanguageError(ValueError):
    """Malformed input or unavailable requested execution backend."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class Ports:
    """Edge-independent, disjoint neuron roles indexed by character identity."""

    sensory: tuple[int, ...]
    output: tuple[int, ...]


class Metrics(Settings):
    """Conditional policy metrics averaged over seeded stochastic hidden paths."""

    windows: int
    greedy_accuracy: float
    expected_reward: float
    sampled_reward: float
    nll: float
    bits_per_character: float


class DeviceEvidence(Settings):
    """Observed backend, not merely a requested device string."""

    requested: str
    tensor_device: str
    name: str
    torch_version: str
    cuda_version: str | None
    allocated_bytes: int
    peak_allocated_bytes: int
    deterministic_algorithms: bool
    matmul: Literal["dense masked source-target NxN float32"] = (
        "dense masked source-target NxN float32"
    )
