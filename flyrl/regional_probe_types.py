"""Typed feature, fitting and evidence values for regional probes."""

from dataclasses import dataclass
from typing import Annotated, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from flyrl.language_data import IntVector
from flyrl.language_models import Settings


@dataclass(frozen=True, slots=True)
class FeatureCache:
    """CPU arrays in story-major, then increasing input-position order."""

    features: NDArray[np.float32]
    labels: IntVector


class ExtractionConfig(Settings):
    """Bound recurrent extraction memory without truncating story context."""

    chunk_size: Annotated[int, Field(ge=1)] = 128
    story_batch_size: Annotated[int, Field(ge=1)] = 1


class ProbeConfig(Settings):
    """Fixed updates; L2 is half the coefficient times summed squared weights."""

    vocab_size: Annotated[int, Field(ge=2)]
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 0
    updates: Annotated[int, Field(ge=0)] = 500
    batch_size: Annotated[int, Field(ge=1)] = 256
    learning_rate: Annotated[float, Field(gt=0)] = 0.01
    min_learning_rate: Annotated[float, Field(ge=0)] = 0.0001
    std_floor: Annotated[float, Field(gt=0)] = 1e-4
    l2: Annotated[float, Field(ge=0)] = 1e-4
    gradient_clip: Annotated[float, Field(gt=0)] = 1.0
    device: Annotated[str, Field(pattern=r"^cpu$|^cuda(?::\d+)?$")] = "cpu"

    @model_validator(mode="after")
    def learning_rate_bounds(self) -> Self:
        """Keep the cosine schedule nonincreasing and nonnegative."""
        if self.min_learning_rate > self.learning_rate:
            message = "Minimum learning rate cannot exceed its initial value"
            raise ValueError(message)
        return self


class ProbeParameters(Settings):
    """Serializable float32 values; logits use standardized features."""

    weight: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]


class ProbeStep(Settings):
    """Pre-update sampled NLL and unclipped norm, with applied learning rate."""

    update: int
    nll: float
    gradient_norm: float
    learning_rate: float


class ProbeMetrics(Settings):
    """All cached pairs, without L2 or subsampling."""

    tokens: int
    nll: float
    perplexity: float | None
    accuracy: float


class FeatureStats(Settings):
    """Raw variance, saturation and exact C-order hashes."""

    feature_sha256: str
    label_sha256: str
    variance: tuple[float, ...]
    saturation_fraction: float


class ProbeResult(Settings):
    """Final probe, never heldout-selected."""

    config: ProbeConfig
    parameters: ProbeParameters
    parameter_count: int
    trace: tuple[ProbeStep, ...]
    features: tuple[FeatureStats, FeatureStats, FeatureStats]
    valid: ProbeMetrics
    test: ProbeMetrics
