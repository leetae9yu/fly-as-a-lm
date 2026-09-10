"""Validated boundaries and machine-readable metrics for sparse likelihood learning."""

from typing import Annotated, Literal

from pydantic import Field

from flyrl.language_models import Settings


class ARConfig(Settings):
    """Update-independent configuration; checkpoints require exact equality."""

    alphabet_size: Annotated[int, Field(ge=2, le=48)]
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 0
    device: str = "cpu"
    control: Literal["real", "shuffled", "frozen"] = "real"
    context: Annotated[int, Field(ge=1)] = 32
    batch_size: Annotated[int, Field(ge=1)] = 8
    learning_rate: Annotated[float, Field(gt=0, le=1)] = 0.003
    weight_decay: Annotated[float, Field(ge=0)] = 0.0
    gradient_clip: Annotated[float, Field(gt=0)] = 1.0
    leak: Annotated[float, Field(gt=0, le=1)] = 0.5
    initial_gain: Annotated[float, Field(gt=0, le=1)] = 0.9
    readout_neurons: Annotated[int, Field(ge=1, le=1024)] = 256
    edge_chunk: Annotated[int, Field(ge=1)] = 65536
    eval_windows: Annotated[int, Field(ge=1)] = 512
    sample_length: Annotated[int, Field(ge=0)] = 120


class ARMetrics(Settings):
    """Final-target metrics on exactly the baseline's disjoint windows."""

    windows: int
    greedy_accuracy: float
    nll: float
    bits_per_character: float


class TraceEntry(Settings):
    """All-position training loss before an optimizer update."""

    update: int
    nll: float
    gradient_norm: float
