"""Validated experiment settings and machine-readable result schemas."""

from enum import StrEnum
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class Task(StrEnum):
    """Two small binary reward tasks, not language tasks."""

    ASSOCIATION = "association"
    MEMORY = "memory"


class Control(StrEnum):
    """Topology and plasticity interventions."""

    REAL = "real"
    SHUFFLED = "shuffled"
    FROZEN = "frozen"


class Settings(BaseModel):
    """Shared immutable validation policy for persisted settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False
    )


class Ports(Settings):
    """A clamped sensory neuron and a distinct fixed binary output neuron."""

    sensory: Annotated[int, Field(ge=0)]
    output: Annotated[int, Field(ge=0)]


class Config(Settings):
    """Complete per-run dynamics; episode budget is deliberately external."""

    task: Task = Task.ASSOCIATION
    control: Control = Control.REAL
    seed: Annotated[int, Field(ge=0)] = 0
    delay: Annotated[int, Field(ge=1)] = 3
    learning_rate: Annotated[float, Field(gt=0, le=1)] = 0.03
    baseline_rate: Annotated[float, Field(gt=0, le=1)] = 0.05
    initial_gain: Annotated[float, Field(gt=0, le=4)] = 1.5
    bias: Annotated[float, Field(ge=-20, le=20)] = -1.5
    max_weight: Annotated[float, Field(ge=4, le=100)] = 4.0
    eval_trials: Annotated[int, Field(ge=2)] = 512


class Evaluation(Settings):
    """Reproducible held-out stochastic-policy accuracy."""

    correct: int
    trials: int
    accuracy: float
    per_cue: tuple[float, float]


class RunResult(Settings):
    """Single-seed before/after result and checkpoint location."""

    config: Config
    ports: Ports
    episodes: int
    initial: Evaluation
    final: Evaluation
    training_accuracy: float
    weight_change_l2: float
    checkpoint: str
    direct_edge_present: bool
