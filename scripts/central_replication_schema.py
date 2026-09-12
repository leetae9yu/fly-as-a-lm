"""Shared immutable protocol and prospective central replication decision."""

import math
import statistics
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict


class Pair(BaseModel):
    """One seed and its counterbalanced condition order."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    seed: int
    order: tuple[Literal["real", "shuffled"], Literal["real", "shuffled"]]


class Protocol(BaseModel):
    """Complete prospective identity and decision contract."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    format: Literal["flyrl-central-replication-v1"] = "flyrl-central-replication-v1"
    graph: str
    graph_sha256: str
    graph_nodes: int
    graph_edges: int
    corpus_fingerprint: str
    corpus_sha256: str
    source_sha256: str
    worker_sha256: str
    pairs: tuple[Pair, ...]
    context: int = 64
    batch_size: int = 8
    updates: int = 1000
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    leak: float = 0.5
    initial_gain: float = 0.9
    readout_neurons: int = 256
    trainable_codes: bool = True
    checkpoint_steps: int = 100
    sample_length: int = 64
    activation_neurons: int = 64
    prompt: str = "Once upon a time"
    median_advancement_threshold: float = 0.10
    heldout_policy: str = "sealed until all twelve conditions complete"


class ReplicationRow(BaseModel):
    """One fresh seed's paired held-out outcome."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    seed: int
    real_test_nll: float
    shuffled_test_nll: float
    delta: float
    real_accuracy: float
    shuffled_accuracy: float


class ReplicationAnalysis(BaseModel):
    """Prospective advancement rule evaluated over all fresh seeds."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    verdict: Literal["advance", "reversed", "inconclusive"]
    advance: bool
    unanimous_positive: bool
    median_delta: float
    mean_delta: float
    range: tuple[float, float]
    geometric_mean_ppl_ratio_real_over_shuffled: float
    two_sided_unanimous_sign_p: float | None


class ReplicationSummary(BaseModel):
    """Portable rows and decision produced only after the sealed matrix."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    format: Literal["flyrl-central-replication-results-v1"]
    rows: tuple[ReplicationRow, ...]
    analysis: ReplicationAnalysis


def evaluate(rows: tuple[ReplicationRow, ...], threshold: float) -> ReplicationSummary:
    """Apply the frozen unanimous-sign and practical-median decision."""
    deltas = [row.delta for row in rows]
    positive = all(delta > 0 for delta in deltas)
    negative = all(delta < 0 for delta in deltas)
    median, mean = statistics.median(deltas), statistics.mean(deltas)
    advance = positive and median >= threshold
    if advance:
        verdict = "advance"
    elif negative:
        verdict = "reversed"
    else:
        verdict = "inconclusive"
    return ReplicationSummary(
        format="flyrl-central-replication-results-v1",
        rows=rows,
        analysis=ReplicationAnalysis(
            verdict=verdict,
            advance=advance,
            unanimous_positive=positive,
            median_delta=median,
            mean_delta=mean,
            range=(min(deltas), max(deltas)),
            geometric_mean_ppl_ratio_real_over_shuffled=math.exp(-mean),
            two_sided_unanimous_sign_p=(0.03125 if positive or negative else None),
        ),
    )
