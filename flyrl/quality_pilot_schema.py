"""Immutable protocol and additive evidence for the fixed-budget quality pilot."""

import hashlib
from enum import StrEnum
from math import cos, pi
from pathlib import Path
from typing import Annotated, Final, Literal, Self, TypeAlias

from pydantic import Field, model_validator

from flyrl.ar_config import ARConfig
from flyrl.language_models import Settings
from flyrl.quality_generation import QualityGenerationOptions, QualityGenerationRecord
from flyrl.quality_metrics import QualityMetrics

Positive: TypeAlias = Annotated[int, Field(gt=0)]
Digest: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PROMPTS: Final = (
    "Once upon a time",
    "Lily found a small red box under her bed.",
    "Ben wanted to fly his kite, but there was no wind.",
    "A little rabbit lost the key to her house.",
    '"Can I play with you?" asked Tom.',
    "Mia promised to look after her brother's toy.",
    "The rain stopped, and the children opened the door.",
    "A small bird was afraid to leave its nest.",
)
EVALUATIONS: Final = (0, 1000, 5000, 10000, 15000, 20000, 25000, 30000)


class ArtifactIdentity(Settings):
    """Exact graph file and logical graph identity, including original edge count."""

    path: Path
    sha256: Digest
    fingerprint: Digest
    nodes: Positive
    edges: Positive


class CorpusIdentity(Settings):
    """Exact corpus artifact with canonical train/validation/test counts."""

    path: Path
    sha256: Digest
    fingerprint: Digest
    stories: tuple[Positive, Positive, Positive]
    tokens: tuple[Positive, Positive, Positive]
    vocabulary: Annotated[int, Field(ge=257)]


class Schedule(Settings):
    """One-based linear warmup followed by cosine decay to the exact minimum."""

    target_updates: Positive = 30_000
    warmup_updates: Positive = 500
    peak: Annotated[float, Field(gt=0, le=1)] = 0.003
    minimum: Annotated[float, Field(gt=0, le=1)] = 0.0003

    @model_validator(mode="after")
    def valid_decay(self) -> Self:
        """Require a nonempty decay and nonincreasing post-warmup rate."""
        if self.warmup_updates >= self.target_updates or self.minimum > self.peak:
            message = "Schedule needs warmup < target and minimum <= peak"
            raise ValueError(message)
        return self

    def rate(self, update: int) -> float:
        """Return LR for the update about to execute, never for update zero."""
        if not 1 <= update <= self.target_updates:
            message = "Learning-rate update is outside the declared schedule"
            raise ValueError(message)
        if update <= self.warmup_updates:
            return self.peak * update / self.warmup_updates
        position = (update - self.warmup_updates) / (
            self.target_updates - self.warmup_updates
        )
        return self.minimum + (self.peak - self.minimum) * (1 + cos(pi * position)) / 2


class QualityProtocol(Settings):
    """JSON-only strict parsing binds every choice; smoke never earns quality success.

    Paths are interpreted from the invocation working directory. The canonical
    hash includes defaults, paths and every nested setting, ignoring JSON layout.
    """

    schema_version: Literal[1] = 1
    profile: Literal["quality", "smoke"] = "quality"
    graph: ArtifactIdentity
    corpus: CorpusIdentity
    config: ARConfig
    schedule: Schedule = Schedule()
    evaluation_updates: tuple[int, ...] = EVALUATIONS
    early_update: Positive = 1000
    checkpoint_interval: Positive = 1000
    cpu_threads: Annotated[int, Field(ge=1, le=4)] = 1
    prompts: tuple[Annotated[str, Field(min_length=1)], ...] = PROMPTS
    generation: QualityGenerationOptions = QualityGenerationOptions()
    activation_prompt_indices: tuple[int, ...] = (0, 3)

    @property
    def sha256(self) -> str:
        """Identify normalized immutable settings, not incidental whitespace."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    @model_validator(mode="after")
    def valid_panel(self) -> Self:
        """Reject ambiguous histories, empty panels and nonanatomical learners."""
        config, updates = self.config, self.evaluation_updates
        if (
            not updates
            or updates[0] != 0
            or updates[-1] != self.schedule.target_updates
            or tuple(sorted(set(updates))) != updates
            or self.early_update not in updates
            or not self.prompts
            or self.generation.length < 1
            or not self.activation_prompt_indices
            or len(set(self.activation_prompt_indices))
            != len(self.activation_prompt_indices)
            or any(
                not 0 <= i < len(self.prompts) for i in self.activation_prompt_indices
            )
        ):
            message = "Invalid evaluation or prompt panel"
            raise ValueError(message)
        if (
            config.architecture != "connectome"
            or config.control != "real"
            or config.tokenization != "bpe"
            or not config.trainable_codes
            or config.generation_context != "stateful"
            or config.port_policy != "legacy_random"
            or config.alphabet_size != self.corpus.vocabulary
            or config.learning_rate != self.schedule.peak
        ):
            message = "Quality protocol requires the existing real BPE connectome"
            raise ValueError(message)
        if self.profile == "quality":
            self._fixed_quality()
        return self

    def _fixed_quality(self) -> None:
        """Prevent altered experiments from claiming the fixed protocol."""
        config = self.config
        if (
            self.schedule != Schedule()
            or self.evaluation_updates != EVALUATIONS
            or (self.early_update, self.checkpoint_interval) != (1000, 1000)
            or self.prompts != PROMPTS
            or self.generation != QualityGenerationOptions()
            or self.activation_prompt_indices != (0, 3)
            or (self.graph.nodes, self.graph.edges) != (16_384, 1_187_999)
            or self.corpus.stories != (50_000, 256, 512)
            or config.device not in {"cuda", "cuda:0"}
            or (self.corpus.vocabulary, config.seed, config.context, config.batch_size)
            != (4096, 0, 128, 8)
            or (config.readout_neurons, config.weight_decay, config.gradient_clip)
            != (256, 0, 1)
            or (config.leak, config.initial_gain) != (0.5, 0.9)
        ):
            message = "Quality profile must match the fixed 30,000-update protocol"
            raise ValueError(message)


class Evaluation(Settings):
    """Validation selection input bound to one immutable checkpoint."""

    update: Annotated[int, Field(ge=0)]
    checkpoint_sha256: Digest
    metrics: QualityMetrics


def select_update(validation: tuple[Evaluation, ...]) -> int:
    """Select by validation NLL, excluding initialization and breaking ties early."""
    candidates = tuple(
        (item.metrics.nll, item.update)
        for item in validation
        if item.update > 0 and item.metrics.nll is not None
    )
    if not candidates:
        message = "Selection requires nonzero checkpoint validation evidence"
        raise ValueError(message)
    return min(candidates)[1]


class ArtifactFile(Settings):
    """Checksummed relative activation artifact, safe to resolve inside run output."""

    path: str
    sha256: Digest

    @model_validator(mode="after")
    def relative_path(self) -> Self:
        """Reject absolute paths and parent traversal from loaded progress files."""
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            message = "Artifact path must remain inside the output directory"
            raise ValueError(message)
        return self


class CheckpointResult(Settings):
    """Untouched-test evidence, unfiltered draws and full-state recording paths."""

    update: Positive
    checkpoint_sha256: Digest
    test: QualityMetrics
    generation: tuple[QualityGenerationRecord, ...]
    activations: tuple[str, ...]
    artifacts: tuple[ArtifactFile, ...]


class SuccessLabel(StrEnum):
    """The protocol's three possible reporting decisions."""

    IMPROVED = "language-quality improvement demonstrated"
    UNRELIABLE = (
        "Held-out prediction improved; "
        "reliable short-story generation was not demonstrated."
    )
    NOT_DEMONSTRATED = "language-quality improvement not demonstrated"


class MetricSummary(Settings):
    """Explicit machine-readable aggregates of the published additive evidence."""

    update: int
    split: Literal["valid", "test"]
    loss_sum: float
    count: int
    correct: int
    nll: float | None
    perplexity: float | None
    accuracy: float | None


class QualityProgress(Settings):
    """Resume transaction and final report; completed artifacts use relative paths."""

    schema_version: Literal[1] = 1
    protocol: QualityProtocol
    protocol_sha256: Digest
    runtime: str
    updates: Annotated[int, Field(ge=0)]
    next_lr: float | None
    checkpoint_sha256: Digest
    validation: tuple[Evaluation, ...] = ()
    completed: tuple[CheckpointResult, ...] = ()
    selected_update: int | None = None
    repeat_collapse_count: int | None = None
    recovery_gates_complete: bool = False
    summaries: tuple[MetricSummary, ...] = ()
    success_label: SuccessLabel = SuccessLabel.NOT_DEMONSTRATED
