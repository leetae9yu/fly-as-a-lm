"""Centered outgoing interventions and recurrence-free dense score recovery.

All caches are story-major next-token pairs. Replay uses saved float32 values
without fitting, renormalizing, or changing the source model. Loss evidence is
summed in float64 from per-target float32 cross entropy. For bitwise recovery,
use the same dense batch size, device, and floating-point runtime settings.
"""

from dataclasses import dataclass
from itertools import accumulate, pairwise
from math import isfinite
from typing import Final, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn.functional import cross_entropy

from flyrl.ar_model import ConnectomeLM, requested_device
from flyrl.language_data import IntVector
from flyrl.regional_probe import InterventionExtraction, extract_features
from flyrl.regional_probe_types import ExtractionConfig, FeatureCache, ProbeParameters
from flyrl.story_data import StorySplit

FloatArray: TypeAlias = NDArray[np.float32]
MATRIX_DIMENSIONS: Final = 2
MIN_VOCABULARY: Final = 2


@dataclass(frozen=True, slots=True)
class CenteredOutgoing:
    """Source neuron indices and a full-node old-training mean; empty is a no-op."""

    indices: tuple[int, ...]
    training_mean: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject ambiguous groups and nonfinite calibration at the boundary."""
        if len(set(self.indices)) != len(self.indices) or any(
            index < 0 or index >= len(self.training_mean) for index in self.indices
        ):
            message = "Intervention indices must be unique and within the mean vector"
            raise ValueError(message)
        if not all(isfinite(value) for value in self.training_mean):
            message = "Training mean must be finite"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ScoreEvidence:
    """Additive evidence, including zero-target stories without fabricated means."""

    loss_sum: float
    count: int
    correct: int

    @property
    def nll(self) -> float | None:
        """Return token-weighted natural-log loss when targets exist."""
        return self.loss_sum / self.count if self.count else None

    @property
    def accuracy(self) -> float | None:
        """Return greedy accuracy when targets exist."""
        return self.correct / self.count if self.count else None


@dataclass(frozen=True, slots=True)
class DenseReplay:
    """Evidence in the supplied split's original story order."""

    story_sha256: tuple[str, ...]
    stories: tuple[ScoreEvidence, ...]

    @property
    def total(self) -> ScoreEvidence:
        """Pool additive evidence, never average story NLLs or perplexities."""
        return ScoreEvidence(
            loss_sum=sum(story.loss_sum for story in self.stories),
            count=sum(story.count for story in self.stories),
            correct=sum(story.correct for story in self.stories),
        )


@dataclass(frozen=True, slots=True)
class StateEvaluation:
    """Retained original-readout states/targets and independently replayable scores."""

    cache: FeatureCache
    scores: DenseReplay


def _pair_offsets(split: StorySplit, pairs: int) -> tuple[int, ...]:
    if (
        not split.sha256
        or len(split.offsets) != len(split.sha256) + 1
        or split.offsets[0] != 0
        or any(b < a for a, b in pairwise(split.offsets))
    ):
        message = "Story boundaries must cover tokens in order"
        raise ValueError(message)
    offsets = tuple(
        accumulate(
            (max(0, b - a - 1) for a, b in pairwise(split.offsets)),
            initial=0,
        )
    )
    if offsets[-1] != pairs:
        message = "Story boundaries do not match the retained target count"
        raise ValueError(message)
    return offsets


def _validate_head(cache: FeatureCache, weight: FloatArray, bias: FloatArray) -> None:
    if (
        cache.features.ndim != MATRIX_DIMENSIONS
        or cache.features.dtype != np.float32
        or cache.labels.ndim != 1
        or cache.labels.dtype != np.int64
        or cache.features.shape[0] != cache.labels.size
        or weight.ndim != MATRIX_DIMENSIONS
        or weight.shape[0] != cache.features.shape[1]
        or not weight.shape[0]
        or weight.shape[1] < MIN_VOCABULARY
        or bias.shape != (weight.shape[1],)
        or any(
            array.dtype != np.float32 or not bool(np.isfinite(array).all())
            for array in (cache.features, weight, bias)
        )
    ):
        message = (
            "Expected finite float32 features/head with aligned shapes and int64 labels"
        )
        raise ValueError(message)
    if bool(((cache.labels < 0) | (cache.labels >= bias.size)).any()):
        message = "Replay labels exceed vocabulary bounds"
        raise ValueError(message)


@torch.no_grad()
def _replay(
    cache: FeatureCache,
    split: StorySplit,
    tensors: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ],
    batch_size: int,
) -> DenseReplay:
    if batch_size < 1:
        message = "Replay batch_size must be positive"
        raise ValueError(message)
    offsets = _pair_offsets(split, cache.labels.size)
    weight, bias, mean, std = tensors
    losses: NDArray[np.float64] = np.empty(cache.labels.size, dtype=np.float64)
    correct: NDArray[np.bool_] = np.empty(cache.labels.size, dtype=np.bool_)
    for start in range(0, cache.labels.size, batch_size):
        stop = start + batch_size
        features = torch.tensor(cache.features[start:stop], device=weight.device)
        labels = torch.tensor(cache.labels[start:stop], device=weight.device)
        if mean is not None and std is not None:
            features = (features - mean) / std
        logits = features @ weight + bias
        losses[start:stop] = (
            cross_entropy(logits, labels, reduction="none").cpu().numpy()
        )
        correct[start:stop] = (logits.argmax(dim=1) == labels).cpu().numpy()
    return DenseReplay(
        story_sha256=split.sha256,
        stories=tuple(
            ScoreEvidence(
                loss_sum=sum(
                    float(cast("np.float64", losses[index])) for index in range(a, b)
                ),
                count=b - a,
                correct=sum(
                    int(cast("np.bool_", correct[index])) for index in range(a, b)
                ),
            )
            for a, b in pairwise(offsets)
        ),
    )


def replay_original_readout(
    cache: FeatureCache,
    split: StorySplit,
    head: tuple[FloatArray, FloatArray],
    *,
    batch_size: int = 2048,
    device: str = "cpu",
) -> DenseReplay:
    """Score retained port-ordered states using saved unnormalized original head.

    Weight is [ports, vocabulary], bias is [vocabulary]. No model or recurrent
    operator is accessed; callers can replay a readout-mean positive control by
    supplying a cache whose features have been replaced with training port means.
    """
    weight, bias = head
    _validate_head(cache, weight, bias)
    target = requested_device(device)
    return _replay(
        cache,
        split,
        (
            torch.tensor(weight, device=target),
            torch.tensor(bias, device=target),
            None,
            None,
        ),
        batch_size,
    )


def replay_probe(
    cache: FeatureCache,
    split: StorySplit,
    parameters: ProbeParameters,
    *,
    batch_size: int = 2048,
    device: str = "cpu",
) -> DenseReplay:
    """Score saved probes with their exact weight, bias, training mean and std.

    Features must follow the saved probe's neuron order. Normalization is reused
    verbatim: there is no fresh-data calibration and no additional std floor.
    """
    weight = np.asarray(parameters.weight, dtype=np.float32)
    bias = np.asarray(parameters.bias, dtype=np.float32)
    mean = np.asarray(parameters.mean, dtype=np.float32)
    std = np.asarray(parameters.std, dtype=np.float32)
    _validate_head(cache, weight, bias)
    if (
        mean.shape != (weight.shape[0],)
        or std.shape != mean.shape
        or not bool(np.isfinite(mean).all())
        or not bool(np.isfinite(std).all())
        or bool((std <= 0).any())
    ):
        message = (
            "Saved probe normalization must be finite, aligned, and positive in std"
        )
        raise ValueError(message)
    target = requested_device(device)
    return _replay(
        cache,
        split,
        (
            torch.tensor(weight, device=target),
            torch.tensor(bias, device=target),
            torch.tensor(mean, device=target),
            torch.tensor(std, device=target),
        ),
        batch_size,
    )


@torch.no_grad()
def evaluate_intervention(
    model: ConnectomeLM,
    tokens: IntVector,
    split: StorySplit,
    *,
    intervention: CenteredOutgoing | None = None,
    config: ExtractionConfig | None = None,
) -> StateEvaluation:
    """Evaluate one baseline/intervention arm with full story context and reset.

    The original readout scores every retained state. Chunking bounds recurrence
    memory; finished lanes are never recycled or scored. Parameters, buffers,
    gradients, mode and RNG are untouched. Nothing is printed or persisted.
    """
    indices = tuple(int(index.item()) for index in model.ports.detach().cpu().unbind())
    options: ExtractionConfig | InterventionExtraction = config or ExtractionConfig()
    if intervention is not None:
        options = InterventionExtraction(intervention, options)
    cache = extract_features(model, tokens, split, indices, config=options)
    scores = replay_original_readout(
        cache,
        split,
        (
            model.readout.detach().cpu().numpy(),
            model.output_bias.detach().cpu().numpy(),
        ),
        device=str(model.weight.device),
    )
    return StateEvaluation(cache=cache, scores=scores)
