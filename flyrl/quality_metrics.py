"""Immutable per-story evidence for full-prefix heldout quality evaluation."""

from itertools import pairwise
from math import exp, fsum, log
from sys import float_info
from typing import Annotated, Self

import numpy as np
import torch
from pydantic import Field, model_validator

from flyrl.ar_learning import ARLearner
from flyrl.language_data import IntVector
from flyrl.language_models import Settings
from flyrl.story_data import StorySplit


class StoryEvidence(Settings):
    """Additive scores for one supplied story identity, including zero targets."""

    sha256: str
    loss_sum: Annotated[float, Field(ge=0)]
    count: Annotated[int, Field(ge=0)]
    correct: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        """Reject impossible evidence rather than fabricate aggregate metrics."""
        if self.correct > self.count or (self.count == 0 and self.loss_sum != 0):
            message = (
                "Story evidence needs correct <= count and zero loss without targets"
            )
            raise ValueError(message)
        return self


class QualityMetrics(Settings):
    """Ordered story evidence; every aggregate is derived from additive scores."""

    per_story: tuple[StoryEvidence, ...]

    @property
    def loss_sum(self) -> float:
        """Pool natural-log losses without weighting stories equally."""
        return fsum(story.loss_sum for story in self.per_story)

    @property
    def count(self) -> int:
        """Count only within-story next-token pairs, not first tokens."""
        return sum(story.count for story in self.per_story)

    @property
    def correct(self) -> int:
        """Pool greedy correct predictions over the same target denominator."""
        return sum(story.correct for story in self.per_story)

    @property
    def nll(self) -> float | None:
        """Return token-weighted NLL, undefined when there are no targets."""
        count = self.count
        return self.loss_sum / count if count else None

    @property
    def perplexity(self) -> float | None:
        """Exponentiate pooled NLL, never average per-story perplexities."""
        nll = self.nll
        return exp(nll) if nll is not None and nll <= log(float_info.max) else None

    @property
    def accuracy(self) -> float | None:
        """Return token-weighted greedy accuracy, undefined without targets."""
        count = self.count
        return self.correct / count if count else None


def _validate_split(tokens: IntVector, split: StorySplit, vocabulary: int) -> None:
    """Validate direct callers independently of StoryCorpus's stricter lengths."""
    if (
        tokens.ndim != 1
        or tokens.dtype != np.int64
        or bool(((tokens < 0) | (tokens >= vocabulary)).any())
    ):
        message = "Evaluation tokens must be a 1D in-vocabulary int64 array"
        raise ValueError(message)
    if (
        len(split.offsets) != len(split.sha256) + 1
        or not split.offsets
        or split.offsets[0] != 0
        or split.offsets[-1] != tokens.size
        or any(stop < start for start, stop in pairwise(split.offsets))
    ):
        message = (
            "Story boundaries must cover all tokens in order, one interval per hash"
        )
        raise ValueError(message)


@torch.no_grad()
def evaluate_quality_split(
    learner: ARLearner,
    tokens: IntVector,
    split: StorySplit,
    *,
    chunk_size: int | None = None,
) -> QualityMetrics:
    """Score all within-story pairs with local state and bounded logits storage.

    The chunk default is the learner's training context, but it never truncates
    evaluation history. Only story boundaries reset state. Supplied SHA256 values
    are retained verbatim; verifying text identities belongs to corpus loading.
    Neither model mode, parameters, gradients, optimizer nor window RNG changes.
    """
    chunk = learner.config.context if chunk_size is None else chunk_size
    if chunk < 1:
        message = "Evaluation chunk size must be positive"
        raise ValueError(message)
    _validate_split(tokens, split, learner.config.alphabet_size)
    model = learner.model
    device = model.weight.device
    evidence: list[StoryEvidence] = []
    for start, stop, identity in zip(
        split.offsets[:-1], split.offsets[1:], split.sha256, strict=True
    ):
        state = model.weight.new_zeros((model.nodes, 1))
        loss_sum, count, correct = 0.0, 0, 0
        for offset in range(start, stop - 1, chunk):
            end = min(offset + chunk, stop - 1)
            inputs = torch.tensor(tokens[offset:end], device=device)
            targets = torch.tensor(tokens[offset + 1 : end + 1], device=device)
            outputs: list[torch.Tensor] = []
            for token in inputs.unbind():
                logits, state = model.step(token.reshape(1), state)
                outputs.append(logits)
            stacked = torch.cat(outputs)
            losses = torch.nn.functional.cross_entropy(
                stacked, targets, reduction="none"
            )
            # Accumulate float32 token losses in float64 to avoid chunk-dependent
            # float32 reduction rounding without retaining full-story logits.
            loss_sum += float(losses.double().sum().item())
            correct += int((stacked.argmax(dim=1) == targets).sum().item())
            count += end - offset
        evidence.append(
            StoryEvidence(
                sha256=identity, loss_sum=loss_sum, count=count, correct=correct
            )
        )
    return QualityMetrics(per_story=tuple(evidence))
