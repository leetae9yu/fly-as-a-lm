"""Train-only add-half references on exactly the within-story target positions."""

from math import exp

import numpy as np

from flyrl.language_data import IntVector
from flyrl.regional_probe_types import ProbeMetrics
from flyrl.story_data import StorySplit


def story_pairs(
    tokens: IntVector, split: StorySplit, vocab_size: int
) -> tuple[IntVector, IntVector]:
    """Return story-major contexts and targets, never crossing a boundary."""
    if (
        vocab_size <= 1
        or tokens.dtype != np.int64
        or tokens.ndim != 1
        or bool(((tokens < 0) | (tokens >= vocab_size)).any())
        or len(split.offsets) != len(split.sha256) + 1
        or not split.sha256
        or split.offsets[0] != 0
        or split.offsets[-1] != tokens.size
        or any(
            b < a for a, b in zip(split.offsets[:-1], split.offsets[1:], strict=True)
        )
    ):
        message = "Invalid story baseline tokens or boundaries"
        raise ValueError(message)
    bounds = tuple(zip(split.offsets[:-1], split.offsets[1:], strict=True))
    positions = np.fromiter(
        (position for a, b in bounds for position in range(a, b - 1)), dtype=np.int64
    )
    contexts, targets = tokens[positions], tokens[positions + 1]
    if not targets.size:
        message = "Story baseline requires next-token pairs"
        raise ValueError(message)
    return contexts, targets


def story_baselines(
    train: IntVector,
    train_split: StorySplit,
    test: IntVector,
    test_split: StorySplit,
    vocab_size: int,
) -> tuple[ProbeMetrics, ProbeMetrics]:
    """Fit add-half unigram/bigram counts on train targets; score every test pair.

    Unseen bigram contexts are uniform, with lowest-token-ID argmax tie breaking.
    Unigram counts match the probe's train-target bias initialization.
    """
    contexts, targets = story_pairs(train, train_split, vocab_size)
    test_contexts, test_targets = story_pairs(test, test_split, vocab_size)
    unigram = np.bincount(targets, minlength=vocab_size).astype(np.float64) + 0.5
    unigram /= unigram.sum()
    bigram = (
        np.bincount(contexts * vocab_size + targets, minlength=vocab_size * vocab_size)
        .reshape(vocab_size, vocab_size)
        .astype(np.float64)
        + 0.5
    )
    bigram /= bigram.sum(axis=1, keepdims=True)
    scores: list[ProbeMetrics] = []
    unigram_predictions: IntVector = np.full(
        test_targets.size, int(unigram.argmax()), dtype=np.int64
    )
    bigram_predictions: IntVector = np.asarray(bigram.argmax(axis=1), dtype=np.int64)[
        test_contexts
    ]
    for probabilities, predictions in (
        (unigram[test_targets], unigram_predictions),
        (bigram[test_contexts, test_targets], bigram_predictions),
    ):
        nll = float(-np.log(probabilities).mean())
        scores.append(
            ProbeMetrics(
                tokens=test_targets.size,
                nll=nll,
                perplexity=exp(nll),
                accuracy=sum(
                    int(predictions.item(i)) == int(test_targets.item(i))
                    for i in range(test_targets.size)
                )
                / test_targets.size,
            )
        )
    return scores[0], scores[1]
