"""Train-only smoothed n-gram reference models for the same character targets."""

from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np

from flyrl.language_data import Corpus, CorpusError, IntVector
from flyrl.models import Settings

FloatVector: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
FloatMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
FloatCube: TypeAlias = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
MIN_CONTEXT: Final = 2


@dataclass(frozen=True, slots=True)
class Ngrams:
    """Add-half probabilities fitted exclusively to training character counts."""

    unigram: FloatVector
    bigram: FloatMatrix
    trigram: FloatCube


class BaselineScore(Settings):
    """A baseline measured at the exact same positions as the neural model."""

    name: str
    greedy_accuracy: float
    expected_reward: float
    bits_per_character: float
    trials: int


def fit_baselines(corpus: Corpus) -> Ngrams:
    """Fit unigram, bigram and trigram counts without reading held-out tokens."""
    tokens = corpus.train
    size = len(corpus.alphabet)
    unigram = np.bincount(tokens, minlength=size).astype(np.float64) + 0.5
    pairs = tokens[:-1] * size + tokens[1:]
    bigram = np.bincount(pairs, minlength=size**2).reshape(size, size) + 0.5
    triples = (tokens[:-2] * size + tokens[1:-1]) * size + tokens[2:]
    trigram = np.bincount(triples, minlength=size**3).reshape(size, size, size) + 0.5
    return Ngrams(
        unigram / unigram.sum(),
        bigram / bigram.sum(axis=-1, keepdims=True),
        trigram / trigram.sum(axis=-1, keepdims=True),
    )


def score_baselines(
    model: Ngrams, tokens: IntVector, positions: IntVector
) -> tuple[BaselineScore, ...]:
    """Score supplied target positions, which must have two preceding tokens."""
    if (
        not positions.size
        or ((positions < MIN_CONTEXT) | (positions >= tokens.size)).any()
    ):
        raise CorpusError(reason="Baseline targets require two preceding characters")
    targets = tokens[positions]
    probabilities = (
        ("unigram", np.ones((positions.size, 1)) * model.unigram),
        ("bigram", model.bigram[tokens[positions - 1]]),
        ("trigram", model.trigram[tokens[positions - 2], tokens[positions - 1]]),
    )
    scores: list[BaselineScore] = []
    for name, probability in probabilities:
        assigned = probability[np.arange(positions.size), targets]
        scores.append(
            BaselineScore(
                name=name,
                greedy_accuracy=float(
                    np.equal(probability.argmax(axis=1), targets).mean()
                ),
                expected_reward=float(assigned.mean()),
                bits_per_character=float(-np.log2(assigned).mean()),
                trials=positions.size,
            )
        )
    return tuple(scores)
