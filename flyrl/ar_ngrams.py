"""Observed-context n-grams for large token vocabularies."""

from collections import Counter
from dataclasses import dataclass
from math import exp, log
from typing import Final

from flyrl.ar_config import ARMetrics
from flyrl.language_data import CorpusError, IntVector

SMOOTHING: Final = 0.5
ORDERS: Final = (0, 1, 2)


@dataclass(frozen=True, slots=True)
class ContextCounts:
    """Observed outcomes and a cached denominator/argmax for one prefix."""

    counts: Counter[int]
    total: int
    mode: int


@dataclass(frozen=True, slots=True)
class SparseNgram:
    """Add-half reference with O(observed contexts and transitions) storage."""

    order: int
    vocabulary_size: int
    contexts: dict[tuple[int, ...], ContextCounts]

    def score(self, tokens: IntVector, positions: IntVector) -> ARMetrics:
        """Score only the supplied targets; unseen contexts are uniform."""
        if (
            not positions.size
            or ((positions < self.order) | (positions >= tokens.size)).any()
        ):
            raise CorpusError(reason="N-gram targets need their preceding context")
        nll, correct = 0.0, 0
        for position in positions:
            index = int(position)
            key = tuple(int(tokens.item(j)) for j in range(index - self.order, index))
            target = int(tokens.item(index))
            context = self.contexts.get(key)
            if context is None:
                probability, mode = 1 / self.vocabulary_size, 0
            else:
                probability = (context.counts.get(target, 0) + SMOOTHING) / (
                    context.total + SMOOTHING * self.vocabulary_size
                )
                mode = context.mode
            nll -= log(probability)
            correct += int(mode == target)
        nll /= positions.size
        return ARMetrics(
            windows=int(positions.size),
            greedy_accuracy=correct / positions.size,
            nll=nll,
            bits_per_token=nll / log(2),
            perplexity=exp(nll),
        )


def fit_ngrams(tokens: IntVector, vocabulary_size: int) -> tuple[SparseNgram, ...]:
    """Fit unigram, bigram and trigram counts only from supplied training tokens."""
    models: list[SparseNgram] = []
    for order in ORDERS:
        observed: dict[tuple[int, ...], Counter[int]] = {}
        for index in range(order, tokens.size):
            key = tuple(int(tokens.item(j)) for j in range(index - order, index))
            if key not in observed:
                observed[key] = Counter()
            observed[key][int(tokens.item(index))] += 1
        contexts = {
            key: ContextCounts(
                counts,
                sum(counts.values()),
                min(counts, key=lambda token: (-counts[token], token)),
            )
            for key, counts in observed.items()
        }
        models.append(SparseNgram(order, vocabulary_size, contexts))
    return tuple(models)
