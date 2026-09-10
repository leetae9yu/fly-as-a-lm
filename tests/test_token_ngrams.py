"""Sparse token references must agree with the existing dense character formula."""

import tracemalloc

import numpy as np
import pytest

from flyrl import ar_ngrams
from flyrl.language_baselines import fit_baselines, score_baselines
from flyrl.language_data import CorpusLimits, TextSplits, make_corpus


def test_sparse_counts_match_dense_add_half_reference() -> None:
    # Given: a small vocabulary including contexts absent from training.
    corpus = make_corpus(
        TextSplits("aababa", "bbabba", "bbabba", "synthetic"), CorpusLimits()
    )
    positions = np.arange(2, corpus.test.size, dtype=np.int64)
    expected = score_baselines(fit_baselines(corpus), corpus.test, positions)
    # When: scoring the identical targets with sparse observed-context counts.
    sparse = ar_ngrams.fit_ngrams(corpus.train, len(corpus.alphabet))
    actual = [model.score(corpus.test, positions) for model in sparse]
    # Then: smoothing, unseen contexts and argmax tie-breaking agree.
    for reference, observed in zip(expected, actual, strict=True):
        assert observed.greedy_accuracy == reference.greedy_accuracy
        assert observed.bits_per_token == pytest.approx(reference.bits_per_character)
        assert observed.bits_per_character is None


def test_token_baselines_do_not_allocate_vocabulary_cubes() -> None:
    # Given: 4096 token IDs but only a few observed contexts.
    tokens = np.array([0, 4095] * 1000, dtype=np.int64)
    positions = np.array([2, 3], dtype=np.int64)
    tracemalloc.start()
    # When: fitting and scoring large-vocabulary references.
    try:
        models = ar_ngrams.fit_ngrams(tokens, 4096)
        scores = [model.score(tokens, positions) for model in models]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Then: allocation follows observed data, rather than a 4096**3 cube.
    assert peak < 2_000_000
    assert len(scores) == 3
    assert all(score.windows == 2 for score in scores)
