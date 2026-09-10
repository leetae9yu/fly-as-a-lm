import numpy as np

from flyrl.language_baselines import fit_baselines, score_baselines
from flyrl.language_data import CorpusLimits, TextSplits, make_corpus


def test_bigram_beats_constant_action_on_alternating_text() -> None:
    # Given: training text whose next character is determined by the previous one.
    corpus = make_corpus(TextSplits("abababab", "abab", "baba", "x"), CorpusLimits())
    fitted = fit_baselines(corpus)
    # When: scoring held-out targets using each fitted baseline.
    scores = score_baselines(fitted, corpus.test, np.array([2, 3], dtype=np.int64))
    # Then: constant-action frequency alone cannot match conditional prediction.
    assert scores[0].greedy_accuracy == 0.5
    assert scores[1].greedy_accuracy == 1.0
    assert scores[2].greedy_accuracy == 1.0
    assert scores[1].bits_per_character < scores[0].bits_per_character


def test_heldout_text_cannot_change_fitted_probabilities() -> None:
    # Given: identical training text, different validation/test distributions.
    first = make_corpus(TextSplits("abababab", "aaaa", "aaaa", "x"), CorpusLimits())
    second = make_corpus(TextSplits("abababab", "bbbb", "bbbb", "x"), CorpusLimits())
    # When: fitting reference models.
    one, two = fit_baselines(first), fit_baselines(second)
    # Then: no held-out labels entered the probability estimates.
    np.testing.assert_array_equal(one.unigram, two.unigram)
    np.testing.assert_array_equal(one.bigram, two.bigram)
    np.testing.assert_array_equal(one.trigram, two.trigram)


def test_unseen_contexts_have_finite_log_scores() -> None:
    # Given: test-only characters mapped to the reserved unknown symbol.
    corpus = make_corpus(TextSplits("abababab", "xxx", "xxx", "x"), CorpusLimits())
    # When: scoring a never-seen context and target.
    scores = score_baselines(
        fit_baselines(corpus), corpus.test, np.array([2], dtype=np.int64)
    )
    # Then: smoothing supplies finite, positive probabilities for every baseline.
    assert all(np.isfinite(score.bits_per_character) for score in scores)
    assert all(score.expected_reward > 0 for score in scores)
