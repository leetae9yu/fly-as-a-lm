from pathlib import Path

import numpy as np
import pytest

from flyrl.language_data import (
    CorpusError,
    CorpusLimits,
    TextSplits,
    evaluation_starts,
    load_corpus,
    make_corpus,
    save_corpus,
)


def test_vocabulary_uses_training_prefix_only() -> None:
    # Given: validation/test-only symbols and a training character after the cap.
    text = TextSplits("AB ABxy", "zz zz", "qq qq", "synthetic")
    # When: preparing capped, separately normalized splits.
    corpus = make_corpus(text, CorpusLimits(train_chars=5))
    # Then: unseen symbols never expand the action space and map to index zero.
    assert set(corpus.alphabet) == {"?", "a", "b", " "}
    assert np.equal(corpus.valid, 0).sum() == 4
    assert np.equal(corpus.test, 0).sum() == 4
    assert corpus.train.size == 5


def test_preparation_preserves_split_boundaries() -> None:
    # Given: distinguishable split contents after normalization.
    text = TextSplits("AA bb\ncc", "bb\nbb", "cc\tcc", "synthetic")
    # When: preparing a corpus.
    corpus = make_corpus(text, CorpusLimits())
    # Then: decoding each encoded split reconstructs only its own normalized text.
    assert "".join(corpus.alphabet[i] for i in corpus.train) == "aa bb cc"
    assert "".join(corpus.alphabet[i] for i in corpus.valid) == "bb bb"
    assert "".join(corpus.alphabet[i] for i in corpus.test) == "cc cc"


def test_content_changes_change_fingerprint() -> None:
    # Given: corpora differing only in held-out data.
    original = make_corpus(TextSplits("ab ab", "ab", "ab", "x"), CorpusLimits())
    # When: another test split is supplied.
    changed = make_corpus(TextSplits("ab ab", "ab", "ba", "x"), CorpusLimits())
    # Then: resumable training cannot silently reuse a different corpus.
    assert original.fingerprint != changed.fingerprint


def test_npz_roundtrip_is_exact(tmp_path: Path) -> None:
    # Given: a corpus with fitted vocabulary.
    corpus = make_corpus(TextSplits("ab ab", "ab", "ba", "x"), CorpusLimits())
    path = tmp_path / "corpus.npz"
    save_corpus(corpus, path)
    # When: reopening the artifact without pickle.
    restored = load_corpus(path)
    # Then: IDs, vocabulary, and fingerprint match exactly.
    assert restored.alphabet == corpus.alphabet
    assert restored.fingerprint == corpus.fingerprint
    np.testing.assert_array_equal(restored.train, corpus.train)
    np.testing.assert_array_equal(restored.valid, corpus.valid)
    np.testing.assert_array_equal(restored.test, corpus.test)


def test_evaluation_windows_do_not_overlap_or_cross_boundary() -> None:
    # Given: a split with enough characters for more windows than requested.
    tokens = np.arange(100, dtype=np.int64)
    # When: selecting three windows with four context characters and one target.
    starts = evaluation_starts(tokens, 4, 3)
    # Then: windows cover the split without shared tokens or an out-of-range target.
    np.testing.assert_array_equal(starts, [0, 45, 95])
    assert ((starts[1:] - starts[:-1]) >= 5).all()
    assert starts.item(-1) + 4 < len(tokens)


@pytest.mark.parametrize(("context", "limit"), [(0, 3), (4, 0), (100, 3)])
def test_invalid_window_request_fails(context: int, limit: int) -> None:
    # Given: an impossible or empty evaluation request.
    # When / Then: the boundary rejects it instead of returning an empty score.
    with pytest.raises(CorpusError):
        _ = evaluation_starts(np.arange(100, dtype=np.int64), context, limit)
