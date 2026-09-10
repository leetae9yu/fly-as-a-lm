import numpy as np
import pytest

from flyrl.language_data import CorpusError, CorpusLimits, TextSplits, make_corpus
from scripts.prepare_ar_corpus import reserve_test_region


def test_fresh_test_region_preserves_training_and_vocabulary() -> None:
    # Given: a corpus whose original test prefix has already been evaluated.
    base = make_corpus(
        TextSplits("abc abc abc", "abc", "aaa", "fixture"), CorpusLimits()
    )
    # When: reserving a later normalized region with an unseen symbol.
    fresh = reserve_test_region(base, "AAA BBB ZZZ", 8)
    # Then: only the test content/identity changes; no vocabulary is refitted.
    assert fresh.alphabet == base.alphabet
    np.testing.assert_array_equal(fresh.train, base.train)
    np.testing.assert_array_equal(fresh.valid, base.valid)
    np.testing.assert_array_equal(fresh.test, [0, 0, 0])
    assert fresh.fingerprint != base.fingerprint


@pytest.mark.parametrize("start", [-1, 1, 100])
def test_overlapping_or_incomplete_region_is_rejected(start: int) -> None:
    # Given: a requested slice that overlaps the original or exceeds the source.
    base = make_corpus(TextSplits("abc abc", "abc", "aaa", "fixture"), CorpusLimits())
    # When / Then: no short or reused final-test set is silently accepted.
    with pytest.raises(CorpusError):
        _ = reserve_test_region(base, "aaa bbb ccc", start)
