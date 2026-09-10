"""Train-only BPE fitting and lossless byte-level text encoding."""

import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from flyrl.bpe_data import make_bpe_corpus
from flyrl.language_data import TextSplits


@pytest.fixture
def text() -> TextSplits:
    return TextSplits("ab ac ad ba ca da " * 8, "held out", "test text", "fixture")


def test_fitting_is_deterministic_with_merge_ties(text: TextSplits) -> None:
    # Given: many equal-frequency competing merges.
    first = make_bpe_corpus(text, 280)
    # When: fitting independent tokenizers.
    others = [make_bpe_corpus(text, 280) for _ in range(4)]
    # Then: merge ordering, IDs, and identity are reproducible.
    assert all(other.tokenizer_json == first.tokenizer_json for other in others)
    assert all(other.fingerprint == first.fingerprint for other in others)


def test_fitting_is_reproducible_across_processes() -> None:
    # Given: fresh interpreters with different hash seeds and thread counts.
    # When: fitting through the public API in each process.
    outputs = [
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                (
                    "from flyrl.bpe_data import make_bpe_corpus; "
                    "from flyrl.language_data import TextSplits; "
                    "print(make_bpe_corpus(TextSplits('ab ac ad ba ca da '*8, "
                    "'valid', 'test', 'fixture'), 280).tokenizer_json)"
                ),
            ],
            env={**os.environ, "PYTHONHASHSEED": seed, "RAYON_NUM_THREADS": seed},
            timeout=30,
        )
        for seed in ("1", "2", "4")
    ]
    # Then: process-level randomness cannot change the serialized tokenizer.
    assert outputs[0] == outputs[1] == outputs[2]


def test_heldout_changes_do_not_fit_vocabulary(text: TextSplits) -> None:
    # Given: fixed training data.
    first = make_bpe_corpus(text, 300)
    # When: held-out text changes completely.
    other = make_bpe_corpus(
        replace(text, valid="\u6f22\u5b57" * 30, test="xyz" * 30), 300
    )
    # Then: only held-out tokens and corpus identity change.
    assert first.vocabulary == other.vocabulary
    assert first.tokenizer_json == other.tokenizer_json
    np.testing.assert_array_equal(first.train, other.train)
    assert first.fingerprint != other.fingerprint


@pytest.mark.parametrize(
    "source",
    [
        "",
        " \t\r\n  ",
        "\U0001f98b\u6f22\u5b57\u00e9e\u0301\x00",
        "Hi <unk><unk>\n  THERE",
    ],
)
def test_unseen_unicode_and_whitespace_roundtrip(text: TextSplits, source: str) -> None:
    # Given: a tokenizer fitted on ASCII only, without normalization.
    corpus = make_bpe_corpus(text)
    # When: unseen text is encoded and decoded.
    decoded = corpus.decode(corpus.encode(source))
    # Then: neither bytes nor literal unknown markers are discarded.
    assert decoded == source
