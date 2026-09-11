"""Story boundaries, leakage prevention and train-only tokenizer identity."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from flyrl import bpe_data
from flyrl.bpe_tokenizer import fit_tokenizer
from flyrl.language_data import CorpusError
from flyrl.story_data import load_story_corpus, prepare_stories
from flyrl.story_source import Preparation


def test_story_preparation_contract_exists() -> None:
    # Given/When: discovering the new preparation boundary.
    spec = importlib.util.find_spec("flyrl.story_data")
    # Then: preparation is available without using the flat-stream pipeline.
    assert spec is not None, "story-safe preparation module is missing"


def test_boundaries_deduplication_and_roundtrip(tmp_path: Path) -> None:
    # Given: duplicates both within and across raw source files.
    stories = [f"The little bird sang song {i}." for i in range(8)]
    raw = tmp_path / "raw.txt"
    _ = raw.write_text(
        "\n<|endoftext|>\n".join([stories[0], *stories[:3]]), encoding="utf-8"
    )
    second = tmp_path / "second.txt"
    _ = second.write_text(
        "\n<|endoftext|>\n".join([stories[0], *stories[3:]]), encoding="utf-8"
    )
    options = Preparation(
        raw_files=(raw, second),
        source="synthetic",
        revision="fixture-v1",
        license_text="Synthetic fixture; CC0",
        train_stories=4,
        valid_stories=2,
        test_stories=2,
        vocab_size=257,
    )
    # When: preparing and loading through the existing BPE artifact boundary.
    prepared = prepare_stories(options)
    path = tmp_path / "corpus.npz"
    bpe_data.save_bpe_corpus(prepared.corpus, path)
    loaded = load_story_corpus(path)
    # Then: no duplicates survive, the boundaries roundtrip and windows stay inside.
    assert prepared.corpus.fingerprint == loaded.corpus.fingerprint
    assert prepared.metadata.duplicates_removed == 2
    hashes = [h for split in loaded.metadata.splits for h in split.sha256]
    assert len(hashes) == len(set(hashes)) == 8
    for tokens, split in zip(loaded.streams, loaded.metadata.splits, strict=True):
        assert split.offsets[0] == 0
        assert split.offsets[-1] == tokens.size
        for start in loaded.starts(split, 3):
            assert any(
                a <= start and start + 3 < b
                for a, b in zip(
                    split.offsets[:-1],
                    split.offsets[1:],
                    strict=True,
                )
            )
        for a, b in zip(split.offsets[:-1], split.offsets[1:], strict=True):
            assert (
                loaded.corpus.decode([int(tokens.item(i)) for i in range(a, b)])
                in stories
            )


def test_fingerprint_includes_boundaries(tmp_path: Path) -> None:
    # Given: a complete prepared artifact.
    raw = tmp_path / "raw.txt"
    _ = raw.write_text("\n<|endoftext|>\n".join(f"story {i} abc" for i in range(4)))
    prepared = prepare_stories(
        Preparation(
            raw_files=(raw,),
            source="synthetic",
            revision="fixture",
            license_text="CC0",
            train_stories=2,
            valid_stories=1,
            test_stories=1,
            vocab_size=257,
        )
    )
    train = prepared.metadata.train
    altered = train.model_copy(
        update={"offsets": (0, train.offsets[1] + 1, train.offsets[2])}
    )
    metadata = prepared.metadata.model_copy(update={"train": altered})
    corpus = prepared.corpus
    # When/Then: changing only boundaries invalidates the existing corpus identity.
    with pytest.raises(CorpusError, match="fingerprint"):
        _ = bpe_data.BPECorpus(
            corpus.vocabulary,
            corpus.train,
            corpus.valid,
            corpus.test,
            corpus.tokenizer_json,
            corpus.fingerprint,
            metadata.model_dump_json(),
        )


def test_train_only_bpe_and_bounded_inputs(tmp_path: Path) -> None:
    # Given: nonidentical stories and a strict raw-byte budget.
    raw = tmp_path / "raw.txt"
    _ = raw.write_text(
        "\n<|endoftext|>\n".join(f"unique tale {i} " * 4 for i in range(6))
    )
    options = Preparation(
        raw_files=(raw,),
        source="synthetic",
        revision="fixture",
        license_text="CC0",
        train_stories=4,
        valid_stories=1,
        test_stories=1,
        vocab_size=280,
    )
    # When: fitting BPE and independently fitting only the selected training text.
    prepared = prepare_stories(options)
    pieces = [
        prepared.corpus.decode(
            [int(prepared.corpus.train.item(i)) for i in range(a, b)]
        )
        for a, b in zip(
            prepared.metadata.train.offsets[:-1],
            prepared.metadata.train.offsets[1:],
            strict=True,
        )
    ]
    expected = fit_tokenizer("\n".join(pieces), 280)
    # Then: heldout text did not affect merges; undersized budgets fail explicitly.
    assert prepared.corpus.tokenizer_json == expected
    with pytest.raises(CorpusError, match="byte budget"):
        _ = prepare_stories(options.model_copy(update={"max_source_bytes": 10}))
    with pytest.raises(CorpusError, match="context"):
        _ = prepared.starts(prepared.metadata.train, 10000)
    assert all(stream.dtype == np.int64 for stream in prepared.streams)
