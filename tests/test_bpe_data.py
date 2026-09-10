import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, assert_never
from zipfile import ZipFile

import numpy as np
import pytest
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from pydantic import TypeAdapter

from flyrl.bpe_data import (
    load_bpe_corpus,
    make_bpe_corpus,
    save_bpe_corpus,
)
from flyrl.language_data import CorpusError, TextSplits
from scripts.prepare_bpe_corpus import prepare
from scripts.prepare_language import CHECKSUMS

if TYPE_CHECKING:
    from numpy import generic

TamperField: TypeAlias = Literal[
    "tokenizer_json",
    "vocabulary",
    "train",
    "valid",
    "test",
    "fingerprint",
    "provenance",
    "format",
    "dtype",
    "shape",
    "out_of_range",
    "missing",
    "scalar_shape",
]


@pytest.fixture
def text() -> TextSplits:
    return TextSplits("ab ac ad ba ca da " * 8, "held out", "test text", "fixture")


def test_splits_are_exact_readonly_int64(text: TextSplits) -> None:
    # Given / When: separated source text is encoded.
    corpus = make_bpe_corpus(text)
    # Then: each stream retains its exact source and cannot be mutated.
    for source, stream in zip(
        (text.train, text.valid, text.test),
        (corpus.train, corpus.valid, corpus.test),
        strict=True,
    ):
        assert stream.dtype == np.int64
        assert corpus.decode(stream.tolist()) == source
        with pytest.raises(ValueError, match="read-only"):
            stream[0] = 0
    for name in ("fingerprint", "tokenizer_json"):
        with pytest.raises(FrozenInstanceError):
            setattr(corpus, name, "changed")


@pytest.mark.parametrize("size", [256, 65537])
def test_vocabulary_budget_rejects_invalid_sizes(text: TextSplits, size: int) -> None:
    # Given / When / Then: byte coverage cannot be underfunded or unbounded.
    with pytest.raises(CorpusError):
        _ = make_bpe_corpus(text, size)


@pytest.mark.parametrize("size", [257, 4096, 65536])
def test_tiny_training_may_underfill_budget(size: int) -> None:
    # Given / When: a tiny source cannot supply thousands of useful merges.
    corpus = make_bpe_corpus(TextSplits("a", "b", "c", "tiny"), size)
    # Then: all byte tokens and the unknown token still exist.
    assert len(corpus.vocabulary) == 257
    assert corpus.vocabulary[0] == "<unk>"


def test_npz_roundtrip_and_bytes_are_reproducible(
    tmp_path: Path, text: TextSplits
) -> None:
    # Given: a fitted corpus, saved twice.
    corpus = make_bpe_corpus(text)
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    save_bpe_corpus(corpus, first)
    save_bpe_corpus(corpus, second)
    # When: loading the unpickled artifact.
    restored = load_bpe_corpus(first)
    # Then: persisted identity and all tokens survive unchanged.
    assert first.read_bytes() == second.read_bytes()
    assert restored.vocabulary == corpus.vocabulary
    assert restored.tokenizer_json == corpus.tokenizer_json
    assert restored.provenance == corpus.provenance
    assert restored.fingerprint == corpus.fingerprint
    for actual, expected in zip(
        (restored.train, restored.valid, restored.test),
        (corpus.train, corpus.valid, corpus.test),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    with first.open("rb") as stream:
        archive: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with archive:
            assert archive["format"].shape == ()
            assert archive["format"].item() == "flyrl-bpe-v1"


@pytest.mark.parametrize(
    "field",
    [
        "tokenizer_json",
        "vocabulary",
        "train",
        "valid",
        "test",
        "fingerprint",
        "provenance",
        "format",
        "dtype",
        "shape",
        "out_of_range",
        "missing",
        "scalar_shape",
    ],
)
def test_tampered_archive_is_rejected(
    tmp_path: Path, text: TextSplits, field: TamperField
) -> None:
    # Given: a valid artifact with one corrupted field.
    path = tmp_path / "corpus.npz"
    save_bpe_corpus(make_bpe_corpus(text), path)
    with path.open("rb") as stream:
        archive: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with archive:
            arrays = {name: archive[name] for name in archive.files}
    match field:
        case "train" | "valid" | "test":
            arrays[field] = arrays[field].copy()
            arrays[field][0] = (
                TypeAdapter(int).validate_python(arrays[field].item(0)) + 1
            ) % len(arrays["vocabulary"])
        case "dtype":
            arrays["train"] = arrays["train"].astype(np.float64)
        case "shape":
            arrays["train"] = arrays["train"].reshape(1, -1)
        case "out_of_range":
            arrays["train"] = np.asarray([-1, 999999], dtype=np.int64)
        case "missing":
            del arrays["tokenizer_json"]
        case "scalar_shape":
            arrays["format"] = arrays["format"].reshape(1)
        case "vocabulary":
            arrays[field] = arrays[field][::-1]
        case "tokenizer_json" | "fingerprint" | "provenance" | "format":
            arrays[field] = np.asarray("tampered")
        case _:
            assert_never(field)
    with ZipFile(path, "w") as modified:
        for name, array in arrays.items():
            with modified.open(f"{name}.npy", "w") as stream:
                write_array(stream, array, allow_pickle=False)
    # When / Then: no corrupt artifact reaches training.
    with pytest.raises(CorpusError):
        _ = load_bpe_corpus(path)


def test_tokenizer_serialization_participates_in_fingerprint(text: TextSplits) -> None:
    # Given: a valid corpus and semantically identical but changed JSON bytes.
    corpus = make_bpe_corpus(text)
    # When / Then: retaining the old fingerprint cannot hide the change.
    with pytest.raises(CorpusError):
        _ = replace(corpus, tokenizer_json=corpus.tokenizer_json + " ")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('"lstrip":false', '"lstrip":true'),
        ('"rstrip":false', '"rstrip":true'),
        ('"add_prefix_space":false', '"add_prefix_space":true'),
        ('"<unk>":0', '"<unk>":1'),
    ],
)
def test_inconsistent_tokenizer_is_rejected_even_with_matching_hash(
    text: TextSplits, old: str, new: str
) -> None:
    # Given: a self-consistently hashed tokenizer with invalid IDs or lossy settings.
    corpus = make_bpe_corpus(text)
    serialized = corpus.tokenizer_json.replace(old, new)
    digest = hashlib.sha256(
        json.dumps(("flyrl-bpe-v1", serialized, corpus.provenance)).encode()
    )
    for tokens in (corpus.train, corpus.valid, corpus.test):
        digest.update(tokens.size.to_bytes(8, "little"))
        digest.update(tokens.astype("<i8").tobytes())
    fingerprint = digest.hexdigest()
    # When / Then: hash integrity cannot replace validation of tokenizer semantics.
    with pytest.raises(CorpusError):
        _ = replace(corpus, tokenizer_json=serialized, fingerprint=fingerprint)


def test_invalid_generated_bytes_do_not_relabel_vocabulary(text: TextSplits) -> None:
    # Given: the byte 255 label is not a standalone valid UTF-8 sequence.
    corpus = make_bpe_corpus(text)
    index = corpus.vocabulary.index("\u00ff")
    # When / Then: display uses replacement text but keeps the raw model label intact.
    assert corpus.decode([index]) == "\ufffd"
    assert corpus.vocabulary[index] == "\u00ff"


def test_preparation_uses_verified_fresh_normalized_slices(tmp_path: Path) -> None:
    # Given: pinned local raw files, never downloaded during preparation.
    root = Path(__file__).resolve().parents[1]
    raw = tmp_path / "data" / "wikitext2" / "raw"
    raw.mkdir(parents=True)
    for name in CHECKSUMS:
        (raw / f"{name}.txt").symlink_to(
            root / "data" / "wikitext2" / "raw" / f"{name}.txt"
        )
    # When: the real preparation entry point runs.
    corpus = prepare(tmp_path)
    # Then: exact normalized boundaries and companion identity are published.
    for (name, start, end), tokens in zip(
        (("train", 0, 250000), ("valid", 0, 65536), ("test", 131072, 196608)),
        (corpus.train, corpus.valid, corpus.test),
        strict=True,
    ):
        normalized = " ".join((raw / f"{name}.txt").read_text().lower().split())
        assert corpus.decode(tokens.tolist()) == normalized[start:end]
    directory = tmp_path / "data" / "bpe_corpus"
    assert (directory / "tokenizer.json").read_text() == corpus.tokenizer_json
    report = TypeAdapter(
        dict[str, str | int | dict[str, str | int | list[int]]]
    ).validate_json((directory / "provenance.json").read_text())
    assert report["fingerprint"] == corpus.fingerprint
    assert report["requested_vocab_size"] == report["actual_vocab_size"] == 4096
    assert (
        report["tokenizer_sha256"]
        == hashlib.sha256(corpus.tokenizer_json.encode()).hexdigest()
    )


def test_preparation_rejects_source_checksum_mismatch(tmp_path: Path) -> None:
    # Given: a substituted local training file.
    raw = tmp_path / "data" / "wikitext2" / "raw"
    raw.mkdir(parents=True)
    _ = (raw / "train.txt").write_text("wrong source")
    # When / Then: preparation fails before writing any corpus.
    with pytest.raises(CorpusError):
        _ = prepare(tmp_path)
    assert not (tmp_path / "data" / "bpe_corpus").exists()
