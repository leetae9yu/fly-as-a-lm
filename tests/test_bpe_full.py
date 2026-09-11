import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from flyrl.bpe_data import load_bpe_corpus
from flyrl.bpe_tokenizer import fit_tokenizer
from flyrl.language_data import CorpusError, TextSplits
from scripts import prepare_bpe_full


def _install_sources(
    root: Path, text: TextSplits, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = root / "data" / "wikitext2" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for name, contents in zip(
        ("train", "valid", "test"), (text.train, text.valid, text.test), strict=True
    ):
        payload = contents.encode("utf-8")
        _ = (raw / f"{name}.txt").write_bytes(payload)
        checksums[name] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(prepare_bpe_full, "CHECKSUMS", checksums)


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TextSplits:
    text = TextSplits(
        "A " * 125_000 + "ZYXWVUT \t\n" * 1024 + "caf\u00e9 <unk> @-@",
        "Validation \n\t" * 8192 + "UNSEENVALID",
        "X" * 196_608 + " FreshTEST \n\t" * 64 + "\u03a9",
        "fixture",
    )
    _install_sources(tmp_path, text, monkeypatch)
    return text


def test_preparation_keeps_full_normalized_splits(
    tmp_path: Path, source: TextSplits
) -> None:
    # Given: training and validation exceed every historical prefix budget.
    expected = tuple(
        " ".join(text.lower().split())
        for text in (source.train, source.valid, source.test)
    )
    # When: the public preparation API runs with its default vocabulary budget.
    corpus = prepare_bpe_full.prepare(tmp_path)
    # Then: exact full train/valid and only fresh test characters roundtrip.
    assert len(expected[0]) > 250_000
    assert len(expected[1]) > 65_536
    for original, tokens in zip(
        (expected[0], expected[1], expected[2][196_608:]),
        (corpus.train, corpus.valid, corpus.test),
        strict=True,
    ):
        assert corpus.decode(tokens.tolist()) == original
    assert corpus.tokenizer_json == fit_tokenizer(expected[0], 4096)
    assert corpus.tokenizer_json != fit_tokenizer(expected[0][:250_000], 4096)
    identity = prepare_bpe_full.FullSourceIdentity.model_validate_json(
        corpus.provenance
    )
    assert identity.split_bounds == {
        "train": (0, len(expected[0])),
        "valid": (0, len(expected[1])),
        "test": (196_608, len(expected[2])),
    }


@pytest.mark.parametrize("split", ["valid", "test"])
def test_heldout_changes_cannot_change_vocabulary(
    tmp_path: Path, source: TextSplits, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    # Given: one preparation and changed held-out bytes with unchanged training.
    first = prepare_bpe_full.prepare(tmp_path, vocab_size=300)
    changed = replace(source, **{split: "H" * 196_608 + " heldoutonly" * 1024})
    _install_sources(tmp_path, changed, monkeypatch)
    # When: preparing with the same explicit vocabulary budget.
    second = prepare_bpe_full.prepare(tmp_path, vocab_size=300)
    # Then: the tokenizer stays identical while the content identity changes.
    assert first.tokenizer_json == second.tokenizer_json
    assert first.vocabulary == second.vocabulary
    assert first.fingerprint != second.fingerprint


@pytest.mark.usefixtures("source")
def test_artifacts_are_reproducible_and_old_datasets_survive(tmp_path: Path) -> None:
    # Given: pre-existing historical datasets and a completed full preparation.
    old: list[Path] = []
    for directory in ("wikitext2", "ar_corpus", "bpe_corpus"):
        path = tmp_path / "data" / directory
        path.mkdir(parents=True, exist_ok=True)
        for name in ("corpus.npz", "tokenizer.json", "provenance.json"):
            artifact = path / name
            _ = artifact.write_bytes(b"historical artifact")
            old.append(artifact)
    first = prepare_bpe_full.prepare(tmp_path)
    directory = tmp_path / "data" / "bpe_full"
    artifacts = {path.name: path.read_bytes() for path in directory.iterdir()}
    # When: deterministically preparing again through the same public API.
    second = prepare_bpe_full.prepare(tmp_path)
    # Then: all full artifacts reproduce byte-for-byte without touching old data.
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == artifacts
    assert set(artifacts) == {"corpus.npz", "tokenizer.json", "provenance.json"}
    assert all(path.read_bytes() == b"historical artifact" for path in old)
    restored = load_bpe_corpus(directory / "corpus.npz")
    assert restored.fingerprint == first.fingerprint == second.fingerprint
    assert artifacts["tokenizer.json"].decode() == first.tokenizer_json
    report = prepare_bpe_full.FullPreparationReport.model_validate_json(
        artifacts["provenance.json"]
    )
    assert report.fingerprint == first.fingerprint
    assert report.raw_sha256 == {
        name: hashlib.sha256(
            (tmp_path / "data" / "wikitext2" / "raw" / f"{name}.txt").read_bytes()
        ).hexdigest()
        for name in ("train", "valid", "test")
    }
    assert report.requested_vocab_size == 4096
    assert report.actual_vocab_size == len(first.vocabulary)
    assert report.token_counts == {
        "train": first.train.size,
        "valid": first.valid.size,
        "test": first.test.size,
    }
    assert (
        report.tokenizer_sha256
        == hashlib.sha256(artifacts["tokenizer.json"]).hexdigest()
    )


@pytest.mark.usefixtures("source")
@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_checksum_failure_leaves_outputs_untouched(tmp_path: Path, split: str) -> None:
    # Given: a substituted source but the original checksum pins.
    raw = tmp_path / "data" / "wikitext2" / "raw"
    _ = (raw / f"{split}.txt").write_text("substitution", encoding="utf-8")
    # When / Then: no output is written from unverified bytes.
    with pytest.raises(CorpusError, match="checksum mismatch"):
        _ = prepare_bpe_full.prepare(tmp_path)
    assert not (tmp_path / "data" / "bpe_full").exists()


@pytest.mark.parametrize(
    ("split", "contents"),
    [
        ("train", " \n\t"),
        ("valid", " \n\t"),
        ("test", "X" * 196_607),
        ("test", "X" * 196_608),
    ],
)
def test_empty_or_exhausted_normalized_region_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str, contents: str
) -> None:
    # Given: checksum-valid bytes whose normalized selection is empty or too short.
    text = replace(
        TextSplits("train", "valid", "X" * 196_609, "fixture"), **{split: contents}
    )
    _install_sources(tmp_path, text, monkeypatch)
    # When / Then: the preparation boundary rejects the region before publication.
    with pytest.raises(CorpusError):
        _ = prepare_bpe_full.prepare(tmp_path)
    assert not (tmp_path / "data" / "bpe_full").exists()


def test_first_fresh_character_is_an_inclusive_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: precisely one normalized character beyond the historical test end.
    _install_sources(
        tmp_path,
        TextSplits("train", "valid", "X" * 196_608 + "Y", "fixture"),
        monkeypatch,
    )
    # When: selecting the smallest possible fresh region.
    corpus = prepare_bpe_full.prepare(tmp_path, vocab_size=257)
    # Then: the boundary character is retained, not excluded or overlapped.
    assert corpus.decode(corpus.test.tolist()) == "y"


@pytest.mark.usefixtures("source")
@pytest.mark.parametrize("vocab_size", [256, 65_537])
def test_invalid_vocabulary_budget_is_rejected(tmp_path: Path, vocab_size: int) -> None:
    # Given / When / Then: the public API preserves the BPE budget constraints.
    with pytest.raises(CorpusError):
        _ = prepare_bpe_full.prepare(tmp_path, vocab_size=vocab_size)
    assert not (tmp_path / "data" / "bpe_full").exists()
