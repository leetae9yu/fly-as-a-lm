"""Official-source isolation and reproducible, story-safe quality artifacts."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from flyrl.bpe_data import save_bpe_corpus
from flyrl.bpe_tokenizer import fit_tokenizer
from flyrl.language_data import CorpusError
from flyrl.quality_corpus import (
    QualityPreparation,
    QualityReport,
    prepare_quality_corpus,
)
from flyrl.story_data import StoryCorpus, load_story_corpus
from scripts.prepare_quality_corpus import app


def _raw(path: Path, stories: list[str]) -> Path:
    _ = path.write_text("\n<|endoftext|>\n".join(stories), encoding="utf-8")
    return path


def _options(tmp_path: Path) -> QualityPreparation:
    train = _raw(
        tmp_path / "train.txt",
        [
            "  A bird  sings.\nInside\tspace stays.  ",
            "A bird  sings.\nInside\tspace stays.",
            "The fox dances in sunshine.",
            "Unused training quasar quasar quasar.",
        ],
    )
    valid = _raw(
        tmp_path / "valid.txt",
        [
            "The fox dances in sunshine.",
            "  Heldout café café café.  ",
            "Heldout café café café.",
            "A test robot hums softly.",
            "Another test robot likes snow.",
            "Unconsumed validation suffix.",
        ],
    )
    return QualityPreparation(
        raw_train=train,
        raw_valid=valid,
        source="synthetic official splits",
        revision="fixture-v1",
        license_text="Synthetic fixture; CC0\n",
        train_stories=2,
        valid_stories=1,
        test_stories=2,
    )


def _texts(stories: StoryCorpus) -> tuple[list[str], ...]:
    return tuple(
        [
            stories.corpus.decode([int(tokens.item(i)) for i in range(a, b)])
            for a, b in zip(split.offsets[:-1], split.offsets[1:], strict=True)
        ]
        for tokens, split in zip(stories.streams, stories.metadata.splits, strict=True)
    )


def test_official_split_order_and_exact_deduplication(tmp_path: Path) -> None:
    options = _options(tmp_path)
    prepared = prepare_quality_corpus(options)
    assert _texts(prepared) == (
        ["A bird  sings.\nInside\tspace stays.", "The fox dances in sunshine."],
        ["Heldout café café café."],
        ["A test robot hums softly.", "Another test robot likes snow."],
    )
    identities = [h for split in prepared.metadata.splits for h in split.sha256]
    assert len(identities) == len(set(identities)) == 5
    assert prepared.metadata.duplicates_removed == 3
    assert prepared.metadata.source == options.source
    assert prepared.metadata.revision == options.revision
    assert prepared.metadata.license_text == options.license_text
    for consumed in prepared.metadata.consumed:
        prefix = Path(consumed.path).read_bytes()[: consumed.bytes_consumed]
        assert consumed.prefix_sha256 == hashlib.sha256(prefix).hexdigest()
        assert consumed.bytes_consumed < Path(consumed.path).stat().st_size


def test_tokenizer_uses_only_selected_training(tmp_path: Path) -> None:
    options = _options(tmp_path)
    prepared = prepare_quality_corpus(options)
    train, valid, test = _texts(prepared)
    assert prepared.metadata.requested_vocab_size == 4096
    assert prepared.corpus.tokenizer_json == fit_tokenizer("\n".join(train), 4096)
    assert prepared.corpus.tokenizer_json != fit_tokenizer(
        "\n".join([*train, *valid, *test]), 4096
    )
    _ = _raw(
        options.raw_valid,
        [f"Different heldout xenoglossia xenoglossia story {i}." for i in range(3)],
    )
    changed = prepare_quality_corpus(options)
    assert changed.corpus.tokenizer_json == prepared.corpus.tokenizer_json
    assert np.array_equal(changed.corpus.train, prepared.corpus.train)
    assert changed.corpus.fingerprint != prepared.corpus.fingerprint


def test_boundaries_and_deterministic_roundtrip(tmp_path: Path) -> None:
    options = _options(tmp_path)
    first = prepare_quality_corpus(options)
    second = prepare_quality_corpus(options)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    save_bpe_corpus(first.corpus, a)
    save_bpe_corpus(second.corpus, b)
    assert a.read_bytes() == b.read_bytes()
    loaded = load_story_corpus(a)
    assert loaded.metadata == first.metadata
    assert loaded.corpus.fingerprint == first.corpus.fingerprint
    assert _texts(loaded) == _texts(first)
    for tokens, split, texts in zip(
        loaded.streams, loaded.metadata.splits, _texts(loaded), strict=True
    ):
        assert split.offsets[0] == 0
        assert split.offsets[-1] == tokens.size
        assert tokens.dtype == np.int64
        assert not tokens.flags.writeable
        for text, identity, start, end in zip(
            texts, split.sha256, split.offsets[:-1], split.offsets[1:], strict=True
        ):
            assert identity == hashlib.sha256(text.encode("utf-8")).hexdigest()
            assert list(tokens[start:end]) == loaded.corpus.encode(text)
        for start in loaded.starts(split, context=2):
            assert any(
                a <= start and start + 2 < b
                for a, b in zip(split.offsets[:-1], split.offsets[1:], strict=True)
            )


@pytest.mark.parametrize("split", ["train", "valid"])
def test_source_exhaustion_never_borrows_other_split(
    tmp_path: Path, split: str
) -> None:
    options = _options(tmp_path)
    path = options.raw_train if split == "train" else options.raw_valid
    _ = _raw(path, ["Only one unique story."] * 4)
    with pytest.raises(CorpusError, match="unique stories"):
        _ = prepare_quality_corpus(options)


def test_byte_budget_and_separate_files(tmp_path: Path) -> None:
    options = _options(tmp_path)
    with pytest.raises(CorpusError, match="byte budget"):
        _ = prepare_quality_corpus(options.model_copy(update={"max_source_bytes": 10}))
    with pytest.raises(CorpusError, match="separate"):
        _ = prepare_quality_corpus(
            options.model_copy(update={"raw_valid": options.raw_train})
        )


def test_empty_delimiters_and_final_eof_story(tmp_path: Path) -> None:
    options = _options(tmp_path)
    _ = _raw(options.raw_train, ["", "  ", "First tale.", "Second tale."])
    _ = _raw(options.raw_valid, ["", "Third tale.", "Fourth tale.", "Fifth tale."])
    prepared = prepare_quality_corpus(options)
    assert _texts(prepared) == (
        ["First tale.", "Second tale."],
        ["Third tale."],
        ["Fourth tale.", "Fifth tale."],
    )
    for consumed in prepared.metadata.consumed:
        assert consumed.bytes_consumed == Path(consumed.path).stat().st_size


def test_production_defaults(tmp_path: Path) -> None:
    options = QualityPreparation(
        raw_train=tmp_path / "train.txt",
        raw_valid=tmp_path / "valid.txt",
        source="synthetic",
        revision="v1",
        license_text="CC0",
    )
    assert (options.train_stories, options.valid_stories, options.test_stories) == (
        50_000,
        256,
        512,
    )


def test_cli_output_and_existing_directory_protection(tmp_path: Path) -> None:
    options = _options(tmp_path)
    license_file = tmp_path / "LICENSE.txt"
    _ = license_file.write_text(options.license_text, encoding="utf-8")
    output = tmp_path / "output"
    arguments = [
        "--raw-train",
        str(options.raw_train),
        "--raw-valid",
        str(options.raw_valid),
        "--source",
        options.source,
        "--revision",
        options.revision,
        "--license-file",
        str(license_file),
        "--output",
        str(output),
        "--train-stories",
        "2",
        "--valid-stories",
        "1",
        "--test-stories",
        "2",
    ]
    runner = CliRunner()
    result = runner.invoke(app, arguments, catch_exceptions=False)
    assert result.exit_code == 0
    assert {p.name for p in output.iterdir()} == {
        "corpus.npz",
        "tokenizer.json",
        "provenance.json",
        "SOURCE_LICENSE.txt",
    }
    loaded = load_story_corpus(output / "corpus.npz")
    report = QualityReport.model_validate_json(
        (output / "provenance.json").read_text(encoding="utf-8")
    )
    assert report.fingerprint == loaded.corpus.fingerprint
    assert report.fingerprint in result.stdout
    assert report.token_counts == {
        "train": loaded.corpus.train.size,
        "valid": loaded.corpus.valid.size,
        "test": loaded.corpus.test.size,
    }
    assert report.actual_vocab_size == len(loaded.corpus.vocabulary)
    assert (
        report.tokenizer_sha256
        == hashlib.sha256(loaded.corpus.tokenizer_json.encode("utf-8")).hexdigest()
    )
    assert report.raw_train == str(options.raw_train)
    assert report.raw_valid == str(options.raw_valid)
    assert (output / "tokenizer.json").read_text() == loaded.corpus.tokenizer_json
    assert (output / "SOURCE_LICENSE.txt").read_text() == options.license_text
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    refused = runner.invoke(app, arguments)
    assert refused.exit_code != 0
    assert isinstance(refused.exception, FileExistsError)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
