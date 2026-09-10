# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2,<3", "pydantic>=2.10,<3", "tokenizers==0.22.2", "typer>=0.15,<1"
# ]
# ///
# Run from the project root: uv run --no-sync python -m scripts.prepare_bpe_corpus
"""Prepare pinned local WikiText-2 slices using train-only byte-level BPE."""

import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Final

import typer

from flyrl.bpe_data import BPECorpus, make_bpe_corpus, save_bpe_corpus
from flyrl.language_data import CorpusError, TextSplits
from flyrl.models import Settings
from scripts.prepare_language import CHECKSUMS, SOURCE

SPLIT_BOUNDS: Final = {
    "train": (0, 250_000),
    "valid": (0, 65_536),
    "test": (131_072, 196_608),
}


class SourceIdentity(Settings):
    """Fixed source and normalization choices included in the corpus fingerprint."""

    source: str = SOURCE
    raw_sha256: dict[str, str]
    split_bounds: dict[str, tuple[int, int]]
    requested_vocab_size: int
    tokenizers_version: str
    normalization: str = (
        "lowercase; collapse whitespace independently per split; "
        "slice normalized Unicode characters with exclusive end"
    )
    vocabulary_fit: str = (
        "training slice only; all 256 initial bytes plus <unk>; "
        "BPE minimum pair frequency 2"
    )
    source_form: str = "tokenized WikiText-2; literal <unk> and @-@ markers retained"
    decoding: str = (
        "exact source roundtrip; arbitrary generated invalid UTF-8 renders as U+FFFD; "
        "vocabulary labels remain byte-level tokens"
    )


class PreparationReport(SourceIdentity):
    """Published source identity plus the resulting tokenizer and corpus content."""

    actual_vocab_size: int
    token_counts: dict[str, int]
    tokenizer_sha256: str
    fingerprint: str


def prepare(root: Path, vocab_size: int = 4096) -> BPECorpus:
    """Verify local source bytes, fit on train, and publish all three artifacts."""
    raw = root / "data" / "wikitext2" / "raw"
    selected: dict[str, str] = {}
    for name, expected in CHECKSUMS.items():
        contents = (raw / f"{name}.txt").read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected:
            raise CorpusError(reason=f"WikiText-2 source checksum mismatch: {name}")
        normalized = " ".join(contents.decode("utf-8").lower().split())
        start, end = SPLIT_BOUNDS[name]
        if len(normalized) < end:
            raise CorpusError(reason=f"Incomplete normalized WikiText-2 split: {name}")
        selected[name] = normalized[start:end]
    identity = SourceIdentity(
        raw_sha256=CHECKSUMS,
        split_bounds=SPLIT_BOUNDS,
        requested_vocab_size=vocab_size,
        tokenizers_version=version("tokenizers"),
    )
    corpus = make_bpe_corpus(
        TextSplits(
            selected["train"],
            selected["valid"],
            selected["test"],
            identity.model_dump_json(),
        ),
        vocab_size,
    )
    directory = root / "data" / "bpe_corpus"
    save_bpe_corpus(corpus, directory / "corpus.npz")
    _ = (directory / "tokenizer.json").write_text(
        corpus.tokenizer_json, encoding="utf-8"
    )
    report = PreparationReport.model_validate(
        {
            **identity.model_dump(),
            "actual_vocab_size": len(corpus.vocabulary),
            "token_counts": {
                "train": corpus.train.size,
                "valid": corpus.valid.size,
                "test": corpus.test.size,
            },
            "tokenizer_sha256": hashlib.sha256(
                corpus.tokenizer_json.encode()
            ).hexdigest(),
            "fingerprint": corpus.fingerprint,
        }
    )
    _ = (directory / "provenance.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    return corpus


def main(vocab_size: Annotated[int, typer.Option(min=257, max=65_536)] = 4096) -> None:
    """Build a corpus from already-present checksum-pinned WikiText-2 sources."""
    corpus = prepare(Path(__file__).resolve().parents[1], vocab_size)
    typer.echo(
        f"BPE vocabulary={len(corpus.vocabulary)} fingerprint={corpus.fingerprint}"
    )


if __name__ == "__main__":
    typer.run(main)
