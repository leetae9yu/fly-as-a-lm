# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2,<3", "pydantic>=2.10,<3", "tokenizers==0.22.2", "typer>=0.15,<1"
# ]
# ///
# Run from the project root: uv run --no-sync python -m scripts.prepare_bpe_full
"""Prepare all pinned WikiText-2 training text and a fresh final-test suffix."""

import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Final

import typer

from flyrl.bpe_data import BPECorpus, make_bpe_corpus, save_bpe_corpus
from flyrl.language_data import CorpusError, TextSplits
from scripts.prepare_bpe_corpus import (
    SPLIT_BOUNDS,
    PreparationReport,
    SourceIdentity,
)
from scripts.prepare_language import CHECKSUMS

SPLIT_STARTS: Final = {"train": 0, "valid": 0, "test": SPLIT_BOUNDS["test"][1]}


class FullSourceIdentity(SourceIdentity):
    """Full split selection, unchanged byte BPE settings, and upstream attribution."""

    vocabulary_fit: str = (
        "all normalized training text only; all 256 initial bytes plus <unk>; "
        "BPE minimum pair frequency 2"
    )
    source_attribution: str = (
        "WikiText-2: Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher; "
        "Wikipedia contributors"
    )
    source_license: str = (
        "Upstream dataset card metadata: CC BY-SA 3.0/GFDL; "
        "licensing discussion links CC BY-SA 4.0. "
        "Original upstream attribution/share-alike terms apply, not project MIT."
    )
    source_license_url: str = "https://huggingface.co/datasets/Salesforce/wikitext"
    source_license_record: str = "data/wikitext2/README.md; THIRD_PARTY.md"


class FullPreparationReport(PreparationReport, FullSourceIdentity):
    """Published full-corpus identity and resulting token counts and hashes."""


def prepare(root: Path, vocab_size: int = 4096) -> BPECorpus:
    """Verify raw bytes, fit all train text, and publish only in data/bpe_full."""
    raw = root / "data" / "wikitext2" / "raw"
    selected: dict[str, str] = {}
    bounds: dict[str, tuple[int, int]] = {}
    for name, start in SPLIT_STARTS.items():
        payload = (raw / f"{name}.txt").read_bytes()
        if hashlib.sha256(payload).hexdigest() != CHECKSUMS[name]:
            raise CorpusError(reason=f"WikiText-2 source checksum mismatch: {name}")
        normalized = " ".join(payload.decode("utf-8").lower().split())
        if len(normalized) <= start:
            raise CorpusError(reason=f"Empty normalized WikiText-2 region: {name}")
        selected[name] = normalized[start:]
        bounds[name] = (start, len(normalized))
    identity = FullSourceIdentity(
        raw_sha256=CHECKSUMS,
        split_bounds=bounds,
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
    report = FullPreparationReport.model_validate(
        {
            **identity.model_dump(),
            "actual_vocab_size": len(corpus.vocabulary),
            "token_counts": {
                "train": corpus.train.size,
                "valid": corpus.valid.size,
                "test": corpus.test.size,
            },
            "tokenizer_sha256": hashlib.sha256(
                corpus.tokenizer_json.encode("utf-8")
            ).hexdigest(),
            "fingerprint": corpus.fingerprint,
        }
    )
    directory = root / "data" / "bpe_full"
    save_bpe_corpus(corpus, directory / "corpus.npz")
    for name, contents in (
        ("tokenizer.json", corpus.tokenizer_json),
        ("provenance.json", report.model_dump_json(indent=2)),
    ):
        _ = (directory / name).write_text(contents, encoding="utf-8")
    return corpus


def main(vocab_size: Annotated[int, typer.Option(min=257, max=65_536)] = 4096) -> None:
    """Publish the full-training corpus from already-present checksum-pinned files."""
    corpus = prepare(Path(__file__).resolve().parents[1], vocab_size)
    typer.echo(
        " ".join(
            (
                f"BPE full vocabulary={len(corpus.vocabulary)}",
                f"train={corpus.train.size}",
                f"valid={corpus.valid.size}",
                f"test={corpus.test.size}",
                f"fingerprint={corpus.fingerprint}",
            )
        )
    )


if __name__ == "__main__":
    typer.run(main)
