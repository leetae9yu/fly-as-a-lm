# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "pydantic>=2.10,<3"]
# ///
# Run from the project root: uv run python -m scripts.prepare_language
"""Prepare pinned WikiText-2 prefixes and matched evaluation baselines."""

import hashlib
from pathlib import Path
from typing import Final

import numpy as np

from flyrl.language_baselines import BaselineScore, fit_baselines, score_baselines
from flyrl.language_data import (
    Corpus,
    CorpusError,
    CorpusLimits,
    TextSplits,
    evaluation_starts,
    make_corpus,
    save_corpus,
)
from flyrl.models import Settings

COMMIT: Final = "acc295dc7b90714f1bf47f06004fc19a7fe235c4"
SOURCE: Final = (
    f"https://raw.githubusercontent.com/pytorch/examples/{COMMIT}"
    "/word_language_model/data/wikitext-2"
)
CHECKSUMS: Final = {
    "train": "9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f",
    "valid": "f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8",
    "test": "d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0",
}


class BaselineReport(Settings):
    """Fixed target selection shared by language experiments and references."""

    corpus_fingerprint: str
    context: int = 16
    evaluation_windows: int = 2048
    theoretical_uniform_accuracy: float
    train: tuple[BaselineScore, ...]
    valid: tuple[BaselineScore, ...]
    test: tuple[BaselineScore, ...]


class CorpusReport(Settings):
    """Reproducible language-data identity and preprocessing assumptions."""

    source: str = SOURCE
    commit: str = COMMIT
    raw_sha256: dict[str, str]
    fingerprint: str
    limits: CorpusLimits
    alphabet: tuple[str, ...]
    split_lengths: dict[str, int]
    unknown_token_fractions: dict[str, float]
    normalization: str = "lowercase; collapse whitespace independently per split"
    vocabulary: str = "training prefix only; top 47 characters plus reserved '?'"
    source_form: str = "tokenized WikiText-2 with <unk> and @-@ markers retained"
    selection: str = "fixed split prefixes; no performance-dependent data selection"


def prepare(root: Path) -> Corpus:
    """Check source identity, encode split prefixes, and save baseline metrics."""
    directory = root / "data" / "wikitext2"
    raw = directory / "raw"
    for split, expected in CHECKSUMS.items():
        actual = hashlib.sha256((raw / f"{split}.txt").read_bytes()).hexdigest()
        if actual != expected:
            raise CorpusError(reason=f"WikiText-2 source checksum mismatch: {split}")
    limits = CorpusLimits()
    corpus = make_corpus(
        TextSplits(
            (raw / "train.txt").read_text(),
            (raw / "valid.txt").read_text(),
            (raw / "test.txt").read_text(),
            f"WikiText-2 tokenized; pytorch_examples_commit={COMMIT}; fixed-prefixes",
        ),
        limits,
    )
    save_corpus(corpus, directory / "corpus.npz")
    tokens = (corpus.train, corpus.valid, corpus.test)
    split_names = ("train", "valid", "test")
    report = CorpusReport(
        raw_sha256=CHECKSUMS,
        fingerprint=corpus.fingerprint,
        limits=limits,
        alphabet=corpus.alphabet,
        split_lengths={
            name: split.size for name, split in zip(split_names, tokens, strict=True)
        },
        unknown_token_fractions={
            name: float(np.equal(split, 0).mean())
            for name, split in zip(split_names, tokens, strict=True)
        },
    )
    _ = (directory / "provenance.json").write_text(report.model_dump_json(indent=2))
    models = fit_baselines(corpus)
    train, valid, test = (
        score_baselines(models, split, evaluation_starts(split, 16, 2048) + 16)
        for split in tokens
    )
    baselines = BaselineReport(
        corpus_fingerprint=corpus.fingerprint,
        theoretical_uniform_accuracy=1 / len(corpus.alphabet),
        train=train,
        valid=valid,
        test=test,
    )
    _ = (directory / "baselines.json").write_text(baselines.model_dump_json(indent=2))
    return corpus


if __name__ == "__main__":
    _ = prepare(Path(__file__).resolve().parents[1])
