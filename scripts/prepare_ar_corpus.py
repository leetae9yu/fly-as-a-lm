# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "pydantic>=2.10,<3"]
# ///
# Run from the project root: uv run --no-sync python -m scripts.prepare_ar_corpus
"""Keep training data fixed and reserve a fresh final-test text region."""

import hashlib
from pathlib import Path

import numpy as np

from flyrl.language_data import (
    Corpus,
    CorpusError,
    corpus_fingerprint,
    load_corpus,
    save_corpus,
)
from flyrl.models import Settings
from scripts.prepare_language import CHECKSUMS, SOURCE


class RegionReport(Settings):
    """Final-test identity; training and validation remain the existing prefixes."""

    base_fingerprint: str
    fingerprint: str
    raw_test_sha256: str
    raw_test_url: str
    normalized_test_start: int
    normalized_test_end_exclusive: int
    selection: str = "next equal-length region after the previously evaluated prefix"


def reserve_test_region(base: Corpus, raw_test: str, start: int) -> Corpus:
    """Keep vocabulary/train/valid fixed and replace only the final-test region."""
    normalized = " ".join(raw_test.lower().split())
    end = start + base.test.size
    if start < base.test.size or end > len(normalized):
        raise CorpusError(reason="Final-test region overlaps or exceeds its source")
    lookup = {symbol: index for index, symbol in enumerate(base.alphabet)}
    test = np.asarray(
        [lookup.get(symbol, 0) for symbol in normalized[start:end]], dtype=np.int64
    )
    provenance = f"{base.provenance}; final_test_normalized_slice={start}:{end}"
    fingerprint = corpus_fingerprint(
        base.alphabet, (base.train, base.valid, test), provenance
    )
    return Corpus(base.alphabet, base.train, base.valid, test, fingerprint, provenance)


def prepare(root: Path) -> Corpus:
    """Verify the original test source and save the new held-out corpus."""
    base = load_corpus(root / "data" / "wikitext2" / "corpus.npz")
    raw = (root / "data" / "wikitext2" / "raw" / "test.txt").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CHECKSUMS["test"]:
        raise CorpusError(reason="Original WikiText-2 test source checksum mismatch")
    corpus = reserve_test_region(base, raw.decode(), base.test.size)
    directory = root / "data" / "ar_corpus"
    save_corpus(corpus, directory / "corpus.npz")
    report = RegionReport(
        base_fingerprint=base.fingerprint,
        fingerprint=corpus.fingerprint,
        raw_test_sha256=digest,
        raw_test_url=f"{SOURCE}/test.txt",
        normalized_test_start=base.test.size,
        normalized_test_end_exclusive=2 * base.test.size,
    )
    _ = (directory / "provenance.json").write_text(report.model_dump_json(indent=2))
    return corpus


if __name__ == "__main__":
    _ = prepare(Path(__file__).resolve().parents[1])
