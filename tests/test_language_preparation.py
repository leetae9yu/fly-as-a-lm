import shutil
from pathlib import Path

import pytest

from flyrl.language_data import CorpusError, load_corpus
from scripts.prepare_language import prepare


def test_wikitext_preparation_keeps_prespecified_budgets(tmp_path: Path) -> None:
    # Given: the original separately downloaded WikiText-2 files.
    raw = Path(__file__).resolve().parents[1] / "data" / "wikitext2" / "raw"
    _ = shutil.copytree(raw, tmp_path / "data" / "wikitext2" / "raw")
    # When: preparing the real language task.
    corpus = prepare(tmp_path)
    # Then: every split has its prespecified budget, with a reloadable artifact.
    restored = load_corpus(tmp_path / "data" / "wikitext2" / "corpus.npz")
    assert corpus.train.size == 250_000
    assert corpus.valid.size == corpus.test.size == 65_536
    assert len(corpus.alphabet) == 48
    assert restored.fingerprint == corpus.fingerprint
    assert (tmp_path / "data" / "wikitext2" / "baselines.json").exists()


def test_changed_wikitext_source_is_rejected(tmp_path: Path) -> None:
    # Given: a changed training file under a pinned dataset name.
    raw = tmp_path / "data" / "wikitext2" / "raw"
    raw.mkdir(parents=True)
    _ = (raw / "train.txt").write_text("This is not the pinned training corpus.")
    # When / Then: preparation rejects it before fitting any vocabulary.
    with pytest.raises(CorpusError):
        _ = prepare(tmp_path)
