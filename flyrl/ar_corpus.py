"""Autoregressive corpus dispatch without weakening the legacy character format."""

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, assert_never

from numpy.lib.npyio import NpzFile
from pydantic import TypeAdapter

from flyrl.ar_config import Tokenization
from flyrl.bpe_data import BPECorpus, load_bpe_corpus
from flyrl.language_data import Corpus, load_corpus

if TYPE_CHECKING:
    from numpy import generic

ARCorpus: TypeAlias = Corpus | BPECorpus
CorpusFormat: TypeAlias = Literal["character", "flyrl-bpe-v1"]


def load_ar_corpus(path: Path) -> ARCorpus:
    """Dispatch by the versioned artifact marker; old character files have no marker."""
    with path.open("rb") as stream:
        archive: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with archive:
            format_name = TypeAdapter[CorpusFormat](CorpusFormat).validate_python(
                archive["format"].item() if "format" in archive.files else "character",
                strict=True,
            )
    match format_name:
        case "character":
            return load_corpus(path)
        case "flyrl-bpe-v1":
            return load_bpe_corpus(path)
        case _:
            assert_never(format_name)


def vocabulary(corpus: ARCorpus) -> tuple[str, ...]:
    """Return ordered model labels, which need not be independently decoded text."""
    match corpus:
        case Corpus():
            return corpus.alphabet
        case BPECorpus():
            return corpus.vocabulary
        case _:
            assert_never(corpus)


def tokenization(corpus: ARCorpus) -> Tokenization:
    """Identify prediction units from a validated corpus, not a user override."""
    match corpus:
        case Corpus():
            return "character"
        case BPECorpus():
            return "bpe"
        case _:
            assert_never(corpus)


def decode(corpus: ARCorpus, ids: tuple[int, ...]) -> str:
    """Decode complete token sequences; never concatenate BPE vocabulary labels."""
    match corpus:
        case Corpus():
            return "".join(corpus.alphabet[index] for index in ids)
        case BPECorpus():
            return corpus.decode(list(ids))
        case _:
            assert_never(corpus)
