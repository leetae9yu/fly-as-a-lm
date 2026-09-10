"""Train-only character vocabulary and immutable, separated text splits."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, TypeAlias
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from pydantic import Field, TypeAdapter
from typing_extensions import override

from flyrl.models import Settings

if TYPE_CHECKING:
    from numpy import generic

IntVector: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int64]]
MAX_SYMBOLS: Final = 48


@dataclass(frozen=True, slots=True)
class CorpusError(ValueError):
    """An invalid corpus artifact or evaluation boundary."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class TextSplits:
    """Raw text already assigned to independent train, validation, and test sets."""

    train: str
    valid: str
    test: str
    provenance: str


class CorpusLimits(Settings):
    """Fixed preparation budget, chosen without consulting test performance."""

    max_symbols: Annotated[int, Field(ge=2, le=48)] = 48
    train_chars: Annotated[int, Field(ge=2)] = 250_000
    evaluation_chars: Annotated[int, Field(ge=2)] = 65_536


@dataclass(frozen=True, slots=True)
class Corpus:
    """Integer text splits with a vocabulary fitted only on the training prefix."""

    alphabet: tuple[str, ...]
    train: IntVector
    valid: IntVector
    test: IntVector
    fingerprint: str
    provenance: str

    def __post_init__(self) -> None:
        """Validate IDs once, then prevent accidental mutation of held-out text."""
        if (
            not self.alphabet
            or self.alphabet[0] != "?"
            or len(self.alphabet) > MAX_SYMBOLS
            or len(set(self.alphabet)) != len(self.alphabet)
            or any(len(symbol) != 1 for symbol in self.alphabet)
        ):
            raise CorpusError(reason="Invalid character alphabet or unknown symbol")
        for tokens in (self.train, self.valid, self.test):
            if tokens.ndim != 1 or tokens.dtype != np.int64 or tokens.size <= 1:
                raise CorpusError(reason="Each split needs at least two int64 tokens")
            if ((tokens < 0) | (tokens >= len(self.alphabet))).any():
                raise CorpusError(reason="Character index is outside the alphabet")
            tokens.flags.writeable = False


def corpus_fingerprint(
    alphabet: tuple[str, ...],
    streams: tuple[IntVector, IntVector, IntVector],
    provenance: str,
) -> str:
    """Hash content, split lengths, vocabulary order, and source identity."""
    digest = hashlib.sha256(json.dumps((alphabet, provenance)).encode())
    for stream in streams:
        digest.update(stream.size.to_bytes(8, "little"))
        digest.update(stream.astype("<i8").tobytes())
    return digest.hexdigest()


def make_corpus(text: TextSplits, limits: CorpusLimits) -> Corpus:
    """Normalize each split independently, fit training vocabulary, and encode."""
    normalized = (
        " ".join(text.train.lower().split())[: limits.train_chars],
        " ".join(text.valid.lower().split())[: limits.evaluation_chars],
        " ".join(text.test.lower().split())[: limits.evaluation_chars],
    )
    counts = Counter(normalized[0])
    ranked = sorted(
        (symbol for symbol in counts if symbol != "?"),
        key=lambda symbol: (-counts[symbol], symbol),
    )
    alphabet = ("?", *ranked[: limits.max_symbols - 1])
    lookup = {symbol: index for index, symbol in enumerate(alphabet)}
    train, valid, test = (
        np.asarray([lookup.get(symbol, 0) for symbol in split], dtype=np.int64)
        for split in normalized
    )
    fingerprint = corpus_fingerprint(alphabet, (train, valid, test), text.provenance)
    return Corpus(alphabet, train, valid, test, fingerprint, text.provenance)


def save_corpus(corpus: Corpus, path: Path) -> None:
    """Save plain NumPy arrays without pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = (
        ("alphabet", np.asarray(corpus.alphabet)),
        ("train", corpus.train),
        ("valid", corpus.valid),
        ("test", corpus.test),
        ("fingerprint", np.asarray(corpus.fingerprint)),
        ("provenance", np.asarray(corpus.provenance)),
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, array in arrays:
            with archive.open(f"{name}.npy", "w") as stream:
                write_array(stream, array, allow_pickle=False)


def load_corpus(path: Path) -> Corpus:
    """Validate stored arrays and their content fingerprint."""
    with path.open("rb") as stream:
        archive: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with archive as data:
            if any(data[name].dtype != np.int64 for name in ("train", "valid", "test")):
                raise CorpusError(reason="Corpus token arrays must be int64")
            alphabet = TypeAdapter(tuple[str, ...]).validate_python(
                data["alphabet"].tolist()
            )
            train = np.asarray(data["train"], dtype=np.int64)
            valid = np.asarray(data["valid"], dtype=np.int64)
            test = np.asarray(data["test"], dtype=np.int64)
            fingerprint = TypeAdapter(str).validate_python(data["fingerprint"].item())
            provenance = TypeAdapter(str).validate_python(data["provenance"].item())
    if fingerprint != corpus_fingerprint(alphabet, (train, valid, test), provenance):
        raise CorpusError(reason="Corpus content fingerprint mismatch")
    return Corpus(alphabet, train, valid, test, fingerprint, provenance)


def evaluation_starts(tokens: IntVector, context: int, limit: int) -> IntVector:
    """Select non-overlapping windows across a split, with targets inside it."""
    if context < 1 or limit < 1 or tokens.size <= context:
        raise CorpusError(
            reason="Evaluation needs a context, targets and a positive cap"
        )
    starts = np.arange(0, tokens.size - context, context + 1, dtype=np.int64)
    if starts.size <= limit:
        return starts
    indices = np.linspace(0, starts.size - 1, limit).astype(np.int64)
    return starts[indices]
