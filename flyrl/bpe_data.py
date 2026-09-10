"""Train-only byte BPE, exact text encoding, and immutable versioned artifacts.

Vocabulary entries are byte-level model labels, not independently decoded text.
Arbitrary generated token sequences can contain invalid UTF-8 and decode to U+FFFD;
never use their decoded strings to relabel vocabulary IDs.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from pydantic import JsonValue, TypeAdapter

from flyrl.bpe_tokenizer import (
    BPETokenizer,
    fit_tokenizer,
    initial_alphabet,
    restore_tokenizer,
)
from flyrl.language_data import CorpusError, IntVector, TextSplits

if TYPE_CHECKING:
    from numpy import generic

FORMAT: Final = "flyrl-bpe-v1"
MIN_VOCAB: Final = 257
MAX_VOCAB: Final = 65_536


def _fingerprint(
    tokenizer_json: str,
    streams: tuple[IntVector, IntVector, IntVector],
    provenance: str,
) -> str:
    digest = hashlib.sha256(json.dumps((FORMAT, tokenizer_json, provenance)).encode())
    for stream in streams:
        digest.update(stream.size.to_bytes(8, "little"))
        digest.update(stream.astype("<i8").tobytes())
    return digest.hexdigest()


def _validate_tokenizer(serialized: str, vocabulary: tuple[str, ...]) -> BPETokenizer:
    document = TypeAdapter(dict[str, JsonValue]).validate_json(serialized)
    model = TypeAdapter(dict[str, JsonValue]).validate_python(document.get("model"))
    pre = TypeAdapter(dict[str, JsonValue]).validate_python(
        document.get("pre_tokenizer")
    )
    decoder = TypeAdapter(dict[str, JsonValue]).validate_python(document.get("decoder"))
    if (
        model.get("type") != "BPE"
        or model.get("unk_token") != "<unk>"
        or model.get("dropout") is not None
        or model.get("continuing_subword_prefix") not in (None, "")
        or model.get("end_of_word_suffix") not in (None, "")
        or pre.get("type") != "ByteLevel"
        or pre.get("add_prefix_space") is not False
        or decoder.get("type") != "ByteLevel"
        or any(
            document.get(key) is not None
            for key in ("normalizer", "padding", "truncation")
        )
    ):
        raise CorpusError(reason="Tokenizer is not lossless deterministic byte BPE")
    added = [
        {
            "id": 0,
            "content": "<unk>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    ]
    if document.get("added_tokens") != added:
        raise CorpusError(reason="BPE unknown token must preserve literal source text")
    expected = {token: index for index, token in enumerate(vocabulary)}
    stored = TypeAdapter(dict[str, int]).validate_python(
        model.get("vocab"), strict=True
    )
    if stored != expected or vocabulary[:MIN_VOCAB] != ("<unk>", *initial_alphabet()):
        raise CorpusError(
            reason="Tokenizer vocabulary IDs are not contiguous or consistent"
        )
    tokenizer = restore_tokenizer(serialized)
    if tokenizer.get_vocab() != expected:
        raise CorpusError(reason="Added tokenizer tokens disagree with vocabulary IDs")
    return tokenizer


@dataclass(frozen=True, slots=True)
class BPECorpus:
    """Separated immutable int64 streams with the fitted tokenizer's exact identity."""

    vocabulary: tuple[str, ...]
    train: IntVector
    valid: IntVector
    test: IntVector
    tokenizer_json: str
    fingerprint: str
    provenance: str
    _tokenizer: BPETokenizer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate artifact boundaries and own immutable copies of all token arrays."""
        if not MIN_VOCAB <= len(self.vocabulary) <= MAX_VOCAB or len(
            set(self.vocabulary)
        ) != len(self.vocabulary):
            raise CorpusError(reason="Invalid BPE vocabulary size or duplicate labels")
        for tokens in (self.train, self.valid, self.test):
            if tokens.ndim != 1 or tokens.dtype != np.int64 or tokens.size == 0:
                raise CorpusError(reason="Each BPE split needs a nonempty int64 vector")
            if ((tokens < 0) | (tokens >= len(self.vocabulary))).any():
                raise CorpusError(reason="BPE token ID is outside the vocabulary")
        if self.fingerprint != _fingerprint(
            self.tokenizer_json, (self.train, self.valid, self.test), self.provenance
        ):
            raise CorpusError(reason="BPE corpus content fingerprint mismatch")
        object.__setattr__(
            self,
            "_tokenizer",
            _validate_tokenizer(self.tokenizer_json, self.vocabulary),
        )
        for name, tokens in zip(
            ("train", "valid", "test"), (self.train, self.valid, self.test), strict=True
        ):
            immutable = np.ndarray(
                tokens.shape, dtype=np.int64, buffer=tokens.tobytes()
            )
            object.__setattr__(self, name, immutable)

    def encode(self, text: str) -> list[int]:
        """Encode exact source text, including unseen Unicode and whitespace."""
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        """Keep literal <unk>; invalid generated UTF-8 bytes render as U+FFFD."""
        if any(index < 0 or index >= len(self.vocabulary) for index in ids):
            raise CorpusError(reason="BPE decode ID is outside the vocabulary")
        return self._tokenizer.decode(ids, skip_special_tokens=False)


def make_bpe_corpus(text: TextSplits, vocab_size: int = 4096) -> BPECorpus:
    """Fit only training text; callers own normalization and split selection."""
    if not MIN_VOCAB <= vocab_size <= MAX_VOCAB:
        raise CorpusError(reason="BPE vocabulary budget must be between 257 and 65536")
    serialized = fit_tokenizer(text.train, vocab_size)
    tokenizer = restore_tokenizer(serialized)
    vocabulary = tuple(
        token
        for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    )
    train, valid, test = (
        np.asarray(
            tokenizer.encode(split, add_special_tokens=False).ids, dtype=np.int64
        )
        for split in (text.train, text.valid, text.test)
    )
    fingerprint = _fingerprint(serialized, (train, valid, test), text.provenance)
    return BPECorpus(
        vocabulary, train, valid, test, serialized, fingerprint, text.provenance
    )


def save_bpe_corpus(corpus: BPECorpus, path: Path) -> None:
    """Write reproducible ZIP metadata and plain NumPy arrays without pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = (
        ("format", np.asarray(FORMAT)),
        ("tokenizer_json", np.asarray(corpus.tokenizer_json)),
        ("vocabulary", np.asarray(corpus.vocabulary)),
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


def _scalar(data: "NpzFile[generic]", name: str) -> str:
    array = data[name]
    if array.shape != () or array.dtype.kind != "U":
        raise CorpusError(reason=f"BPE field must be a scalar string: {name}")
    return TypeAdapter(str).validate_python(array.item(), strict=True)


def load_bpe_corpus(path: Path) -> BPECorpus:
    """Reject incompatible, inconsistent, or fingerprint-mismatched artifacts."""
    with path.open("rb") as stream:
        archive: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with archive as data:
            return _load_arrays(data)


def _load_arrays(data: "NpzFile[generic]") -> BPECorpus:
    expected = {
        "format",
        "tokenizer_json",
        "vocabulary",
        "train",
        "valid",
        "test",
        "fingerprint",
        "provenance",
    }
    if set(data.files) != expected or len(data.files) != len(expected):
        raise CorpusError(reason="Missing, duplicate, or unexpected BPE corpus fields")
    if _scalar(data, "format") != FORMAT:
        raise CorpusError(reason="Unsupported BPE corpus format")
    vocabulary = data["vocabulary"]
    if vocabulary.ndim != 1 or vocabulary.dtype.kind != "U":
        raise CorpusError(reason="BPE vocabulary must be a string vector")
    if any(data[name].dtype != np.int64 for name in ("train", "valid", "test")):
        raise CorpusError(reason="BPE token arrays must be int64")
    return BPECorpus(
        TypeAdapter(tuple[str, ...]).validate_python(vocabulary.tolist()),
        np.asarray(data["train"], dtype=np.int64),
        np.asarray(data["valid"], dtype=np.int64),
        np.asarray(data["test"], dtype=np.int64),
        _scalar(data, "tokenizer_json"),
        _scalar(data, "fingerprint"),
        _scalar(data, "provenance"),
    )
