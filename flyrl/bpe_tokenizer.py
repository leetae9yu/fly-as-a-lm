"""Typed serialization boundary for tokenizers' unannotated Rust bindings."""

import json
from collections.abc import Iterator
from typing import Final, Protocol, runtime_checkable

import tokenizers
from pydantic import ConfigDict, JsonValue, TypeAdapter
from tokenizers import pre_tokenizers

UNKNOWN: Final = "<unk>"


class _Encoding(Protocol):
    @property
    def ids(self) -> list[int]: ...


@runtime_checkable
class BPETokenizer(Protocol):
    """The immutable-use subset of the native tokenizer interface."""

    def to_str(self, pretty: bool = False) -> str:
        """Serialize the tokenizer configuration, vocabulary, and merge order."""
        ...

    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> _Encoding:
        """Encode text with the native byte-level model."""
        ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode byte-level IDs to Unicode text."""
        ...

    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]:
        """Return raw model labels and their integer IDs."""
        ...


class _Trainable(BPETokenizer, Protocol):
    def train_from_iterator(
        self,
        iterator: Iterator[str],
        *,
        vocab_size: int,
        min_frequency: int,
        show_progress: bool,
        special_tokens: list[str],
    ) -> None: ...


@runtime_checkable
class _Factory(Protocol):
    def __call__(self, *, add_prefix_space: bool) -> _Trainable: ...


@runtime_checkable
class _Loader(Protocol):
    def from_str(self, json: str) -> BPETokenizer: ...


@runtime_checkable
class _Alphabet(Protocol):
    def alphabet(self) -> list[str]: ...


def initial_alphabet() -> tuple[str, ...]:
    """Return the native byte-to-visible-character mapping in stable ID order."""
    factory = TypeAdapter(
        _Alphabet, config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(vars(pre_tokenizers)["ByteLevel"])
    return tuple(sorted(factory.alphabet()))


_LOADER = TypeAdapter(
    _Loader, config=ConfigDict(arbitrary_types_allowed=True)
).validate_python(vars(tokenizers)["Tokenizer"])


def restore_tokenizer(serialized: str) -> BPETokenizer:
    """Restore a native tokenizer without exposing its incomplete stub types."""
    return _LOADER.from_str(serialized)


def fit_tokenizer(train: str, vocab_size: int) -> str:
    """Fit all 256 initial bytes and deterministic BPE merges on training alone."""
    factory = TypeAdapter(
        _Factory, config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(vars(tokenizers)["ByteLevelBPETokenizer"])
    tokenizer = factory(add_prefix_space=False)
    tokenizer.train_from_iterator(
        iter((train,)),
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=False,
        special_tokens=[UNKNOWN],
    )
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(tokenizer.to_str())
    model = TypeAdapter(dict[str, JsonValue]).validate_python(payload["model"])
    # The byte-level wrapper reserves special tokens but leaves BPE.unk_token null.
    model["unk_token"] = UNKNOWN
    payload["model"] = model
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
