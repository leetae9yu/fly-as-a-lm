"""Typed immutable records for the ALPN causal fresh-evaluation artifact."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

Int64Array = NDArray[np.int64]
SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
FORMAT: Final = "flyrl-alpn-causal-fresh-v1"
SOURCE_PATH: Final = Path("artifacts/tinystories-t4-source/TinyStories-train.txt")
BASE_CORPUS_PATH: Final = Path("artifacts/tinystories-t4-corpus/corpus.npz")
TOKENIZER_PATH: Final = Path("artifacts/tinystories-t4-corpus/tokenizer.json")
SOURCE_URL: Final = "https://huggingface.co/datasets/roneneldan/TinyStories"
SOURCE_REVISION: Final = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
SOURCE_STORIES: Final = 1_100
SOURCE_BYTES: Final = 1_027_216
SOURCE_SHA256: Final = (
    "b888ac8b6858ea8ce4547b19da58740bb364df28b7f28778b73f9121a9137fa6"
)
BASE_CORPUS_SHA256: Final = (
    "c750d2aabe2311cb713d3afed98c6fa8b1f044f092a7727a3a8cef278ec22b5e"
)
BASE_CORPUS_FINGERPRINT: Final = (
    "fd9d30ce06ef753feb2160370d40bceebd41f126aaf6be58d9d6ed42cda45575"
)
TOKENIZER_SHA256: Final = (
    "9dbe72484ae01b374801f3f7f8ffdcba3f126cb71774dc5f36ef75d06d064e27"
)
BASE_STORIES: Final = 1_032
FRESH_STORIES: Final = 68
FRESH_TOKENS: Final = 12_310
FRESH_TARGETS: Final = 12_242
ENCODED_STREAM_SHA256: Final = (
    "62083864627b36147b6c16d3326f94fa6ccc63f5acc2c384e018b4c73891920f"
)
OFFSET_ARRAY_SHA256: Final = (
    "f8b75050de6867464d8464f32618ea99652f3621d4d4a1afd9618465e2391784"
)
STORY_IDENTITY_SHA256: Final = (
    "d87d6abe6ac9c5dfa66f3e34e73a5737225d75ad8f239519ebae68d4277bb96d"
)
NGRAM_WORDS: Final = 5
EXPOSURE_THRESHOLD: Final = 0.8
MIN_STORY_TOKENS: Final = 2
ERROR: Final = "Fresh corpus differs from frozen inputs"


class FreshCorpusError(ValueError):
    """Raised when a fresh-corpus input or artifact differs from the frozen plan."""

    def __init__(self, message: str = ERROR) -> None:
        """Initialize a frozen-input validation error."""
        super().__init__(message)


class FrozenRecord(BaseModel):
    """Reject untyped, extra, or non-finite values at the artifact boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False
    )


class SourceIdentity(FrozenRecord):
    """Pinned complete-source identity."""

    path: str = Field(min_length=1)
    source: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    bytes_consumed: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)
    unique_story_count: int = Field(gt=0)


class TokenizerIdentity(FrozenRecord):
    """Reused tokenizer identity."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    vocabulary_size: int = Field(gt=0)


class BaseCorpusIdentity(FrozenRecord):
    """Complete prior-corpus identity list and file identity."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    fingerprint: str = Field(pattern=SHA256_PATTERN)
    story_count: int = Field(gt=0)
    story_identities: tuple[str, ...] = Field(min_length=1)
    ordered_story_identity_sha256: str = Field(pattern=SHA256_PATTERN)


class FreshStories(FrozenRecord):
    """Ordered fresh story identities and frozen selection contract."""

    count: int = Field(gt=0)
    identities: tuple[str, ...] = Field(min_length=1)
    ordered_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    selection: str = Field(min_length=1)


class ArrayIdentity(FrozenRecord):
    """Portable numeric-array identity."""

    count: int = Field(gt=0)
    dtype: Literal["int64"]
    byte_order: Literal["little"]
    sha256: str = Field(pattern=SHA256_PATTERN)


class ExposureRule(FrozenRecord):
    """Frozen five-word exposure constraint and observed maximum."""

    normalization: Literal["casefold; collapse whitespace"]
    word_ngram_size: Literal[5]
    jaccard_reject_at_or_above: float = Field(ge=0, le=1)
    maximum_jaccard: float = Field(ge=0, le=1)


class FreshMetadata(FrozenRecord):
    """Strict complete JSON metadata schema for the fresh artifact."""

    format: Literal["flyrl-alpn-causal-fresh-v1"]
    source: SourceIdentity
    tokenizer: TokenizerIdentity
    base_corpus: BaseCorpusIdentity
    fresh_stories: FreshStories
    tokens: ArrayIdentity
    offsets: ArrayIdentity
    targets: int = Field(gt=0)
    exposure: ExposureRule


@dataclass(frozen=True, slots=True)
class FreshArtifact:
    """Fresh int64 arrays, identities, and a typed metadata serializer."""

    tokens: Int64Array
    offsets: Int64Array
    story_identities: tuple[str, ...]
    base_story_identities: tuple[str, ...]
    maximum_exposure_jaccard: float

    @property
    def story_count(self) -> int:
        """Return the number of complete fresh stories."""
        return len(self.story_identities)

    @property
    def target_count(self) -> int:
        """Return next-token targets without cross-story transitions."""
        return int(self.tokens.size - self.story_count)

    @property
    def token_sha256(self) -> str:
        """Return the canonical little-endian token-array identity."""
        return _array_sha256(self.tokens)

    @property
    def offset_sha256(self) -> str:
        """Return the canonical little-endian offset-array identity."""
        return _array_sha256(self.offsets)

    @property
    def story_identity_sha256(self) -> str:
        """Return the ordered fresh story-identity digest."""
        return _identities_sha256(self.story_identities)

    def metadata(self, root: Path) -> FreshMetadata:
        """Return strict metadata without serializing or retaining story text."""
        return FreshMetadata(
            format=FORMAT,
            source=SourceIdentity(
                path=str(SOURCE_PATH),
                source=SOURCE_URL,
                revision=SOURCE_REVISION,
                filename=SOURCE_PATH.name,
                bytes_consumed=SOURCE_BYTES,
                sha256=SOURCE_SHA256,
                unique_story_count=SOURCE_STORIES,
            ),
            tokenizer=TokenizerIdentity(
                path=str(TOKENIZER_PATH),
                sha256=_file_sha256(root / TOKENIZER_PATH),
                vocabulary_size=4096,
            ),
            base_corpus=BaseCorpusIdentity(
                path=str(BASE_CORPUS_PATH),
                sha256=_file_sha256(root / BASE_CORPUS_PATH),
                fingerprint=BASE_CORPUS_FINGERPRINT,
                story_count=len(self.base_story_identities),
                story_identities=self.base_story_identities,
                ordered_story_identity_sha256=_identities_sha256(
                    self.base_story_identities
                ),
            ),
            fresh_stories=FreshStories(
                count=self.story_count,
                identities=self.story_identities,
                ordered_identity_sha256=self.story_identity_sha256,
                selection=(
                    "first 1100 unique complete source stories; exclude every "
                    "base-corpus identity; retain remaining stories in source order"
                ),
            ),
            tokens=ArrayIdentity(
                count=int(self.tokens.size),
                dtype="int64",
                byte_order="little",
                sha256=self.token_sha256,
            ),
            offsets=ArrayIdentity(
                count=int(self.offsets.size),
                dtype="int64",
                byte_order="little",
                sha256=self.offset_sha256,
            ),
            targets=self.target_count,
            exposure=ExposureRule(
                normalization="casefold; collapse whitespace",
                word_ngram_size=5,
                jaccard_reject_at_or_above=0.8,
                maximum_jaccard=self.maximum_exposure_jaccard,
            ),
        )


def _array_sha256(values: Int64Array) -> str:
    return hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()


def _identities_sha256(identities: tuple[str, ...]) -> str:
    return hashlib.sha256("".join(identities).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
