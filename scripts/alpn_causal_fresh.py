"""Construct the strict ALPN causal fresh-evaluation token artifact."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

import numpy as np

from flyrl.story_data import StoryCorpus, load_story_corpus
from scripts.alpn_causal_fresh_archive import (
    load_source_manifest,
)
from scripts.alpn_causal_fresh_archive import (
    validate_saved_artifact as _validate_saved_artifact,
)
from scripts.alpn_causal_fresh_types import (
    BASE_CORPUS_FINGERPRINT,
    BASE_CORPUS_PATH,
    BASE_CORPUS_SHA256,
    BASE_STORIES,
    ENCODED_STREAM_SHA256,
    EXPOSURE_THRESHOLD,
    FRESH_STORIES,
    FRESH_TARGETS,
    FRESH_TOKENS,
    MIN_STORY_TOKENS,
    NGRAM_WORDS,
    OFFSET_ARRAY_SHA256,
    SOURCE_BYTES,
    SOURCE_PATH,
    SOURCE_REVISION,
    SOURCE_SHA256,
    SOURCE_STORIES,
    SOURCE_URL,
    STORY_IDENTITY_SHA256,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
    FreshArtifact,
    FreshCorpusError,
    Int64Array,
)

SOURCE_MANIFEST_PATH: Final = Path("artifacts/tinystories-t4-source/source.json")
DELIMITER: Final = b"<|endoftext|>\n"
BOUNDARY_ERROR: Final = "Noncanonical delimiter boundary"
DELIMITER_ERROR: Final = "Source ended without a complete delimiter boundary"


def parse_source_stories(payload: bytes) -> tuple[str, ...]:
    """Parse only exact newline-terminated delimiter boundaries from source bytes."""
    if not payload:
        raise FreshCorpusError
    stories: list[str] = []
    lines: list[bytes] = []
    for line in payload.splitlines(keepends=True):
        if line == DELIMITER:
            stories.append(_complete_story(lines))
            lines.clear()
        elif line.rstrip(b"\r\n").strip() == DELIMITER.rstrip():
            raise FreshCorpusError(BOUNDARY_ERROR)
        else:
            lines.append(line)
    if lines:
        raise FreshCorpusError(DELIMITER_ERROR)
    return tuple(stories)


def five_word_set(text: str) -> set[tuple[str, ...]]:
    """Return case-folded contiguous five-word shingles after whitespace collapse."""
    words = " ".join(text.casefold().split()).split(" ")
    return {
        tuple(words[index : index + NGRAM_WORDS]) for index in range(len(words) - 4)
    }


def build_fresh_artifact(root: Path) -> FreshArtifact:
    """Reconstruct and validate the document-frozen fresh selection without fitting."""
    source_stories, source_identities = _load_source_stories(root)
    base, base_identities, base_texts = _load_base_corpus(root)
    fresh_stories, fresh_identities = _select_fresh_stories(
        source_stories, source_identities, base_texts, base_identities
    )
    artifact = FreshArtifact(
        tokens=_encode_stories(base, fresh_stories),
        offsets=_story_offsets(base, fresh_stories),
        story_identities=fresh_identities,
        base_story_identities=base_identities,
        maximum_exposure_jaccard=_validate_exposure(
            fresh_stories, tuple(base_texts.values())
        ),
    )
    _validate_document_hashes(artifact)
    return artifact


def validate_saved_artifact(root: Path) -> FreshArtifact:
    """Rebuild frozen inputs and require the saved artifact to match exactly."""
    return _validate_saved_artifact(root, build_fresh_artifact(root))


def _complete_story(lines: list[bytes]) -> str:
    if not lines:
        raise FreshCorpusError(BOUNDARY_ERROR)
    try:
        story = b"".join(lines).decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise FreshCorpusError from error
    if not story:
        raise FreshCorpusError(BOUNDARY_ERROR)
    return story


def _load_source_stories(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    _validate_source_manifest(root)
    payload = (root / SOURCE_PATH).read_bytes()
    if len(payload) != SOURCE_BYTES or _sha256(payload) != SOURCE_SHA256:
        raise FreshCorpusError
    stories = parse_source_stories(payload)
    identities = tuple(_text_sha256(story) for story in stories)
    if len(stories) != SOURCE_STORIES or len(set(identities)) != SOURCE_STORIES:
        raise FreshCorpusError
    return stories, identities


def _load_base_corpus(
    root: Path,
) -> tuple[StoryCorpus, tuple[str, ...], dict[str, str]]:
    path = root / BASE_CORPUS_PATH
    if _file_sha256(path) != BASE_CORPUS_SHA256:
        raise FreshCorpusError
    base = load_story_corpus(path)
    tokenizer_json = (root / TOKENIZER_PATH).read_text(encoding="utf-8")
    if (
        base.corpus.fingerprint != BASE_CORPUS_FINGERPRINT
        or _sha256(tokenizer_json.encode("utf-8")) != TOKENIZER_SHA256
        or base.corpus.tokenizer_json != tokenizer_json
    ):
        raise FreshCorpusError
    identities = tuple(
        identity for split in base.metadata.splits for identity in split.sha256
    )
    if len(identities) != BASE_STORIES or len(set(identities)) != BASE_STORIES:
        raise FreshCorpusError
    return base, identities, _base_story_texts(base)


def _select_fresh_stories(
    source_stories: tuple[str, ...],
    source_identities: tuple[str, ...],
    base_texts: dict[str, str],
    base_identities: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    stories = tuple(
        story
        for story, identity in zip(source_stories, source_identities, strict=True)
        if identity not in base_texts
    )
    identities = tuple(_text_sha256(story) for story in stories)
    if len(stories) != FRESH_STORIES or set(identities) & set(base_identities):
        raise FreshCorpusError
    return stories, identities


def _encode_stories(base: StoryCorpus, stories: tuple[str, ...]) -> Int64Array:
    encoded = tuple(
        np.asarray(base.corpus.encode(story), dtype=np.int64) for story in stories
    )
    if any(values.size < MIN_STORY_TOKENS for values in encoded):
        raise FreshCorpusError
    return np.concatenate(encoded, dtype=np.int64)


def _story_offsets(base: StoryCorpus, stories: tuple[str, ...]) -> Int64Array:
    lengths = tuple(len(base.corpus.encode(story)) for story in stories)
    return np.asarray(np.cumsum((0, *lengths), dtype=np.int64), dtype=np.int64)


def _validate_source_manifest(root: Path) -> None:
    manifest = load_source_manifest(root / SOURCE_MANIFEST_PATH)
    if (
        manifest.source != SOURCE_URL
        or manifest.revision != SOURCE_REVISION
        or manifest.filename != SOURCE_PATH.name
        or manifest.stories != SOURCE_STORIES
        or manifest.prefix_bytes != SOURCE_BYTES
        or manifest.prefix_sha256 != SOURCE_SHA256
    ):
        raise FreshCorpusError


def _base_story_texts(base: StoryCorpus) -> dict[str, str]:
    """Decode base stories after StoryCorpus has verified their stored identities."""
    texts: dict[str, str] = {}
    for stream, split in zip(base.streams, base.metadata.splits, strict=True):
        for start, end, identity in zip(
            split.offsets[:-1], split.offsets[1:], split.sha256, strict=True
        ):
            text = base.corpus.decode(
                [int(stream.item(index)) for index in range(start, end)]
            )
            if texts.setdefault(identity, text) != text:
                raise FreshCorpusError
    return texts


def _validate_exposure(
    fresh_stories: tuple[str, ...], base_stories: tuple[str, ...]
) -> float:
    maximum = 0.0
    base_sets = tuple(five_word_set(story) for story in base_stories)
    for story in fresh_stories:
        words = five_word_set(story)
        for previous in base_sets:
            union = words | previous
            similarity = 1.0 if not union else len(words & previous) / len(union)
            if similarity >= EXPOSURE_THRESHOLD:
                raise FreshCorpusError
            maximum = max(maximum, similarity)
    return maximum


def _validate_document_hashes(artifact: FreshArtifact) -> None:
    checks = (
        artifact.story_count == FRESH_STORIES,
        artifact.tokens.size == FRESH_TOKENS,
        artifact.target_count == FRESH_TARGETS,
        artifact.token_sha256 == ENCODED_STREAM_SHA256,
        artifact.offset_sha256 == OFFSET_ARRAY_SHA256,
        artifact.story_identity_sha256 == STORY_IDENTITY_SHA256,
    )
    if not all(checks):
        raise FreshCorpusError


def _text_sha256(text: str) -> str:
    return _sha256(text.encode("utf-8"))


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
