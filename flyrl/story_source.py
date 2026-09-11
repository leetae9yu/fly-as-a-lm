"""Bounded, explicit local TinyStories source selection; no network operations."""

import hashlib
from pathlib import Path
from typing import Annotated

from pydantic import Field

from flyrl.language_data import CorpusError
from flyrl.language_models import Settings


class Preparation(Settings):
    """Local source identity and a fixed unique-story selection budget."""

    raw_files: Annotated[tuple[Path, ...], Field(min_length=1)]
    source: Annotated[str, Field(min_length=1)]
    revision: Annotated[str, Field(min_length=1)]
    license_text: Annotated[str, Field(min_length=1)]
    train_stories: Annotated[int, Field(ge=1)] = 1000
    valid_stories: Annotated[int, Field(ge=1)] = 16
    test_stories: Annotated[int, Field(ge=1)] = 16
    vocab_size: Annotated[int, Field(ge=257, le=65536)] = 4096
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 0
    max_source_bytes: Annotated[int, Field(ge=1)] = 64 * 1024 * 1024


class ConsumedSource(Settings):
    """Identity of the exact consumed prefix, not a claimed whole-file checksum."""

    path: str
    bytes_consumed: int
    prefix_sha256: str


class SelectedStories(Settings):
    """Normalized unique stories plus auditable source consumption."""

    stories: tuple[str, ...]
    consumed: tuple[ConsumedSource, ...]
    duplicates_removed: int


def select_stories(options: Preparation) -> SelectedStories:
    """Read complete delimiter-separated stories until the unique budget is met.

    Files are pooled in caller order, not interpreted as official dataset splits.
    Leading/trailing story whitespace is stripped; interior text is untouched.
    Deduplication is exact after this normalization, before any split assignment.
    """
    wanted = options.train_stories + options.valid_stories + options.test_stories
    selected: list[str] = []
    seen: set[str] = set()
    consumed: list[ConsumedSource] = []
    total_bytes, duplicates = 0, 0
    for path in options.raw_files:
        digest = hashlib.sha256()
        read_bytes = 0
        pieces: list[bytes] = []
        with path.open("rb") as stream:
            while len(selected) < wanted:
                line = stream.readline(options.max_source_bytes - total_bytes + 1)
                total_bytes += len(line)
                read_bytes += len(line)
                if total_bytes > options.max_source_bytes:
                    raise CorpusError(
                        reason="Source byte budget exhausted before selection"
                    )
                digest.update(line)
                if line and line.strip() != b"<|endoftext|>":
                    pieces.append(line)
                    continue
                story = b"".join(pieces).decode("utf-8").strip()
                pieces.clear()
                if story:
                    identity = hashlib.sha256(story.encode()).hexdigest()
                    if identity in seen:
                        duplicates += 1
                    else:
                        seen.add(identity)
                        selected.append(story)
                if not line:
                    break
        consumed.append(
            ConsumedSource(
                path=str(path),
                bytes_consumed=read_bytes,
                prefix_sha256=digest.hexdigest(),
            )
        )
        if len(selected) == wanted:
            return SelectedStories(
                stories=tuple(selected),
                consumed=tuple(consumed),
                duplicates_removed=duplicates,
            )
    raise CorpusError(reason=f"Need {wanted} unique stories; found {len(selected)}")
