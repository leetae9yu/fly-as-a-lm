"""Local official-split quality corpus with training-only byte BPE and provenance."""

import hashlib
from importlib.metadata import version
from itertools import accumulate
from pathlib import Path
from typing import Annotated, Final

import numpy as np
from pydantic import Field

from flyrl.bpe_data import BPECorpus, bpe_fingerprint
from flyrl.bpe_tokenizer import BPETokenizer, fit_tokenizer, restore_tokenizer
from flyrl.language_data import CorpusError, IntVector
from flyrl.language_models import Settings
from flyrl.story_data import StoryCorpus, StoryMetadata, StorySplit
from flyrl.story_source import ConsumedSource, SelectedStories

VOCAB_SIZE: Final = 4096


class QualityPreparation(Settings):
    """Separate local official sources and positive, source-ordered split budgets."""

    raw_train: Path
    raw_valid: Path
    source: Annotated[str, Field(min_length=1)]
    revision: Annotated[str, Field(min_length=1)]
    license_text: Annotated[str, Field(min_length=1)]
    train_stories: Annotated[int, Field(ge=1)] = 50_000
    valid_stories: Annotated[int, Field(ge=1)] = 256
    test_stories: Annotated[int, Field(ge=1)] = 512
    max_source_bytes: Annotated[int, Field(ge=1)] = 256 * 1024 * 1024


class QualityReport(StoryMetadata):
    """Human-readable sidecar retaining embedded metadata and actual artifact counts."""

    raw_train: str
    raw_valid: str
    actual_vocab_size: int
    token_counts: dict[str, int]
    tokenizer_sha256: str
    fingerprint: str


def _select(path: Path, wanted: int, seen: set[str], max_bytes: int) -> SelectedStories:
    """Consume a bounded prefix, deduplicating against all previously selected text."""
    selected: list[str] = []
    digest = hashlib.sha256()
    read_bytes, duplicates = 0, 0
    pieces: list[bytes] = []
    with path.open("rb") as stream:
        while len(selected) < wanted:
            line = stream.readline(max_bytes - read_bytes + 1)
            read_bytes += len(line)
            if read_bytes > max_bytes:
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
                identity = hashlib.sha256(story.encode("utf-8")).hexdigest()
                if identity in seen:
                    duplicates += 1
                else:
                    seen.add(identity)
                    selected.append(story)
            if not line:
                break
    if len(selected) != wanted:
        raise CorpusError(
            reason=f"Need {wanted} unique stories from {path}; found {len(selected)}"
        )
    return SelectedStories(
        stories=tuple(selected),
        consumed=(
            ConsumedSource(
                path=str(path),
                bytes_consumed=read_bytes,
                prefix_sha256=digest.hexdigest(),
            ),
        ),
        duplicates_removed=duplicates,
    )


def _encode(
    tokenizer: BPETokenizer, stories: tuple[str, ...]
) -> tuple[IntVector, StorySplit]:
    """Encode independently so no merge or target spans two source stories."""
    encoded = [
        tokenizer.encode(story, add_special_tokens=False).ids for story in stories
    ]
    tokens = np.fromiter(
        (token for story in encoded for token in story), dtype=np.int64
    )
    split = StorySplit(
        offsets=tuple(accumulate((len(story) for story in encoded), initial=0)),
        sha256=tuple(
            hashlib.sha256(story.encode("utf-8")).hexdigest() for story in stories
        ),
    )
    return tokens, split


def prepare_quality_corpus(options: QualityPreparation) -> StoryCorpus:
    """Select train first, then unique nontraining official-validation stories.

    The validation file never supplies training text, even when train is exhausted.
    An identity present in both sources is kept only in training. No shuffle,
    Unicode normalization, whitespace collapse, or network operation is performed.
    The shared byte budget covers the consumed prefixes of both files, in order.
    """
    if options.raw_train.samefile(options.raw_valid):
        raise CorpusError(
            reason="Official train and validation need separate raw files"
        )
    seen: set[str] = set()
    training = _select(
        options.raw_train, options.train_stories, seen, options.max_source_bytes
    )
    heldout = _select(
        options.raw_valid,
        options.valid_stories + options.test_stories,
        seen,
        options.max_source_bytes - training.consumed[0].bytes_consumed,
    )
    serialized = fit_tokenizer("\n".join(training.stories), VOCAB_SIZE)
    tokenizer = restore_tokenizer(serialized)
    vocabulary = tuple(
        token
        for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    )
    train, train_split = _encode(tokenizer, training.stories)
    valid, valid_split = _encode(tokenizer, heldout.stories[: options.valid_stories])
    test, test_split = _encode(tokenizer, heldout.stories[options.valid_stories :])
    metadata = StoryMetadata(
        source=options.source,
        revision=options.revision,
        license_text=options.license_text,
        consumed=(*training.consumed, *heldout.consumed),
        duplicates_removed=training.duplicates_removed + heldout.duplicates_removed,
        seed=0,
        requested_vocab_size=VOCAB_SIZE,
        tokenizers_version=version("tokenizers"),
        selection=(
            "consumed[0]: official training; consumed[1]: official validation; "
            "strip outer whitespace only; exact UTF-8 SHA256 deduplication; "
            "first requested unique official-training stories in source order; "
            "exclude selected training identities from official validation; "
            "first remaining unique official-validation stories in source order, "
            "validation then test; no permutation"
        ),
        train=train_split,
        valid=valid_split,
        test=test_split,
    )
    provenance = metadata.model_dump_json()
    fingerprint = bpe_fingerprint(serialized, (train, valid, test), provenance)
    return StoryCorpus(
        BPECorpus(vocabulary, train, valid, test, serialized, fingerprint, provenance),
        metadata,
    )


def quality_report(stories: StoryCorpus) -> QualityReport:
    """Summarize actual arrays without a self-referential embedded fingerprint."""
    corpus = stories.corpus
    return QualityReport.model_validate(
        {
            **stories.metadata.model_dump(),
            "raw_train": stories.metadata.consumed[0].path,
            "raw_valid": stories.metadata.consumed[1].path,
            "actual_vocab_size": len(corpus.vocabulary),
            "token_counts": {
                "train": corpus.train.size,
                "valid": corpus.valid.size,
                "test": corpus.test.size,
            },
            "tokenizer_sha256": hashlib.sha256(
                corpus.tokenizer_json.encode("utf-8")
            ).hexdigest(),
            "fingerprint": corpus.fingerprint,
        }
    )
