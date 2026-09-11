"""Standard BPE artifacts with fingerprinted story boundaries and source provenance."""

import hashlib
from dataclasses import dataclass
from importlib.metadata import version
from itertools import accumulate
from pathlib import Path
from typing import Final, Literal

import numpy as np

from flyrl.bpe_data import BPECorpus, bpe_fingerprint, load_bpe_corpus
from flyrl.bpe_tokenizer import fit_tokenizer, restore_tokenizer
from flyrl.language_data import CorpusError, IntVector
from flyrl.language_models import Settings
from flyrl.story_source import ConsumedSource, Preparation, select_stories

MIN_STORY_TOKENS: Final = 2


class StorySplit(Settings):
    """Exclusive token offsets and normalized UTF-8 SHA256 per selected story."""

    offsets: tuple[int, ...]
    sha256: tuple[str, ...]


class StoryMetadata(Settings):
    """Embedded provenance participates in the existing BPE content fingerprint."""

    format: Literal["flyrl-stories-v1"] = "flyrl-stories-v1"
    source: str
    revision: str
    license_text: str
    consumed: tuple[ConsumedSource, ...]
    duplicates_removed: int
    seed: int
    requested_vocab_size: int
    tokenizers_version: str
    selection: str = (
        "pool local files in supplied order; first requested unique complete stories; "
        "strip outer whitespace; SHA256 exact deduplication before seeded permutation; "
        "split permutation into train/valid/test counts; not official source splits"
    )
    vocabulary_fit: str = (
        "train stories only, newline joined; encode each story separately"
    )
    train: StorySplit
    valid: StorySplit
    test: StorySplit

    @property
    def splits(self) -> tuple[StorySplit, StorySplit, StorySplit]:
        """Return the canonical train/valid/test order."""
        return self.train, self.valid, self.test


@dataclass(frozen=True, slots=True)
class StoryCorpus:
    """Story-safe view over an unchanged, immutable BPECorpus artifact."""

    corpus: BPECorpus
    metadata: StoryMetadata

    def __post_init__(self) -> None:
        """Validate boundary coverage, selected hashes and exact duplicate isolation."""
        if self.metadata != StoryMetadata.model_validate_json(self.corpus.provenance):
            raise CorpusError(
                reason="Story metadata differs from fingerprinted provenance"
            )
        seen: set[str] = set()
        for tokens, split in zip(self.streams, self.metadata.splits, strict=True):
            if (
                len(split.offsets) != len(split.sha256) + 1
                or not split.sha256
                or split.offsets[0] != 0
                or split.offsets[-1] != tokens.size
                or any(
                    b - a < MIN_STORY_TOKENS
                    for a, b in zip(
                        split.offsets[:-1],
                        split.offsets[1:],
                        strict=True,
                    )
                )
            ):
                raise CorpusError(
                    reason="Story boundaries need complete coverage and token pairs"
                )
            for a, b, identity in zip(
                split.offsets[:-1],
                split.offsets[1:],
                split.sha256,
                strict=True,
            ):
                text = self.corpus.decode([int(tokens.item(i)) for i in range(a, b)])
                if hashlib.sha256(text.encode()).hexdigest() != identity:
                    raise CorpusError(reason="Selected story hash mismatch")
                if identity in seen:
                    raise CorpusError(reason="Exact duplicate story leakage")
                seen.add(identity)

    @property
    def streams(self) -> tuple[IntVector, IntVector, IntVector]:
        """Return token streams in canonical split order."""
        return self.corpus.train, self.corpus.valid, self.corpus.test

    def starts(self, split: StorySplit, context: int) -> IntVector:
        """Enumerate uniform legal windows without crossing a story boundary.

        Stories shorter than context+target contribute no training windows.
        They still contribute all next-token pairs to heldout evaluation.
        """
        if context < 1:
            raise CorpusError(reason="Story context must be positive")
        starts = np.fromiter(
            (
                start
                for a, b in zip(split.offsets[:-1], split.offsets[1:], strict=True)
                for start in range(a, b - context)
            ),
            dtype=np.int64,
        )
        if starts.size == 0:
            raise CorpusError(
                reason="No story has enough tokens for context plus target"
            )
        return starts


def prepare_stories(options: Preparation) -> StoryCorpus:
    """Deduplicate and split complete stories before fitting training-only BPE."""
    selected = select_stories(options)
    order = np.arange(len(selected.stories), dtype=np.int64)
    np.random.default_rng(options.seed).shuffle(order)
    shuffled = [selected.stories[int(i)] for i in order]
    boundary = options.train_stories + options.valid_stories
    texts = (
        shuffled[: options.train_stories],
        shuffled[options.train_stories : boundary],
        shuffled[boundary:],
    )
    serialized = fit_tokenizer("\n".join(texts[0]), options.vocab_size)
    tokenizer = restore_tokenizer(serialized)
    vocabulary = tuple(
        token
        for token, _ in sorted(
            tokenizer.get_vocab().items(),
            key=lambda item: item[1],
        )
    )
    streams: list[IntVector] = []
    splits: list[StorySplit] = []
    for stories in texts:
        encoded = [
            np.asarray(
                tokenizer.encode(story, add_special_tokens=False).ids, dtype=np.int64
            )
            for story in stories
        ]
        offsets = tuple(accumulate((tokens.size for tokens in encoded), initial=0))
        streams.append(
            np.fromiter(
                (int(token) for tokens in encoded for token in tokens), dtype=np.int64
            )
        )
        splits.append(
            StorySplit(
                offsets=offsets,
                sha256=tuple(
                    hashlib.sha256(story.encode()).hexdigest() for story in stories
                ),
            )
        )
    metadata = StoryMetadata(
        source=options.source,
        revision=options.revision,
        license_text=options.license_text,
        consumed=selected.consumed,
        duplicates_removed=selected.duplicates_removed,
        seed=options.seed,
        requested_vocab_size=options.vocab_size,
        tokenizers_version=version("tokenizers"),
        train=splits[0],
        valid=splits[1],
        test=splits[2],
    )
    provenance = metadata.model_dump_json()
    train, valid, test = streams
    fingerprint = bpe_fingerprint(serialized, (train, valid, test), provenance)
    return StoryCorpus(
        BPECorpus(
            vocabulary,
            train,
            valid,
            test,
            serialized,
            fingerprint,
            provenance,
        ),
        metadata,
    )


def load_story_corpus(path: Path) -> StoryCorpus:
    """Load through BPECorpus and reject absent or inconsistent story metadata."""
    corpus = load_bpe_corpus(path)
    return StoryCorpus(corpus, StoryMetadata.model_validate_json(corpus.provenance))
