import hashlib
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from scripts.alpn_causal_fresh import (
    build_fresh_artifact,
    parse_source_stories,
    validate_saved_artifact,
)
from scripts.alpn_causal_fresh_types import (
    ENCODED_STREAM_SHA256,
    OFFSET_ARRAY_SHA256,
    STORY_IDENTITY_SHA256,
    FreshCorpusError,
    FreshMetadata,
)

ROOT = Path(__file__).resolve().parents[1]


def test_builds_the_frozen_fresh_story_partition() -> None:
    artifact = build_fresh_artifact(ROOT)

    assert artifact.tokens.dtype == np.dtype("int64")
    assert artifact.offsets.dtype == np.dtype("int64")
    assert artifact.story_count == 68
    assert artifact.target_count == 12_242
    assert artifact.tokens.size == 12_310
    assert artifact.token_sha256 == ENCODED_STREAM_SHA256
    assert artifact.offset_sha256 == OFFSET_ARRAY_SHA256
    assert artifact.story_identity_sha256 == STORY_IDENTITY_SHA256
    assert artifact.maximum_exposure_jaccard < 0.8
    assert set(artifact.story_identities).isdisjoint(artifact.base_story_identities)


def test_metadata_rejects_nonfinite_or_unrecognized_boundary_values() -> None:
    typed_metadata = build_fresh_artifact(ROOT).metadata(ROOT)
    assert isinstance(typed_metadata, FreshMetadata)
    metadata = typed_metadata.model_dump()
    metadata["exposure"]["maximum_jaccard"] = float("nan")
    metadata["unexpected"] = 0

    with pytest.raises(ValidationError):
        _ = FreshMetadata.model_validate(metadata)


def test_generated_artifact_revalidates_against_frozen_inputs() -> None:
    artifact = validate_saved_artifact(ROOT)

    assert artifact.offsets.size == artifact.story_count + 1


def test_rejects_noncanonical_or_incomplete_boundaries() -> None:
    with pytest.raises(FreshCorpusError, match="boundary"):
        _ = parse_source_stories(b"One.\n<|endoftext|> \n")
    with pytest.raises(FreshCorpusError, match="delimiter"):
        _ = parse_source_stories(b"One.\n")


def test_source_identities_are_hashes_of_normalized_complete_stories() -> None:
    stories = parse_source_stories(b"  One story.  \n<|endoftext|>\n")

    assert stories == ("One story.",)
    assert hashlib.sha256(stories[0].encode("utf-8")).hexdigest() == (
        "0a6e46740a00c6f9a3659ca9539c66661f7c73fbd5ea1f9aa24619a36889e3be"
    )
