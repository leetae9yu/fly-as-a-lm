import hashlib
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.lib.npyio import NpzFile
from pydantic import ValidationError

from scripts.alpn_causal_fresh import (
    parse_source_stories,
)
from scripts.alpn_causal_fresh_archive import load_saved_artifact
from scripts.alpn_causal_fresh_types import (
    ENCODED_STREAM_SHA256,
    OFFSET_ARRAY_SHA256,
    STORY_IDENTITY_SHA256,
    FreshCorpusError,
    FreshMetadata,
)

if TYPE_CHECKING:
    from numpy import generic

ROOT = Path(__file__).resolve().parents[1]


def test_loads_the_frozen_fresh_story_partition() -> None:
    artifact = load_saved_artifact(ROOT)

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
    typed_metadata = load_saved_artifact(ROOT).metadata(ROOT)
    assert isinstance(typed_metadata, FreshMetadata)
    metadata = typed_metadata.model_dump()
    metadata["exposure"]["maximum_jaccard"] = float("nan")
    metadata["unexpected"] = 0

    with pytest.raises(ValidationError):
        _ = FreshMetadata.model_validate(metadata)


def test_saved_artifact_revalidates_from_tracked_inputs() -> None:
    artifact = load_saved_artifact(ROOT)

    assert artifact.offsets.size == artifact.story_count + 1


def test_saved_artifact_rejects_fractional_array_dtypes(tmp_path: Path) -> None:
    data = tmp_path / "data/central_connectome"
    corpus = tmp_path / "artifacts/tinystories-t4-corpus"
    data.mkdir(parents=True)
    corpus.mkdir(parents=True)
    _ = copyfile(
        ROOT / "data/central_connectome/alpn_causal_fresh.json",
        data / "alpn_causal_fresh.json",
    )
    for name in ("corpus.npz", "tokenizer.json"):
        _ = copyfile(ROOT / "artifacts/tinystories-t4-corpus" / name, corpus / name)
    source = ROOT / "data/central_connectome/alpn_causal_fresh.npz"
    with source.open("rb") as stream:
        original: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with original:
            np.savez_compressed(
                data / "alpn_causal_fresh.npz",
                format=np.asarray(original["format"], dtype=np.str_),
                tokens=np.asarray(original["tokens"], dtype=np.float64) + 0.25,
                offsets=np.asarray(original["offsets"], dtype=np.float64) + 0.25,
            )

    with pytest.raises(FreshCorpusError):
        _ = load_saved_artifact(tmp_path)


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
