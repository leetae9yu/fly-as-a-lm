"""Deterministic JSON and NPZ I/O for ALPN causal fresh artifacts."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Final
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from pydantic import Field

from scripts.alpn_causal_fresh_types import (
    FORMAT,
    FreshArtifact,
    FreshCorpusError,
    FreshMetadata,
    FrozenRecord,
    Int64Array,
)

if TYPE_CHECKING:
    from numpy import generic
    from numpy.typing import NDArray

DIRECTORY: Final = Path("data/central_connectome")
JSON_NAME: Final = "alpn_causal_fresh.json"
NPZ_NAME: Final = "alpn_causal_fresh.npz"
SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"


class SourceManifest(FrozenRecord):
    """Strict local source manifest required before source parsing."""

    source: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    stories: int = Field(gt=0)
    bytes_downloaded: int = Field(gt=0)
    prefix_bytes: int = Field(gt=0)
    prefix_sha256: str = Field(pattern=SHA256_PATTERN)
    card_sha256: str = Field(pattern=SHA256_PATTERN)
    selection: str = Field(min_length=1)


def load_source_manifest(path: Path) -> SourceManifest:
    """Parse the saved source identity through its strict Pydantic boundary."""
    try:
        return SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FreshCorpusError from error


def save_fresh_artifact(artifact: FreshArtifact, root: Path) -> None:
    """Write a deterministic typed JSON document and pickle-free int64 arrays."""
    directory = root / DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    _ = (directory / JSON_NAME).write_text(
        artifact.metadata(root).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    arrays = (
        ("format", np.asarray(FORMAT)),
        ("tokens", artifact.tokens),
        ("offsets", artifact.offsets),
    )
    with ZipFile(directory / NPZ_NAME, "w", compression=ZIP_DEFLATED) as archive:
        for name, array in arrays:
            buffer = BytesIO()
            write_array(buffer, array, allow_pickle=False)
            member = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            archive.writestr(member, buffer.getvalue())


def load_saved_artifact(root: Path) -> FreshArtifact:
    """Authenticate the tracked JSON/NPZ pair without requiring ignored raw inputs."""
    directory = root / DIRECTORY
    try:
        metadata = FreshMetadata.model_validate_json(
            (directory / JSON_NAME).read_text(encoding="utf-8")
        )
        tokens, offsets, format_value, fields = _read_arrays(directory / NPZ_NAME)
    except (OSError, ValueError) as error:
        raise FreshCorpusError from error
    artifact = FreshArtifact(
        tokens=tokens,
        offsets=offsets,
        story_identities=metadata.fresh_stories.identities,
        base_story_identities=metadata.base_corpus.story_identities,
        maximum_exposure_jaccard=metadata.exposure.maximum_jaccard,
    )
    if (
        metadata != artifact.metadata(root)
        or fields != {"format", "tokens", "offsets"}
        or format_value.shape != ()
        or format_value.item() != FORMAT
        or tokens.dtype != np.dtype("int64")
        or offsets.dtype != np.dtype("int64")
    ):
        raise FreshCorpusError
    return artifact


def validate_saved_artifact(root: Path, expected: FreshArtifact) -> FreshArtifact:
    """Reject any saved value that differs from a fresh frozen reconstruction."""
    actual = load_saved_artifact(root)
    if (
        actual.story_identities != expected.story_identities
        or actual.base_story_identities != expected.base_story_identities
        or actual.maximum_exposure_jaccard != expected.maximum_exposure_jaccard
        or not np.array_equal(actual.tokens, expected.tokens)
        or not np.array_equal(actual.offsets, expected.offsets)
    ):
        raise FreshCorpusError
    return expected


def _read_arrays(
    path: Path,
) -> tuple[Int64Array, Int64Array, NDArray[np.str_], set[str]]:
    with path.open("rb") as stream:
        archive: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with archive as data:
            return (
                np.asarray(data["tokens"], dtype=np.int64),
                np.asarray(data["offsets"], dtype=np.int64),
                np.asarray(data["format"], dtype=np.str_),
                set(data.files),
            )
