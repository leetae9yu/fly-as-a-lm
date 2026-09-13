"""Pickle-free, hash-bound regional-probe head and heldout feature artifacts."""

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Final
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from pydantic import TypeAdapter

from flyrl.regional_probe import PROBE_WIDTH, SATURATION_THRESHOLD
from flyrl.regional_probe_types import (
    ExtractionConfig,
    FeatureCache,
    FeatureStats,
    ProbeParameters,
    ProbeResult,
)
from scripts.connectome_source import file_digest
from scripts.regional_probe_artifact_types import CompactProbeResult, HeadArtifact
from scripts.regional_probe_manifest import ProbeGroupRecord
from scripts.regional_probe_protocol import ProbeSourceCheckpoint

MATRIX_DIMENSIONS: Final = 2


def write_npz(
    path: Path, arrays: Mapping[str, NDArray[np.float32] | NDArray[np.int64]]
) -> None:
    """Write compressed numerical arrays without NumPy's permissive pickle default."""
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w") as member:
                write_array(member, array, allow_pickle=False)


def feature_stats(cache: FeatureCache) -> FeatureStats:
    """Recompute exact C-order hashes and raw population feature statistics."""
    if (
        cache.features.dtype != np.float32
        or cache.labels.dtype != np.int64
        or cache.labels.ndim != 1
        or cache.features.ndim != MATRIX_DIMENSIONS
        or cache.features.shape[0] != cache.labels.size
        or not cache.labels.size
        or not cache.features.shape[1]
        or bool((cache.labels < 0).any())
        or not bool(np.isfinite(cache.features).all())
    ):
        message = "Invalid finite float32 feature cache"
        raise ValueError(message)
    return FeatureStats(
        feature_sha256=sha256(cache.features.tobytes()).hexdigest(),
        label_sha256=sha256(cache.labels.tobytes()).hexdigest(),
        variance=tuple(float(value) for value in cache.features.var(axis=0).flat),
        saturation_fraction=float(
            (np.abs(cache.features) >= SATURATION_THRESHOLD).mean()
        ),
    )


def save_head(
    path: Path,
    result: ProbeResult,
    source: ProbeSourceCheckpoint,
    group: ProbeGroupRecord,
    extraction: ExtractionConfig,
) -> HeadArtifact:
    """Write compressed float32 parameters and compact JSON; validate the roundtrip."""
    parameters_path = path.with_suffix(".npz")
    write_npz(
        parameters_path,
        {
            "weight": np.asarray(result.parameters.weight, dtype=np.float32),
            "bias": np.asarray(result.parameters.bias, dtype=np.float32),
            "mean": np.asarray(result.parameters.mean, dtype=np.float32),
            "std": np.asarray(result.parameters.std, dtype=np.float32),
        },
    )
    artifact = HeadArtifact(
        source=source,
        group=group,
        extraction=extraction,
        result=CompactProbeResult.model_validate(
            result.model_dump(exclude={"parameters"})
        ),
        parameters_file=parameters_path.name,
        parameters_sha256=file_digest(parameters_path),
    )
    _ = path.write_text(artifact.model_dump_json() + "\n")
    restored_artifact, restored_result = load_head(path)
    if restored_artifact != artifact or restored_result != result:
        message = "Probe head artifact roundtrip differs"
        raise ValueError(message)
    return artifact


def load_head(path: Path) -> tuple[HeadArtifact, ProbeResult]:
    """Reject wrong keys, dtype, shape, nonfinite values, counts and trace progress."""
    artifact = HeadArtifact.model_validate_json(path.read_bytes())
    parameters_path = path.with_suffix(".npz")
    width, vocab = len(artifact.group.indices), artifact.result.config.vocab_size
    expected = {
        "weight": (width, vocab),
        "bias": (vocab,),
        "mean": (width,),
        "std": (width,),
    }
    if (
        artifact.parameters_file != parameters_path.name
        or file_digest(parameters_path) != artifact.parameters_sha256
        or width != PROBE_WIDTH
        or len(set(artifact.group.indices)) != width
        or artifact.result.parameter_count != (width + 1) * vocab
        or tuple(step.update for step in artifact.result.trace)
        != tuple(range(1, artifact.result.config.updates + 1))
    ):
        message = "Probe head artifact identity or completeness differs"
        raise ValueError(message)
    with (
        parameters_path.open("rb") as stream,
        NpzFile(stream, allow_pickle=False) as data,
    ):
        if set(data.files) != set(expected) or len(data.files) != len(expected):
            message = "Probe parameter keys differ"
            raise ValueError(message)
        for name, shape in expected.items():
            array = data[name]
            if (
                array.dtype != np.float32
                or array.shape != shape
                or not np.isfinite(array).all()
            ):
                message = "Probe parameters must be finite float32 with exact shapes"
                raise ValueError(message)
        if bool((data["std"] < artifact.result.config.std_floor).any()):
            message = "Probe normalization is below its frozen floor"
            raise ValueError(message)
        parameters = ProbeParameters(
            weight=TypeAdapter(tuple[tuple[float, ...], ...]).validate_python(
                data["weight"].tolist()
            ),
            bias=TypeAdapter(tuple[float, ...]).validate_python(data["bias"].tolist()),
            mean=TypeAdapter(tuple[float, ...]).validate_python(data["mean"].tolist()),
            std=TypeAdapter(tuple[float, ...]).validate_python(data["std"].tolist()),
        )
    return artifact, ProbeResult.model_validate(
        {
            **artifact.result.model_dump(),
            "parameters": parameters,
        }
    )


def save_heldout(
    path: Path, indices: tuple[int, ...], valid: FeatureCache, test: FeatureCache
) -> str:
    """Store each source's heldout union once, preserving ordered int64 neuron IDs."""
    for cache in (valid, test):
        _ = feature_stats(cache)
        if cache.features.shape[1] != len(indices) or len(set(indices)) != len(indices):
            message = "Heldout union indices differ from feature width"
            raise ValueError(message)
    write_npz(
        path,
        {
            "indices": np.asarray(indices, dtype=np.int64),
            "valid_features": valid.features,
            "valid_labels": valid.labels,
            "test_features": test.features,
            "test_labels": test.labels,
        },
    )
    return file_digest(path)


def load_heldout(
    path: Path, indices: tuple[int, ...], expected_sha256: str
) -> tuple[FeatureCache, FeatureCache]:
    """Read one checked union cache for independent local group-slice verification."""
    if file_digest(path) != expected_sha256:
        message = "Heldout union hash differs"
        raise ValueError(message)
    with path.open("rb") as stream, NpzFile(stream, allow_pickle=False) as data:
        expected = {
            "indices",
            "valid_features",
            "valid_labels",
            "test_features",
            "test_labels",
        }
        if (
            set(data.files) != expected
            or len(data.files) != len(expected)
            or data["indices"].dtype != np.int64
            or data["indices"].shape != (len(indices),)
            or TypeAdapter(tuple[int, ...]).validate_python(data["indices"].tolist())
            != indices
            or len(set(indices)) != len(indices)
        ):
            message = "Heldout union keys or indices differ"
            raise ValueError(message)
        caches: list[FeatureCache] = []
        for split in ("valid", "test"):
            features, labels = data[f"{split}_features"], data[f"{split}_labels"]
            cache = FeatureCache(features, labels)
            _ = feature_stats(cache)
            if features.shape[1] != len(indices):
                message = "Heldout union width differs"
                raise ValueError(message)
            caches.append(cache)
    return caches[0], caches[1]


def seal_zip(destination: Path, files: tuple[Path, ...]) -> None:
    """Publish only a complete, independently CRC-tested archive with unique names."""
    names = tuple(path.name for path in files)
    if (
        not files
        or len(set(names)) != len(names)
        or not all(path.is_file() for path in files)
    ):
        message = "Cannot package incomplete or duplicate regional probe artifacts"
        raise ValueError(message)
    temporary = destination.with_suffix(".partial")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
    with ZipFile(temporary) as archive:
        if tuple(archive.namelist()) != names or archive.testzip() is not None:
            message = "Regional probe archive CRC or membership differs"
            raise ValueError(message)
    _ = temporary.replace(destination)
