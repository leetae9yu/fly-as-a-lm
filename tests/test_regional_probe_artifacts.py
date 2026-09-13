"""Tiny deterministic baseline and independently recoverable artifact checks."""

from math import log
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from flyrl.regional_probe import fit_probe
from flyrl.regional_probe_baseline import story_baselines, story_pairs
from flyrl.regional_probe_types import ExtractionConfig, FeatureCache, ProbeConfig
from flyrl.story_data import StorySplit
from scripts.connectome_source import file_digest
from scripts.regional_probe_artifact_types import HeadArtifact
from scripts.regional_probe_artifacts import (
    feature_stats,
    load_head,
    load_heldout,
    save_head,
    save_heldout,
    seal_zip,
    write_npz,
)
from scripts.regional_probe_manifest import ProbeGroupRecord
from scripts.regional_probe_protocol import ProbeSourceCheckpoint
from scripts.regional_probe_source_worker import slice_cache


def test_story_baselines_exact_train_only_counts_and_unseen_context() -> None:
    train = np.array([0, 1, 1, 2], dtype=np.int64)
    test = np.array([2, 0, 0, 1], dtype=np.int64)
    split = StorySplit(offsets=(0, 2, 4), sha256=("one", "two"))
    contexts, targets = story_pairs(train, split, 3)
    np.testing.assert_array_equal(contexts, [0, 1])
    np.testing.assert_array_equal(targets, [1, 2])
    unigram, bigram = story_baselines(train, split, test, split, 3)
    assert unigram.nll == pytest.approx(-(log(1 / 7) + log(3 / 7)) / 2)
    assert bigram.nll == pytest.approx(-(log(1 / 3) + log(3 / 5)) / 2)
    assert unigram.accuracy == 0.5
    assert bigram.accuracy == 1
    assert unigram.tokens == bigram.tokens == 2
    assert (unigram, bigram) == story_baselines(train, split, test, split, 3)


def test_story_pairs_skip_empty_singleton_and_reject_invalid() -> None:
    tokens = np.array([0, 1, 2], dtype=np.int64)
    split = StorySplit(offsets=(0, 0, 1, 3), sha256=("empty", "single", "pair"))
    contexts, targets = story_pairs(tokens, split, 3)
    np.testing.assert_array_equal(contexts, [1])
    np.testing.assert_array_equal(targets, [2])
    with pytest.raises(ValueError, match="boundaries"):
        _ = story_pairs(tokens, split.model_copy(update={"offsets": (1, 3)}), 3)
    with pytest.raises(ValueError, match="tokens"):
        _ = story_pairs(tokens + 1, split, 3)
    with pytest.raises(ValueError, match="pairs"):
        _ = story_pairs(
            np.array([0], dtype=np.int64), StorySplit(offsets=(0, 1), sha256=("a",)), 3
        )


@pytest.fixture
def source() -> ProbeSourceCheckpoint:
    return ProbeSourceCheckpoint(
        seed=7,
        wiring="real",
        filename="seed-7-real-random_random.zip",
        archive_sha256="a" * 64,
        archive_bytes=123,
        checkpoint_sha256="b" * 64,
        report_sha256="c" * 64,
        runtime_sha256="d" * 64,
        parameter_fingerprint="e" * 64,
    )


@pytest.fixture
def cache() -> FeatureCache:
    return FeatureCache(
        np.arange(4 * 100, dtype=np.float32).reshape(4, 100) / np.float32(400),
        np.array([0, 1, 1, 0], dtype=np.int64),
    )


def test_head_union_roundtrip_and_crc_package(
    tmp_path: Path, source: ProbeSourceCheckpoint, cache: FeatureCache
) -> None:
    torch.set_num_threads(1)
    union = tuple(range(100))
    group = ProbeGroupRecord(
        name="mbon",
        draw=0,
        indices=union[:97],
        node_ids=tuple(str(i) for i in union[:97]),
    )
    sliced = slice_cache(cache, union, group.indices)
    result = fit_probe(sliced, sliced, sliced, ProbeConfig(vocab_size=2, updates=2))
    head_path, heldout_path = tmp_path / "mbon-0.json", tmp_path / "heldout.npz"
    artifact = save_head(head_path, result, source, group, ExtractionConfig())
    restored_artifact, restored = load_head(head_path)
    assert artifact == restored_artifact
    assert restored == result
    assert "parameters" not in artifact.result.model_dump()
    assert artifact.source == source
    assert artifact.result.parameter_count == 196
    digest = save_heldout(heldout_path, union, cache, cache)
    valid, test = load_heldout(heldout_path, union, digest)
    for restored_cache in (valid, test):
        assert (
            feature_stats(slice_cache(restored_cache, union, group.indices))
            == result.features[1]
        )
    with ZipFile(head_path.with_suffix(".npz")) as arrays:
        assert arrays.testzip() is None
        assert set(arrays.namelist()) == {
            "weight.npy",
            "bias.npy",
            "mean.npy",
            "std.npy",
        }
    files = (head_path, head_path.with_suffix(".npz"), heldout_path)
    archive_path = tmp_path / "source.zip"
    seal_zip(archive_path, files)
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert tuple(archive.namelist()) == tuple(path.name for path in files)
        for path in files:
            assert archive.read(path.name) == path.read_bytes()


def test_artifacts_reject_nonfinite_arrays_and_incomplete_output(
    tmp_path: Path, source: ProbeSourceCheckpoint, cache: FeatureCache
) -> None:
    union = tuple(range(100))
    digest = save_heldout(tmp_path / "heldout.npz", union, cache, cache)
    with pytest.raises(ValueError, match="hash"):
        _ = load_heldout(tmp_path / "heldout.npz", union, "wrong")
    with pytest.raises(ValueError, match="indices"):
        _ = load_heldout(tmp_path / "heldout.npz", tuple(reversed(union)), digest)
    invalid = FeatureCache(cache.features.copy(), cache.labels)
    invalid.features[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _ = save_heldout(tmp_path / "invalid.npz", union, invalid, cache)
    with pytest.raises(ValueError, match="incomplete"):
        seal_zip(tmp_path / "bad.zip", (tmp_path / "missing",))
    group = ProbeGroupRecord(
        name="mbon",
        draw=0,
        indices=union[:97],
        node_ids=tuple(str(i) for i in union[:97]),
    )
    sliced = slice_cache(cache, union, group.indices)
    result = fit_probe(sliced, sliced, sliced, ProbeConfig(vocab_size=2, updates=0))
    path = tmp_path / "mbon-0.json"
    artifact = save_head(path, result, source, group, ExtractionConfig())
    write_npz(
        path.with_suffix(".npz"),
        {
            "weight": np.full((97, 2), np.nan, dtype=np.float32),
            "bias": np.zeros(2, dtype=np.float32),
            "mean": np.zeros(97, dtype=np.float32),
            "std": np.ones(97, dtype=np.float32),
        },
    )
    corrupt = HeadArtifact.model_validate(
        {
            **artifact.model_dump(),
            "parameters_sha256": file_digest(path.with_suffix(".npz")),
        }
    )
    _ = path.write_text(corrupt.model_dump_json())
    with pytest.raises(ValueError, match="finite"):
        _ = load_head(path)
