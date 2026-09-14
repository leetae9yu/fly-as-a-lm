"""Independent transaction validation rejects corruption without a learner restore."""

from pathlib import Path
from typing import Literal, assert_never, get_args
from zipfile import ZipFile

import numpy as np
import pytest
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile

from flyrl.babi_checkpoint import Checkpoints
from flyrl.babi_pilot_schema import Manifest, Validation
from flyrl.babi_recovery import capture_state, record_event
from flyrl.babi_recovery_validation import checkpoint
from scripts.connectome_source import file_digest
from scripts.recover_babi_task1 import members
from tests.test_babi_checkpoint import fixture


def rewrite(
    path: Path, key: str, value: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
) -> None:
    with (
        path.open("rb") as stream,
        NpzFile[np.generic](stream, allow_pickle=False) as data,
    ):
        arrays = {name: data[name] for name in data.files}
    arrays[key] = value
    with ZipFile(path, "w") as archive:
        for name, array in arrays.items():
            with archive.open(name + ".npy", "w") as member:
                write_array(member, array, allow_pickle=False)


def test_checkpoint_when_complete_is_read_only(tmp_path: Path) -> None:
    # Given: a real optimizer-bearing transaction.
    protocol, learner, examples = fixture(tmp_path)
    ids = tuple(e.example_id for e in examples)
    store = Checkpoints(tmp_path / "run", protocol, ids)
    history = (Validation(update=0, correct=0, questions=2, nll=8.0),)
    _ = learner.update(examples, learning_rate=protocol.schedule.rate(1))
    expected = store.save(learner, history)
    before = {p: p.read_bytes() for p in store.path(1).iterdir()}
    # When: inspect without constructing or restoring any learner.
    actual = checkpoint(store.path(1), protocol, ids)
    # Then: all progress is bound and every byte remains unchanged.
    assert actual == expected
    assert {p: p.read_bytes() for p in before} == before


Corruption = Literal[
    "tensor",
    "dtype",
    "nan",
    "adam",
    "variance",
    "rng",
    "ledger",
    "lr",
    "trace",
    "history",
    "extra",
    "missing",
    "duplicate",
    "truncated",
]


@pytest.mark.parametrize("kind", get_args(Corruption))
def test_checkpoint_when_rehashed_corruption_rejected(
    tmp_path: Path, kind: Corruption
) -> None:
    # Given: a complete transaction, then corruption with refreshed outer hashes.
    protocol, learner, examples = fixture(tmp_path)
    ids = tuple(e.example_id for e in examples)
    store = Checkpoints(tmp_path / "run", protocol, ids)
    history = (Validation(update=0, correct=0, questions=2, nll=8.0),)
    _ = learner.update(examples, learning_rate=protocol.schedule.rate(1))
    manifest = store.save(learner, history)
    folder = store.path(1)
    path = folder / "checkpoint.npz"
    match kind:
        case "tensor" | "dtype" | "nan" | "adam" | "variance" | "rng":
            arrays_to_change = {
                "tensor": ("model.bias", np.zeros(1, dtype=np.float32)),
                "dtype": ("model.bias", np.zeros(6, dtype=np.float64)),
                "nan": ("model.bias", np.full(6, np.nan, dtype=np.float32)),
                "adam": ("adam.bias.step", np.asarray(0, dtype=np.float32)),
                "variance": ("adam.bias.exp_avg_sq", np.full(6, -1, dtype=np.float32)),
                "rng": ("rng", np.zeros(5056, dtype=np.uint8)),
            }
            key, value = arrays_to_change[kind]
            rewrite(path, key, value)
        case "ledger" | "lr" | "trace" | "history":
            changes = {
                "ledger": {"exposures": (("a", 100), ("b", 0))},
                "lr": {"next_lr": 0.2},
                "trace": {"trace": ()},
                "history": {"validation": ()},
            }
            progress = manifest.progress.model_copy(update=changes[kind])
            manifest = manifest.model_copy(update={"progress": progress})
            rewrite(path, "progress", np.asarray(progress.model_dump_json()))
        case "extra":
            rewrite(path, "extra", np.zeros(1, dtype=np.float32))
        case "missing":
            with ZipFile(path) as archive:
                arrays = {
                    name: archive.read(name)
                    for name in archive.namelist()
                    if name != "rng.npy"
                }
            with ZipFile(path, "w") as archive:
                for name, data in arrays.items():
                    archive.writestr(name, data)
        case "duplicate":
            with (
                ZipFile(path, "a") as archive,
                pytest.warns(UserWarning, match="Duplicate"),
            ):
                archive.writestr("rng.npy", archive.read("rng.npy"))
        case "truncated":
            _ = path.write_bytes(path.read_bytes()[:100])
        case _:
            assert_never(kind)
    manifest = Manifest(checkpoint_sha256=file_digest(path), progress=manifest.progress)
    _ = (folder / "manifest.json").write_text(manifest.model_dump_json())
    # When/Then: corruption fails even after the outer digest is repaired.
    with pytest.raises(ValueError, match=r"[Cc]heckpoint"):
        _ = checkpoint(folder, protocol, ids)


@pytest.mark.parametrize(
    "kind", ["extra", "missing", "symlink", "directory", "traversal"]
)
def test_members_when_unsafe_or_incomplete_rejects(tmp_path: Path, kind: str) -> None:
    # Given: a fixed expected bundle membership and one namespace corruption.
    root = tmp_path / "bundle"
    root.mkdir()
    _ = (root / "evidence.json").write_text("{}")
    expected = {"evidence.json"}
    if kind == "extra":
        _ = (root / "extra.json").write_text("{}")
    elif kind == "missing":
        (root / "evidence.json").unlink()
    elif kind == "symlink":
        (root / "evidence.json").unlink()
        (root / "evidence.json").symlink_to(tmp_path / "target")
    elif kind == "directory":
        (root / "extra").mkdir()
    else:
        expected = {"../evidence.json"}
    # When/Then: no extra, absent, aliased or escaping artifact is accepted.
    with pytest.raises(ValueError, match=r"[Mm]embership|[Uu]nsafe|[Rr]egular"):
        members(root, expected)


def test_capture_when_sampling_preview_preserves_rng(tmp_path: Path) -> None:
    # Given: a real learner with its private generator untouched.
    protocol, learner, examples = fixture(tmp_path)
    ids = tuple(e.example_id for e in examples)
    before = learner.window_rng.get_state().clone()
    # When: record the restoration comparator and next sample.
    result = capture_state(learner, ids, protocol.next_lr(0))
    # Then: no sampling state was consumed by the diagnostic.
    assert learner.window_rng.get_state().equal(before)
    assert len(result.next_ids) == protocol.config.batch_size
    assert result.next_lr == protocol.next_lr(0)


def test_event_when_test_open_precedes_selection_rejects(tmp_path: Path) -> None:
    # Given: a run with no immutable selected checkpoint record.
    protocol, _, _ = fixture(tmp_path)
    # When/Then: the test-open event cannot be recorded out of order.
    with pytest.raises(ValueError, match="event"):
        record_event(tmp_path, protocol, "test_opened")
