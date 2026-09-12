"""Adversarial checks for sealed replication checkpoint recovery."""

from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pytest
import torch
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile

from scripts.central_replication_validation import validate_checkpoint_tensors

SHAPES = {
    "codes": (3, 2),
    "weight": (2,),
    "bias": (2,),
    "readout": (1, 3),
    "output_bias": (3,),
}


def write_complete_checkpoint(
    path: Path, rng: np.ndarray[tuple[int], np.dtype[np.uint8]]
) -> None:
    """Write a small but structurally complete AdamW checkpoint."""
    arrays = {"metadata": np.asarray("{}"), "rng": rng}
    for name, shape in SHAPES.items():
        arrays[f"model.{name}"] = np.zeros(shape, dtype=np.float32)
        arrays[f"adam.{name}.exp_avg"] = np.zeros(shape, dtype=np.float32)
        arrays[f"adam.{name}.exp_avg_sq"] = np.zeros(shape, dtype=np.float32)
        arrays[f"adam.{name}.step"] = np.asarray(1000, dtype=np.float32)
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w") as member:
                write_array(member, array, allow_pickle=False)


def test_checkpoint_validation_rejects_missing_mutable_state(tmp_path: Path) -> None:
    # Given: the two arrays accepted by the old, incomplete recovery check.
    checkpoint = tmp_path / "truncated.npz"
    with ZipFile(checkpoint, "w", compression=ZIP_STORED) as archive:
        for name, array in {
            "metadata": np.asarray("{}"),
            "model.weight": np.zeros(2, dtype=np.float32),
            "model.codes": np.zeros((3, 2), dtype=np.float32),
        }.items():
            with archive.open(f"{name}.npy", "w") as member:
                write_array(member, array, allow_pickle=False)
    # When/Then: absence of RNG, biases, readout and Adam state is fatal.
    with (
        checkpoint.open("rb") as stream,
        NpzFile(stream, allow_pickle=False) as data,
        pytest.raises(ValueError, match="key set"),
    ):
        validate_checkpoint_tensors(
            data,
            SHAPES,
            updates=1000,
        )


def test_checkpoint_validation_accepts_restorable_rng_state(tmp_path: Path) -> None:
    # Given: all expected tensors and a real CPU generator state.
    checkpoint = tmp_path / "complete.npz"
    write_complete_checkpoint(checkpoint, torch.Generator().get_state().numpy())
    # When/Then: the same API used by the loader can restore the RNG.
    with checkpoint.open("rb") as stream, NpzFile(stream, allow_pickle=False) as data:
        validate_checkpoint_tensors(data, SHAPES, updates=1000)


def test_checkpoint_validation_rejects_unrestorable_rng_state(tmp_path: Path) -> None:
    # Given: every expected tensor but an invalid one-byte RNG payload.
    checkpoint = tmp_path / "bad-rng.npz"
    write_complete_checkpoint(checkpoint, np.zeros(1, dtype=np.uint8))
    # When/Then: structural validation must exercise torch.Generator.set_state.
    with (
        checkpoint.open("rb") as stream,
        NpzFile(stream, allow_pickle=False) as data,
        pytest.raises(RuntimeError, match="GeneratorImplState"),
    ):
        validate_checkpoint_tensors(data, SHAPES, updates=1000)
