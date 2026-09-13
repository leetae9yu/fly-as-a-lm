"""Strict runtime identity models for anatomy-factorial recovery."""

import pytest
from pydantic import ValidationError

from scripts.anatomy_factorial_recovery_schema import (
    CheckpointRuntime,
    WorkerRuntime,
)

RuntimeValue = int | float | str | bool


def worker_runtime() -> dict[str, RuntimeValue]:
    return {
        "seed": 7,
        "condition": "real-alpn_mbon",
        "wiring": "real",
        "port_policy": "alpn_mbon",
        "gpu": "Tesla T4",
        "python": "3.12.12",
        "torch": "2.8.0+cu126",
        "cuda": "12.6",
        "seconds": 1.0,
        "peak_allocated_bytes": 1024,
        "resumed": False,
    }


def test_worker_runtime_requires_every_identity_field() -> None:
    runtime = worker_runtime()
    del runtime["torch"]

    with pytest.raises(ValidationError):
        _ = WorkerRuntime.model_validate(runtime)


def test_worker_runtime_allows_exact_checkpoint_resume() -> None:
    runtime = worker_runtime()
    runtime["resumed"] = True

    assert WorkerRuntime.model_validate(runtime).resumed


def test_checkpoint_runtime_requires_deterministic_t4_flags() -> None:
    runtime = {
        "torch": "2.8.0+cu126",
        "device": "cuda",
        "name": "Tesla T4",
        "threads": 1,
        "deterministic": False,
        "tf32": False,
    }
    assert CheckpointRuntime.model_validate(runtime).device == "cuda"

    runtime["tf32"] = True
    with pytest.raises(ValidationError):
        _ = CheckpointRuntime.model_validate(runtime)
