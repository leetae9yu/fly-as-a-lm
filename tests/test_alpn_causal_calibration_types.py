"""Strict calibration records reject ambiguous and nonfinite evidence."""

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from scripts.alpn_causal_calibration_types import MatchingRecord, Matrix, Runtime, Seed
from scripts.alpn_causal_groups import Balance, MatchedControl, MatchingEvidence


def test_matrix_roundtrip_and_corruption() -> None:
    values = np.arange(32, dtype=np.float64).reshape(2, 16)
    matrix = Matrix.capture(values)
    restored = Matrix.model_validate_json(matrix.model_dump_json())
    np.testing.assert_array_equal(restored.array(2, 16), values)
    changed = restored.model_copy(update={"sha256": "a" * 64})
    with pytest.raises(ValueError, match="matrix"):
        _ = changed.array(2, 16)
    with pytest.raises(ValueError, match="matrix"):
        _ = restored.array(1, 16)
    with pytest.raises(ValidationError):
        _ = Matrix.model_validate_json(matrix.model_dump_json()[:-1] + ',"extra":0}')
    with pytest.raises(ValidationError):
        _ = Matrix.capture(np.array([[np.nan]], dtype=np.float64))


def test_infinite_smd_is_explicit_failure_not_null_or_nonfinite_json() -> None:
    evidence = MatchingEvidence(
        (1,),
        (0,),
        1,
        (0.0,),
        (1.0,),
        (
            MatchedControl(
                "S", 0, (0,), (1,), (1.0,), (Balance("in_degree", float("inf"), 1.0),)
            ),
        ),
        (),
    )
    record = MatchingRecord.capture(evidence)
    assert record.controls[0].balance[0].smd == "inf"
    assert record.status == "insufficient_common_support"
    assert MatchingRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threads", True),
        ("threads", 1.0),
        ("deterministic", 0),
        ("tf32", 0.0),
        ("deterministic", True),
        ("tf32", True),
    ],
)
def test_runtime_does_not_coerce_literal_flags(field: str, value: bool | float) -> None:
    fields: dict[str, str | int | float | bool] = {
        "gpu": "Tesla T4",
        "device": "cuda:0",
        "dtype": "torch.float32",
        "python": "test",
        "torch": "test",
        "cuda": "test",
        "threads": 1,
        "deterministic": False,
        "tf32": False,
    }
    fields[field] = value
    with pytest.raises(ValueError, match=r"threads|deterministic|tf32|flags"):
        _ = Runtime.model_validate(fields)


@pytest.mark.parametrize("value", [7.0, True, "7"])
def test_seed_does_not_coerce_numbers(value: float | bool | str) -> None:
    with pytest.raises(ValueError, match="int"):
        _ = TypeAdapter[int](Seed).validate_python(value, strict=True)
