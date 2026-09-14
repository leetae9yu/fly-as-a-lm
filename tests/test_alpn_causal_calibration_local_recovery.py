"""Cross-runtime recovery tolerates only insignificant coordinate rounding."""

import numpy as np
import pytest

from scripts.alpn_causal_calibration_local_recovery import validate_coordinates


def test_coordinate_recovery_allows_only_cross_runtime_rounding() -> None:
    expected = np.ones((2, 16), dtype=np.float64)
    rounded = np.nextafter(expected, np.zeros_like(expected))
    validate_coordinates(rounded, expected)
    changed = expected.copy()
    changed[0, 0] += 1e-9
    with pytest.raises(ValueError, match="coordinate evidence"):
        validate_coordinates(changed, expected)
