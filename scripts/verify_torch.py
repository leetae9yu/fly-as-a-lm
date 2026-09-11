# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6,<3", "pytest>=8"]
# ///
# Run: uv run --no-sync python -m scripts.verify_torch -q tests/test_ar_sparse.py
"""Run Torch checks with sparse invariant validation explicitly enabled."""

import sys

import pytest
import torch


def main() -> None:
    """Use strict sparse checking without filtering framework warnings."""
    with torch.sparse.check_sparse_tensor_invariants(enable=True):
        raise SystemExit(pytest.main(sys.argv[1:]))


if __name__ == "__main__":
    main()
