# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6,<3", "pydantic>=2.10,<3", "pytest>=8"]
# ///
# Run after ar_bootstrap.py on Colab:
# colab exec -s flyrl-ar-optimized -f scripts/ar_gpu_checks.py
"""Exercise the existing dense-reference derivative tests on actual CUDA tensors."""

import runpy
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from pydantic import TypeAdapter

root = Path("/content/flyrl-ar")
for filename, name in (
    ("test_ar_sparse.py", "test_sparse_forward_and_backward"),
    ("test_cached_topology.py", "test_cached_sparse_operator_matches_dense_gradients"),
):
    namespace = runpy.run_path(str(root / "tests" / filename))
    check = TypeAdapter[Callable[[], None]](Callable[[], None]).validate_python(
        namespace[name]
    )
    with torch.device("cuda"):
        check()
    _ = sys.stdout.write(f"AR_GPU_CHECK_PASSED {name}\n")
_ = sys.stdout.write("AR_GPU_GRADIENT_CHECKS_PASSED\n")
