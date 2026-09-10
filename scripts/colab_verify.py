# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: colab exec -s flyrl-prototype -f scripts/colab_verify.py --timeout 180
"""Capture remote test output explicitly for the verification record."""

import subprocess
import sys
from pathlib import Path

verification = subprocess.run(
    [sys.executable, "-m", "pytest", "-q"],
    cwd="/content/flyrl",
    check=False,
    capture_output=True,
    text=True,
    timeout=120,
)
test_output = verification.stdout + verification.stderr
_ = Path("/content/flyrl-tests.txt").write_text(test_output)
_ = sys.stdout.write(test_output)
verification.check_returncode()
