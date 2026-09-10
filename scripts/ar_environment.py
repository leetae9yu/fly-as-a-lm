# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: colab exec -s flyrl-autoregressive-t4 -f scripts/ar_environment.py
"""Report package versions without initializing CUDA or mutating the runtime."""

import json
import platform
import sys
from importlib.metadata import version

_ = sys.stdout.write(
    json.dumps(
        {
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("numpy", "pydantic", "typer", "pytest", "torch", "triton")
            },
        }
    )
    + "\n"
)
