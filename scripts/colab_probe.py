# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: colab exec -s flyrl-prototype -f scripts/colab_probe.py
"""Report the allocated runtime without modifying its environment."""

import json
import os
import platform
import sys
from importlib.metadata import version

_ = sys.stdout.write(
    json.dumps(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "numpy": version("numpy"),
        }
    )
    + "\n"
)
