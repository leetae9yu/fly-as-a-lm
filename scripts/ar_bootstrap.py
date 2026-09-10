# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: colab exec -s flyrl-autoregressive-t4 -f scripts/ar_bootstrap.py
"""Unpack the small source bundle and configure the persistent Colab kernel."""

import os
import sys
import tarfile
from pathlib import Path

root = Path("/content/flyrl-ar")
root.mkdir(exist_ok=True)
with tarfile.open("/content/flyrl-ar-source.tar.gz") as archive:
    archive.extractall(root, filter="data")
os.chdir(root)
sys.path.insert(0, str(root))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
_ = sys.stdout.write("AR_SOURCE_READY\n")
