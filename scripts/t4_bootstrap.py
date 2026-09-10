# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: colab exec -s flyrl-language-t4 -f scripts/t4_bootstrap.py --timeout 1800
"""Unpack the uploaded experiment and enter its T4 verification runner."""

import os
import runpy
import sys
import tarfile
from pathlib import Path

root = Path("/content/flyrl-language")
root.mkdir(exist_ok=True)
with tarfile.open("/content/flyrl-language-source.tar.gz") as archive:
    archive.extractall(root, filter="data")
os.chdir(root)
sys.path.insert(0, str(root))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
_ = runpy.run_module("scripts.t4_language_run", run_name="__main__")
