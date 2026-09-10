# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6,<3", "numpy>=2.1,<3", "pydantic>=2.10,<3", "typer>=0.20"]
# ///
# colab exec -s flyrl-ar-optimized -f scripts/ar_finish_colab.py --timeout 180
"""Verify each trained control, then export only final results and resume evidence."""

import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from flyrl.ar_reporting import file_identity
from scripts.verify_ar_resume import Command

root = Path("/content/flyrl-ar")
for control in ("real", "shuffled", "frozen"):
    _ = Command(
        graph=root / "data/large_connectome/malecns_v1_n16384.npz",
        corpus=root / "data/ar_corpus/corpus.npz",
        run=root / "results/ar-main/seed-0" / control,
        output=root / "results/ar-resume" / control,
    )

destination = root / "ar-results.zip"
with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
    for group in ("ar-main", "ar-capacity", "ar-resume"):
        for path in sorted((root / "results" / group).rglob("*")):
            if path.is_file() and path.name != "midpoint.npz":
                archive.write(path, path.relative_to(root))
with ZipFile(destination) as archive:
    damaged = archive.testzip()
    if damaged is not None:
        message = f"Invalid result archive member: {damaged}"
        raise RuntimeError(message)
_ = sys.stdout.write(
    "AR_RESULTS_READY "
    + json.dumps(
        {
            "path": str(destination),
            "sha256": file_identity(destination),
            "bytes": destination.stat().st_size,
        }
    )
    + "\n"
)
