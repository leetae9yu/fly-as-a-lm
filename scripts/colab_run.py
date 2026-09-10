# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: colab exec -s flyrl-prototype -f scripts/colab_run.py --timeout 900
"""Execute the uploaded source bundle and verify exact checkpoint resumption."""

import hashlib
import json
import os
import platform
import resource
import runpy
import shutil
import subprocess
import sys
import tarfile
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Final

EXPECTED_JSON_FILES: Final = 19  # Two tasks x three controls x three seeds + summary.

root = Path("/content/flyrl")
root.mkdir(exist_ok=True)
with tarfile.open("/content/flyrl-source.tar.gz") as source:
    source.extractall(root, filter="data")
os.chdir(root)
sys.path.insert(0, str(root))
started = time.perf_counter()
_ = subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True, timeout=180)
_ = subprocess.run(
    [sys.executable, "-m", "scripts.prepare_data"], check=True, timeout=60
)

for directory, episodes in (
    ("results/colab-resumed", 1000),
    ("results/colab-resumed", 1500),
    ("results/colab-uninterrupted", 1500),
):
    run_started = time.perf_counter()
    previous_arguments = sys.argv
    sys.argv = [
        "flyrl",
        "--graph",
        "data/larva_left_mb.npz",
        "--output",
        directory,
        "--episodes",
        str(episodes),
        "--seeds",
        "0,1,2",
        "--delay",
        "3",
        "--eval-trials",
        "512",
        "--checkpoint-every",
        "250",
        "--resume",
    ]
    log_path = root / "results" / f"{Path(directory).name}-{episodes}.log"
    log_path.parent.mkdir(exist_ok=True)
    try:
        with log_path.open("w") as log, redirect_stdout(log), redirect_stderr(log):
            try:
                _ = runpy.run_module("flyrl", run_name="__main__")
            except SystemExit as result:
                if result.code not in (None, 0):
                    raise
    finally:
        sys.argv = previous_arguments
    status = f"RUN_COMPLETED directory={directory} episodes={episodes}"
    elapsed = time.perf_counter() - run_started
    _ = sys.stdout.write(f"{status} seconds={elapsed:.3f}\n")

resumed = root / "results" / "colab-resumed"
uninterrupted = root / "results" / "colab-uninterrupted"
compared = 0
checkpoints = sorted(resumed.glob("*.json"))
if len(checkpoints) != EXPECTED_JSON_FILES:
    message = f"Expected {EXPECTED_JSON_FILES} result files, got {len(checkpoints)}"
    raise RuntimeError(message)
for checkpoint in checkpoints:
    if checkpoint.read_bytes() != (uninterrupted / checkpoint.name).read_bytes():
        message = f"Resume differs from uninterrupted run: {checkpoint.name}"
        raise RuntimeError(message)
    compared += 1

evidence = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    "elapsed_seconds": time.perf_counter() - started,
    "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    "peak_kernel_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "resume_equal_json_files": compared,
    "source_sha256": hashlib.sha256(
        Path("/content/flyrl-source.tar.gz").read_bytes()
    ).hexdigest(),
}
_ = (root / "results" / "colab-evidence.json").write_text(
    json.dumps(evidence, indent=2) + "\n"
)
_ = shutil.make_archive("/content/flyrl-results", "zip", root / "results")
_ = sys.stdout.write(f"COLAB_VERIFIED {json.dumps(evidence)}\n")
