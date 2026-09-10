# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6,<3", "numpy>=2,<3", "pydantic>=2.10,<3"]
# ///
# Run inside the uploaded project: python -m scripts.t4_language_run
"""Run the prescribed natural-text experiment and an exact CUDA resume check."""

import json
import platform
import runpy
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import torch

from flyrl.language_experiment import Summary

root = Path.cwd()
output = root / "results" / "language-t4"
output.mkdir(parents=True, exist_ok=True)
gpu = torch.cuda.get_device_name(0)
if "T4" not in gpu:
    message = f"The prescribed run requires a T4, received {gpu}"
    raise RuntimeError(message)

tests = subprocess.run(
    [sys.executable, "-m", "pytest", "-q"],
    check=False,
    capture_output=True,
    text=True,
    timeout=180,
)
_ = (output / "tests.txt").write_text(tests.stdout + tests.stderr)
_ = sys.stdout.write(tests.stdout + tests.stderr)
tests.check_returncode()

started = perf_counter()
for directory, updates, seeds in (
    ("resumed", 500, "0,1,2"),
    ("resumed", 1000, "0,1,2"),
    ("uninterrupted-seed0", 1000, "0"),
):
    previous_arguments = sys.argv
    sys.argv = [
        "flyrl.language",
        "--graph",
        "data/larva_left_mb.npz",
        "--corpus",
        "data/wikitext2/corpus.npz",
        "--output",
        str(output / directory),
        "--device",
        "cuda",
        "--updates",
        str(updates),
        "--batch-size",
        "128",
        "--context",
        "16",
        "--seeds",
        seeds,
        "--eval-windows",
        "2048",
        "--sample-length",
        "120",
        "--checkpoint-every",
        "100",
        "--learning-rate",
        "0.1",
        "--resume",
    ]
    run_started = perf_counter()
    try:
        with (
            (output / f"{directory}-{updates}.log").open("w") as log,
            redirect_stdout(log),
            redirect_stderr(log),
        ):
            try:
                _ = runpy.run_module("flyrl.language", run_name="__main__")
            except SystemExit as result:
                if result.code not in (0, None):
                    raise
    finally:
        sys.argv = previous_arguments
    seconds = perf_counter() - run_started
    _ = sys.stdout.write(f"T4_RUN_COMPLETED {directory} {updates} {seconds:.3f}s\n")

resumed = Summary.model_validate_json(
    (output / "resumed" / "summary.json").read_bytes()
)
reference = Summary.model_validate_json(
    (output / "uninterrupted-seed0" / "summary.json").read_bytes()
)
matches = 0
for run in reference.runs:
    peer = next(
        candidate
        for candidate in resumed.runs
        if candidate.seed == run.seed and candidate.control == run.control
    )
    with (
        ZipFile(output / "resumed" / peer.checkpoint) as left,
        ZipFile(output / "uninterrupted-seed0" / run.checkpoint) as right,
    ):
        if left.namelist() != right.namelist() or any(
            left.read(name) != right.read(name) for name in left.namelist()
        ):
            message = f"CUDA resume checkpoint mismatch: {run.control}-{run.seed}"
            raise RuntimeError(message)
    if (
        peer.final != run.final
        or peer.final_weights != run.final_weights
        or peer.training_rewards != run.training_rewards
        or peer.continuation != run.continuation
    ):
        message = f"CUDA resume behavior mismatch: {run.control}-{run.seed}"
        raise RuntimeError(message)
    matches += 1

if matches != len(("real", "shuffled", "frozen")):
    message = "Resume verification did not cover all three seed-zero controls"
    raise RuntimeError(message)
if any(run.device.tensor_device != "cuda:0" for run in resumed.runs):
    message = "A reported run did not execute on CUDA"
    raise RuntimeError(message)

evidence = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "cuda": torch.version.cuda,
    "gpu": gpu,
    "elapsed_seconds": perf_counter() - started,
    "resume_checkpoint_and_behavior_matches": matches,
    "resume_seed": 0,
    "main_runs": len(resumed.runs),
    "corpus_fingerprint": resumed.corpus_fingerprint,
    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
}
_ = (output / "evidence.json").write_text(json.dumps(evidence, indent=2))
_ = shutil.make_archive("/content/flyrl-language-results", "zip", output)
_ = sys.stdout.write(f"T4_LANGUAGE_VERIFIED {json.dumps(evidence)}\n")
