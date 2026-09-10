# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6,<3", "numpy>=2.1,<3", "pydantic>=2.10,<3", "typer>=0.20"]
# ///
# Run after ar_bootstrap.py and upload the current source:
# colab exec -s flyrl-ar-optimized -f scripts/ar_train_colab.py --timeout 3600
"""Fixed-budget natural-text pilot selected using capacity measurements only."""

import subprocess
import sys

_ = sys.stdout.write("AR_TRAIN_BEGIN n16384 seed0 updates4000\n")
_ = subprocess.run(
    [
        sys.executable,
        "-m",
        "flyrl.autoregressive",
        "--graph",
        "/content/flyrl-ar/data/large_connectome/malecns_v1_n16384.npz",
        "--corpus",
        "/content/flyrl-ar/data/ar_corpus/corpus.npz",
        "--output",
        "/content/flyrl-ar/results/ar-main",
        "--device",
        "cuda",
        "--updates",
        "4000",
        "--batch-size",
        "8",
        "--context",
        "32",
        "--seeds",
        "0",
        "--controls",
        "real,shuffled,frozen",
        "--learning-rate",
        "0.003",
        "--eval-windows",
        "1024",
        "--sample-length",
        "240",
        "--checkpoint-steps",
        "200",
        "--readout-neurons",
        "256",
        "--edge-chunk",
        "65536",
        "--progress",
    ],
    cwd="/content/flyrl-ar",
    check=True,
)
_ = sys.stdout.write("AR_TRAIN_COMPLETE\n")
