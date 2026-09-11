# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.6,<3", "numpy>=2,<3", "pydantic>=2.10,<3", "tokenizers==0.22.2"
# ]
# ///
# Run from the project root: uv run --no-sync python -m scripts.profile_bpe_models
"""Measure disposable T4 models on train tokens without scoring held-out text."""

import sys
from pathlib import Path
from time import perf_counter
from typing import Final, Literal

import torch

from flyrl.ar_benchmark import Benchmark, benchmark, synchronize
from flyrl.ar_config import Architecture, ARConfig
from flyrl.ar_corpus import load_ar_corpus, vocabulary
from flyrl.ar_learning import make_learner
from flyrl.ar_reporting import counts
from flyrl.connectome import load_graph
from flyrl.language_checkpoint import atomic_text
from flyrl.language_models import Settings

WARM_UPDATES: Final = 20


class Profile(Settings):
    """Actual wall time, model scale and GPU peaks for one disposable condition."""

    config: ARConfig
    warm_updates: int
    warm_seconds: float
    seconds_per_update: float
    training_tokens_per_second: float
    peak_gpu_bytes: int
    parameter_counts: dict[str, int]
    cold: Benchmark


def main() -> None:
    """Measure each declared condition sequentially on one T4 in float32."""
    torch.set_num_threads(1)
    graph = load_graph(Path("data/large_connectome/malecns_v1_n16384.npz"))
    corpus = load_ar_corpus(Path("data/bpe_full/corpus.npz"))
    conditions: tuple[
        tuple[Architecture, tuple[Literal["real", "shuffled", "frozen"], ...], float],
        ...,
    ] = (
        ("connectome", ("real", "shuffled", "frozen"), 0.003),
        ("gru", ("real",), 0.001),
        ("transformer", ("real",), 0.001),
    )
    for architecture, controls, learning_rate in conditions:
        for control in controls:
            config = ARConfig.model_validate(
                {
                    "alphabet_size": len(vocabulary(corpus)),
                    "tokenization": "bpe",
                    "architecture": architecture,
                    "control": control,
                    "generation_context": "windowed",
                    "device": "cuda",
                    "batch_size": 8,
                    "context": 32,
                    "seed": 0,
                    "learning_rate": learning_rate,
                }
            )
            learner = make_learner(graph, config)
            cold = benchmark(learner)
            device = learner.model.weight.device
            synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = perf_counter()
            learner.train(corpus.train, WARM_UPDATES)
            synchronize(device)
            elapsed = perf_counter() - started
            result = Profile(
                config=config,
                warm_updates=WARM_UPDATES,
                warm_seconds=elapsed,
                seconds_per_update=elapsed / WARM_UPDATES,
                training_tokens_per_second=WARM_UPDATES * 8 * 32 / elapsed,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(device),
                parameter_counts=counts(learner),
                cold=cold,
            )
            atomic_text(
                Path("results/bpe-profile") / f"{config.condition}.json",
                result.model_dump_json(indent=2),
            )
            _ = sys.stdout.write("BPE_PROFILE " + result.model_dump_json() + "\n")
            _ = sys.stdout.flush()
            del learner
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
