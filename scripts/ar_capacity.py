# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.6,<3", "numpy>=2,<3", "pydantic>=2.10,<3", "pyarrow==21.0.0",
# ]
# ///
# Run on Colab after ar_bootstrap.py:
# colab exec -s flyrl-autoregressive-t4 -f scripts/ar_capacity.py
"""Measure completed sparse forward/backward/optimizer work on real graph sizes."""

import gc
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Final

import torch

from flyrl.ar_benchmark import Benchmark, benchmark
from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.ar_reporting import file_identity
from flyrl.connectome import load_graph
from flyrl.language_models import Settings
from scripts.connectome_manifest import Manifest

PART_BYTES: Final = 33_554_432
LARGE_GRAPH_NODES: Final = 16_384


class CapacityEntry(Settings):
    """Completed optimizer trials, or an explicit allocation failure."""

    graph: str
    graph_sha256: str
    load_seconds: float
    samples: tuple[Benchmark, ...]
    allocation_error: str | None = None


class CapacityReport(Settings):
    """Hardware observation and per-size measurements, without reading test scores."""

    gpu: str
    torch_version: str
    entries: tuple[CapacityEntry, ...]
    context: int = 32
    batch_size: int = 8
    interpretation: str = (
        "first sample cold; later samples warm; compare edge_chunk groups separately"
    )


root = Path("/content/flyrl-ar")
directory = root / "data" / "large_connectome"
manifest = Manifest.model_validate_json((directory / "manifest.json").read_bytes())
full = next(item for item in manifest.artifacts if item.path == "malecns_v1_full.npz")
destination = directory / full.path
if not destination.exists():
    partial = destination.with_suffix(".partial")
    with partial.open("wb") as assembled:
        for index in range((full.bytes + PART_BYTES - 1) // PART_BYTES):
            with Path(f"/content/full.part.{index:03d}").open("rb") as part:
                shutil.copyfileobj(part, assembled)
    if file_identity(partial) != full.sha256:
        message = "Reassembled full graph does not match its verified source hash"
        raise RuntimeError(message)
    _ = partial.replace(destination)
if file_identity(destination) != full.sha256:
    message = "Full graph checksum mismatch"
    raise RuntimeError(message)

torch.set_num_threads(1)
gpu = torch.cuda.get_device_name(0)
if "T4" not in gpu:
    message = f"Expected T4 for the capacity experiment, received {gpu}"
    raise RuntimeError(message)
entries: list[CapacityEntry] = []
output = root / "results" / "ar-capacity"
output.mkdir(parents=True, exist_ok=True)
for size in (1024, 4096, 16384, 166700):
    artifact = next(item for item in manifest.artifacts if item.neurons == size)
    path = directory / artifact.path
    if file_identity(path) != artifact.sha256:
        message = f"Graph source mismatch: {artifact.path}"
        raise RuntimeError(message)
    _ = sys.stdout.write(f"AR_CAPACITY_BEGIN {size} {artifact.directed_edges}\n")
    started = perf_counter()
    graph = load_graph(path)
    load_seconds = perf_counter() - started
    learner = ARLearner(
        graph,
        ARConfig(alphabet_size=48, device="cuda", batch_size=8, context=32),
    )
    samples: list[Benchmark] = []
    allocation_error: str | None = None
    try:
        chunks = (65536, 1048576) if size >= LARGE_GRAPH_NODES else (65536,)
        for chunk in chunks:
            learner.config = learner.config.model_copy(update={"edge_chunk": chunk})
            learner.model.config = learner.config
            for _iteration in range(3):
                learner.optimizer.zero_grad(set_to_none=True)
                measured = benchmark(learner)
                samples.append(measured)
                _ = sys.stdout.write(
                    f"AR_CAPACITY_SAMPLE {measured.model_dump_json()}\n"
                )
    except torch.OutOfMemoryError as error:
        allocation_error = str(error)
        _ = sys.stdout.write(f"AR_CAPACITY_OOM {size} {allocation_error}\n")
    entries.append(
        CapacityEntry(
            graph=artifact.path,
            graph_sha256=artifact.sha256,
            load_seconds=load_seconds,
            samples=tuple(samples),
            allocation_error=allocation_error,
        )
    )
    report = CapacityReport(
        gpu=gpu, torch_version=str(torch.__version__), entries=tuple(entries)
    )
    _ = (output / "capacity.json").write_text(report.model_dump_json(indent=2))
    del learner, graph
    _ = gc.collect()
    torch.cuda.empty_cache()

_ = sys.stdout.write("AR_CAPACITY_COMPLETE\n")
