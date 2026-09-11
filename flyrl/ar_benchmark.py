"""Synchronized forward/backward/AdamW capacity measurements, not throughput guesses."""

import resource
from time import perf_counter
from typing import assert_never

import torch

from flyrl.ar_config import Architecture
from flyrl.ar_framework import optimizer_step
from flyrl.ar_learning import ExperimentLearner
from flyrl.ar_model import ConnectomeLM
from flyrl.gru_model import GRULM
from flyrl.language_models import Settings
from flyrl.transformer_model import TransformerLM


class Benchmark(Settings):
    """Cold full-context phases; CUDA peaks include all live model tensors."""

    device: str
    forward_seconds: float
    backward_seconds: float
    optimizer_seconds: float
    forward_peak_gpu_bytes: int
    backward_peak_gpu_bytes: int
    optimizer_peak_gpu_bytes: int
    process_peak_rss_bytes: int
    nodes: int
    edges: int
    batch_size: int
    context: int
    edge_chunk: int
    architecture: Architecture | None = None
    vocabulary_tokens: int | None = None
    trainable_parameters: int | None = None


def synchronize(device: torch.device) -> None:
    """Wait for the selected GPU only, so wall times measure completed kernels."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _start(device: torch.device) -> float:
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return perf_counter()


def _finish(device: torch.device, start: float) -> tuple[float, int]:
    synchronize(device)
    return perf_counter() - start, (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )


def benchmark(learner: ExperimentLearner) -> Benchmark:
    """Consume one synthetic batch and optimizer step; use a disposable learner."""
    config, device = learner.config, learner.model.weight.device
    tokens = (
        torch.arange(config.batch_size * (config.context + 1), device=device).reshape(
            config.batch_size, config.context + 1
        )
        % config.alphabet_size
    )
    start = _start(device)
    loss = learner.loss(tokens)
    forward, forward_peak = _finish(device, start)
    start = _start(device)
    torch.autograd.backward(loss)
    backward, backward_peak = _finish(device, start)
    start = _start(device)
    _ = torch.nn.utils.clip_grad_norm_(
        learner.model.parameters(), config.gradient_clip, error_if_nonfinite=True
    )
    optimizer_step(learner.optimizer)
    optimizer, optimizer_peak = _finish(device, start)
    match learner.model:
        case ConnectomeLM() as model:
            nodes, edges = model.nodes, model.weight.numel()
        case GRULM() | TransformerLM():
            nodes, edges = 0, 0
        case _:
            assert_never(learner.model)
    return Benchmark(
        device=str(device),
        forward_seconds=forward,
        backward_seconds=backward,
        optimizer_seconds=optimizer,
        forward_peak_gpu_bytes=forward_peak,
        backward_peak_gpu_bytes=backward_peak,
        optimizer_peak_gpu_bytes=optimizer_peak,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        nodes=nodes,
        edges=edges,
        batch_size=config.batch_size,
        context=config.context,
        edge_chunk=config.edge_chunk,
        architecture=config.architecture,
        vocabulary_tokens=config.alphabet_size,
        trainable_parameters=sum(
            p.numel() for p in learner.model.parameters() if p.requires_grad
        ),
    )
