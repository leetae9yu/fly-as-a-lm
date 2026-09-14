"""Story-safe frozen states and a device-explicit linear softmax probe."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import accumulate
from math import cos, exp, log, pi
from sys import float_info
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import torch
from torch.nn.functional import cross_entropy

from flyrl.ar_framework import optimizer_step
from flyrl.ar_model import ConnectomeLM, OutgoingIntervention
from flyrl.ar_prepared import prepare_recurrence
from flyrl.language_data import IntVector
from flyrl.regional_probe_types import (
    ExtractionConfig,
    FeatureCache,
    FeatureStats,
    ProbeConfig,
    ProbeMetrics,
    ProbeParameters,
    ProbeResult,
    ProbeStep,
)
from flyrl.story_data import StorySplit

if TYPE_CHECKING:
    from numpy.typing import NDArray

PROBE_WIDTH: Final = 97
SATURATION_THRESHOLD: Final = 0.99
FEATURE_DIMENSIONS: Final = 2


@dataclass(frozen=True, slots=True)
class InterventionExtraction:
    """Opt-in extraction settings, leaving ordinary ExtractionConfig unchanged."""

    intervention: OutgoingIntervention
    extraction: ExtractionConfig


@torch.no_grad()
def extract_features(
    model: ConnectomeLM,
    tokens: IntVector,
    split: StorySplit,
    indices: tuple[int, ...],
    *,
    config: ExtractionConfig | InterventionExtraction | None = None,
) -> FeatureCache:
    """Consume each input with its full story prefix; padding never enters the cache.

    Batch lanes are independent stories, reset together only at batch boundaries.
    Finished lanes may advance through padding but are never reused within a batch.
    Empty and singleton stories contribute no pairs. No model mode is changed.
    """
    options = config or ExtractionConfig()
    intervention = None
    if not isinstance(options, ExtractionConfig):
        intervention = options.intervention
        options = options.extraction
    chunk_size, story_batch_size = options.chunk_size, options.story_batch_size
    bounds = tuple(zip(split.offsets[:-1], split.offsets[1:], strict=True))
    if (
        not bounds
        or split.offsets[0] != 0
        or split.offsets[-1] != tokens.size
        or len(bounds) != len(split.sha256)
        or any(b < a for a, b in bounds)
    ):
        message = "Story boundaries must cover tokens in order"
        raise ValueError(message)
    if (
        tokens.ndim != 1
        or tokens.dtype != np.int64
        or bool(((tokens < 0) | (tokens >= model.config.alphabet_size)).any())
        or not indices
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= model.nodes
    ):
        message = "Expected int64 vocabulary tokens and unique valid neuron indices"
        raise ValueError(message)
    lengths = tuple(max(0, b - a - 1) for a, b in bounds)
    offsets = tuple(accumulate(lengths, initial=0))
    features = np.empty((offsets[-1], len(indices)), dtype=np.float32)
    labels = np.empty(offsets[-1], dtype=np.int64)
    for i, (a, b) in enumerate(bounds):
        labels[offsets[i] : offsets[i + 1]] = tokens[a + 1 : b]
    selected = torch.tensor(indices, dtype=torch.int64, device=model.weight.device)
    prepared = prepare_recurrence(model.weight, model.topology, need_reverse=False)
    for batch in range(0, len(bounds), story_batch_size):
        lanes = bounds[batch : batch + story_batch_size]
        state: torch.Tensor | None = None
        longest = max(lengths[batch : batch + len(lanes)])
        for start in range(0, longest, chunk_size):
            width = min(chunk_size, longest - start)
            inputs = np.zeros((len(lanes), width), dtype=np.int64)
            for lane, (a, _) in enumerate(lanes):
                count = max(0, min(width, lengths[batch + lane] - start))
                inputs[lane, :count] = tokens[a + start : a + start + count]
            recorded, state = model.selected_states(
                torch.tensor(inputs, device=model.weight.device),
                selected,
                state=state,
                prepared=prepared,
                intervention=intervention,
            )
            values = recorded.cpu().to(torch.float32).numpy()
            for lane in range(len(lanes)):
                count = max(0, min(width, lengths[batch + lane] - start))
                target = offsets[batch + lane] + start
                features[target : target + count] = values[lane, :count]
    features.setflags(write=False)
    labels.setflags(write=False)
    return FeatureCache(features, labels)


def _vector(tensor: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value.item()) for value in tensor.detach().cpu().unbind())


def _matrix(tensor: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(_vector(row) for row in tensor.detach().cpu().unbind())


def feature_stats(cache: FeatureCache) -> FeatureStats:
    """Return hashes and the published float32 population feature statistics."""
    if (
        cache.features.dtype != np.float32
        or cache.labels.dtype != np.int64
        or cache.labels.ndim != 1
        or cache.features.ndim != FEATURE_DIMENSIONS
        or cache.features.shape[0] != cache.labels.size
        or not cache.labels.size
        or not cache.features.shape[1]
        or bool((cache.labels < 0).any())
        or not bool(np.isfinite(cache.features).all())
    ):
        message = "Invalid finite float32 feature cache"
        raise ValueError(message)
    variance = cast(
        "NDArray[np.float32]",
        np.var(cache.features, axis=0),
    )
    return FeatureStats(
        feature_sha256=sha256(cache.features.tobytes()).hexdigest(),
        label_sha256=sha256(cache.labels.tobytes()).hexdigest(),
        variance=tuple(
            float(cast("np.float32", variance[index])) for index in range(variance.size)
        ),
        saturation_fraction=(
            int(np.count_nonzero(np.abs(cache.features) >= SATURATION_THRESHOLD))
            / cache.features.size
        ),
    )


@torch.no_grad()
def _metrics(
    cache: FeatureCache,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
) -> ProbeMetrics:
    weight, bias, mean, std = tensors
    total, correct = 0.0, 0
    for start in range(0, cache.labels.size, batch_size):
        features = torch.tensor(
            cache.features[start : start + batch_size],
            device=weight.device,
        )
        labels = torch.tensor(
            cache.labels[start : start + batch_size],
            device=weight.device,
        )
        logits = ((features - mean) / std) @ weight + bias
        total += float(cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(dim=1) == labels).sum().item())
    nll = total / cache.labels.size
    return ProbeMetrics(
        tokens=cache.labels.size,
        nll=nll,
        accuracy=correct / cache.labels.size,
        perplexity=exp(nll) if nll <= log(float_info.max) else None,
    )


def fit_probe(
    train: FeatureCache, valid: FeatureCache, test: FeatureCache, config: ProbeConfig
) -> ProbeResult:
    """Fit 97->vocabulary with private CPU sampling, no source model access.

    Population normalization and add-half unigram initialization use only train.
    AdamW has no weight decay; explicit L2 excludes bias. Update u in [0,U) uses
    lr * (1 + cos(pi*u/U))/2. Validation/test are evaluated only after all updates.
    """
    for cache in (train, valid, test):
        if (
            cache.features.shape != (cache.labels.size, PROBE_WIDTH)
            or cache.features.dtype != np.float32
            or cache.labels.dtype != np.int64
            or cache.labels.ndim != 1
            or not cache.labels.size
            or not bool(np.isfinite(cache.features).all())
        ):
            message = "Expected finite float32 Nx97 features and nonempty int64 labels"
            raise ValueError(message)
        if bool(((cache.labels < 0) | (cache.labels >= config.vocab_size)).any()):
            message = "Probe labels exceed vocabulary bounds"
            raise ValueError(message)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        message = "CUDA probe fitting requested but unavailable"
        raise ValueError(message)
    features = torch.tensor(train.features, device=device)
    labels = torch.tensor(train.labels, device=device)
    mean = features.mean(dim=0)
    std = features.std(dim=0, correction=0).clamp_min(config.std_floor)
    features = (features - mean) / std
    weight = torch.nn.Parameter(features.new_zeros((PROBE_WIDTH, config.vocab_size)))
    counts = torch.bincount(labels, minlength=config.vocab_size).to(torch.float32) + 0.5
    bias = torch.nn.Parameter((counts / counts.sum()).log())
    optimizer = torch.optim.AdamW(
        (weight, bias), lr=config.learning_rate, weight_decay=0, foreach=False
    )
    rng = torch.Generator(device="cpu").manual_seed(config.seed)
    trace: list[ProbeStep] = []
    for update in range(config.updates):
        span = config.learning_rate - config.min_learning_rate
        lr = (
            config.min_learning_rate
            + span * (1 + cos(pi * update / config.updates)) / 2
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        selected = torch.randint(
            labels.numel(), (config.batch_size,), generator=rng, device="cpu"
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        nll = cross_entropy(features[selected] @ weight + bias, labels[selected])
        torch.autograd.backward(nll + 0.5 * config.l2 * weight.square().sum())
        norm = torch.nn.utils.clip_grad_norm_(
            (weight, bias), config.gradient_clip, error_if_nonfinite=True
        )
        optimizer_step(optimizer)
        trace.append(
            ProbeStep(
                update=update + 1,
                nll=float(nll.detach().item()),
                gradient_norm=float(norm.item()),
                learning_rate=lr,
            )
        )
    tensors = weight, bias, mean, std
    return ProbeResult(
        config=config,
        parameters=ProbeParameters(
            weight=_matrix(weight),
            bias=_vector(bias),
            mean=_vector(mean),
            std=_vector(std),
        ),
        parameter_count=weight.numel() + bias.numel(),
        trace=tuple(trace),
        features=(feature_stats(train), feature_stats(valid), feature_stats(test)),
        valid=_metrics(valid, tensors, config.batch_size),
        test=_metrics(test, tensors, config.batch_size),
    )
