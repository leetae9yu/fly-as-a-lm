"""Portable checkpoint and additive report verification for local quality recovery."""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from pydantic import Field, TypeAdapter

from flyrl.ar_checkpoint import Metadata
from flyrl.ar_config import TraceEntry, parameter_identity_json
from flyrl.ar_learning import ARLearner
from flyrl.language_models import Settings
from flyrl.quality_generation import has_repeated_word_block
from flyrl.quality_pilot import MAXIMUM_COLLAPSE, MINIMUM_GAIN
from flyrl.quality_pilot_schema import (
    CheckpointResult,
    Evaluation,
    MetricSummary,
    QualityProgress,
    SuccessLabel,
)
from flyrl.quality_recovery_validation import read_model, require, validate_result
from flyrl.story_data import StoryCorpus, StorySplit
from scripts.connectome_source import file_digest

if TYPE_CHECKING:
    from numpy import generic

    from flyrl.quality_metrics import QualityMetrics


class Runtime(Settings):
    """Portable execution identity; validating this does not attest continuation."""

    torch: Annotated[str, Field(min_length=1)]
    device: str
    name: Annotated[str, Field(min_length=1)]
    threads: int
    deterministic: bool
    tf32: bool


class RecoveryReport(QualityProgress):
    """Recomputed evidence with explicitly bounded local verification claims."""

    nll_gain: float
    generation_reproduction: Literal["verified locally (CPU)", "remote-worker gate"]
    checkpoint_continuation_verified: Literal[False] = False


def decide(profile: str, gates: bool, gain: float, repeats: int) -> SuccessLabel:
    """Recompute the frozen decision, distinguishing smoke from quality evidence."""
    if (
        profile == "quality"
        and gates
        and gain >= MINIMUM_GAIN
        and repeats <= MAXIMUM_COLLAPSE
    ):
        return SuccessLabel.IMPROVED
    return SuccessLabel.UNRELIABLE if gain > 0 else SuccessLabel.NOT_DEMONSTRATED


def checkpoint(
    path: Path, report: QualityProgress, learner: ARLearner, update: int
) -> tuple[Metadata, str]:
    """Inspect portable arrays without falsifying loader runtime identity."""
    protocol = report.protocol
    with path.open("rb") as stream:
        data: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with data:
            require(
                data["metadata"].shape == () and data["metadata"].dtype.kind == "U",
                "Checkpoint metadata array mismatch",
            )
            meta = Metadata.model_validate_json(
                TypeAdapter(str).validate_python(data["metadata"].item(), strict=True),
                strict=True,
            )
            require(
                meta.config == protocol.config
                and meta.graph == protocol.graph.fingerprint
                and meta.corpus == protocol.corpus.fingerprint
                and meta.runtime == report.runtime
                and meta.updates == update,
                "Checkpoint metadata identity mismatch",
            )
            require(
                tuple(step.update for step in meta.trace) == tuple(range(1, update + 1))
                and all(
                    step.nll >= 0 and step.gradient_norm >= 0 for step in meta.trace
                ),
                "Checkpoint progress mismatch",
            )
            expected = {"metadata", "rng"}
            rng = data["rng"]
            require(
                rng.dtype == np.uint8
                and rng.shape == tuple(learner.window_rng.get_state().shape),
                "Checkpoint RNG mismatch",
            )
            _ = torch.Generator().set_state(torch.tensor(rng))
            digest = hashlib.sha256(parameter_identity_json(protocol.config).encode())
            for name, parameter in learner.model.named_parameters():
                keys = [f"model.{name}"]
                if parameter.requires_grad and update:
                    keys.extend(
                        f"adam.{name}.{key}"
                        for key in ("step", "exp_avg", "exp_avg_sq")
                    )
                expected.update(keys)
                for key in keys:
                    array = data[key]
                    require(
                        array.dtype == np.float32
                        and array.shape
                        == (() if key.endswith(".step") else tuple(parameter.shape))
                        and bool(np.isfinite(array).all()),
                        "Checkpoint tensor shape/dtype/finite mismatch",
                    )
                    require(
                        not key.endswith(".step") or array.item() == update,
                        "Checkpoint Adam step mismatch",
                    )
                    require(
                        not key.endswith(".exp_avg_sq")
                        or bool(np.greater_equal(array, 0).all()),
                        "Checkpoint negative Adam variance",
                    )
                array = data[f"model.{name}"]
                digest.update(name.encode())
                digest.update(array.tobytes())
                with torch.no_grad():
                    _ = parameter.copy_(torch.tensor(array))
            require(
                set(data.files) == expected and len(data.files) == len(expected),
                "Checkpoint array membership mismatch",
            )
    return meta, digest.hexdigest()


def summaries(
    report: QualityProgress, stories: StoryCorpus
) -> tuple[MetricSummary, ...]:
    """Recompute additive aggregates, checking every ordered story hash and count."""
    result: list[MetricSummary] = []
    splits: tuple[
        tuple[
            Literal["valid", "test"], tuple[tuple[int, QualityMetrics], ...], StorySplit
        ],
        ...,
    ] = (
        (
            "valid",
            tuple((v.update, v.metrics) for v in report.validation),
            stories.metadata.valid,
        ),
        (
            "test",
            tuple((r.update, r.test) for r in report.completed),
            stories.metadata.test,
        ),
    )
    for split, evidence, boundaries in splits:
        for update, metrics in evidence:
            require(
                tuple(s.sha256 for s in metrics.per_story) == boundaries.sha256
                and tuple(s.count for s in metrics.per_story)
                == tuple(
                    b - a - 1
                    for a, b in zip(
                        boundaries.offsets[:-1], boundaries.offsets[1:], strict=True
                    )
                ),
                "Metric story identity/count mismatch",
            )
            result.append(
                MetricSummary(
                    update=update,
                    split=split,
                    loss_sum=metrics.loss_sum,
                    count=metrics.count,
                    correct=metrics.correct,
                    nll=metrics.nll,
                    perplexity=metrics.perplexity,
                    accuracy=metrics.accuracy,
                )
            )
    return tuple(result)


def verify(
    root: Path, report: QualityProgress, learner: ARLearner, stories: StoryCorpus
) -> RecoveryReport:
    """Verify every milestone and recompute the final local recovery decision."""
    protocol = report.protocol
    runtime = Runtime.model_validate_json(report.runtime, strict=True)
    device = torch.device(protocol.config.device)
    require(
        runtime.device == ("cuda:0" if str(device) == "cuda" else str(device))
        and runtime.threads == protocol.cpu_threads
        and (runtime.name == "CPU" if device.type == "cpu" else bool(runtime.name))
        and (protocol.profile != "quality" or "T4" in runtime.name),
        "Runtime identity mismatch",
    )
    previous: tuple[TraceEntry, ...] = ()
    for item in report.validation:
        path = root / f"checkpoint-{item.update:06d}.npz"
        require(
            file_digest(path) == item.checkpoint_sha256
            and read_model(Evaluation, root / f"evaluation-{item.update:06d}.json")
            == item,
            "Evaluation/checkpoint hash mismatch",
        )
        meta, fingerprint = checkpoint(path, report, learner, item.update)
        require(
            meta.trace[: len(previous)] == previous, "Checkpoint trace history mismatch"
        )
        previous = meta.trace
        for result in report.completed:
            if result.update != item.update:
                continue
            require(
                result.checkpoint_sha256 == item.checkpoint_sha256
                and read_model(
                    CheckpointResult, root / f"result-{item.update:06d}.json"
                )
                == result,
                "Result/checkpoint identity mismatch",
            )
            validate_result(root, result, report, (learner, stories), fingerprint)
    require(report.summaries == summaries(report, stories), "Additive summary mismatch")
    chosen = next(r for r in report.completed if r.update == report.selected_update)
    early = next(r for r in report.completed if r.update == protocol.early_update)
    gain = (
        early.test.loss_sum / early.test.count
        - chosen.test.loss_sum / chosen.test.count
    )
    repeats = sum(
        has_repeated_word_block(stories.corpus.decode(list(r.token_ids)))
        for r in chosen.generation
        if r.mode == "sampled"
    )
    require(
        report.repeat_collapse_count == repeats
        and report.success_label
        == decide(protocol.profile, report.recovery_gates_complete, gain, repeats),
        "Repeat collapse or success decision mismatch",
    )
    return RecoveryReport.model_validate(
        {
            **report.model_dump(),
            "recovery_gates_complete": True,
            "success_label": decide(
                protocol.profile, gates=True, gain=gain, repeats=repeats
            ),
            "nll_gain": gain,
            "generation_reproduction": "verified locally (CPU)"
            if device.type == "cpu"
            else "remote-worker gate",
        }
    )
