"""Resumable single-connectome quality training using the existing AdamW path."""

import json
import re
from pathlib import Path
from typing import Final, Literal

import typer
from matplotlib import rcParams
from pydantic import TypeAdapter

from flyrl.activations import ActivationOptions, export_activations
from flyrl.ar_checkpoint import load_checkpoint, save_checkpoint
from flyrl.ar_learning import ARLearner
from flyrl.language_checkpoint import atomic_text
from flyrl.quality_generation import generate_quality_panel, has_repeated_word_block
from flyrl.quality_metrics import QualityMetrics, evaluate_quality_split
from flyrl.quality_pilot_schema import (
    ArtifactFile,
    CheckpointResult,
    Evaluation,
    MetricSummary,
    QualityProgress,
    SuccessLabel,
    select_update,
)
from flyrl.story_data import StoryCorpus
from scripts.connectome_source import file_digest

MINIMUM_GAIN: Final = 0.50
MAXIMUM_COLLAPSE: Final = 3


def _status(event: str, update: int) -> None:
    typer.echo(json.dumps({"event": event, "update": update}))


def _checkpoint(output: Path, update: int) -> Path:
    return output / f"checkpoint-{update:06d}.npz"


def exclusive(path: Path, text: str) -> None:
    """Publish an immutable record without replacing an earlier artifact."""
    with path.open("x", encoding="utf-8") as stream:
        _ = stream.write(text)


def persist_progress(
    learner: ARLearner, output: Path, progress: QualityProgress
) -> QualityProgress:
    """Bind an atomic latest checkpoint to its progress and next scheduled LR."""
    checkpoint = output / "checkpoint-latest.npz"
    save_checkpoint(learner, checkpoint, progress.protocol.corpus.fingerprint)
    schedule = progress.protocol.schedule
    progress = progress.model_copy(
        update={
            "updates": learner.updates,
            "next_lr": schedule.rate(learner.updates + 1)
            if learner.updates < schedule.target_updates
            else None,
            "checkpoint_sha256": file_digest(checkpoint),
        }
    )
    atomic_text(output / "progress.json", progress.model_dump_json(indent=2))
    _status("checkpoint", learner.updates)
    return progress


def evaluate_validation(
    learner: ARLearner, stories: StoryCorpus, output: Path, progress: QualityProgress
) -> QualityProgress:
    """Publish one immutable milestone checkpoint and full validation evidence."""
    path = _checkpoint(output, learner.updates)
    if path.exists():
        raise FileExistsError(path)
    save_checkpoint(learner, path, stories.corpus.fingerprint)
    item = Evaluation(
        update=learner.updates,
        checkpoint_sha256=file_digest(path),
        metrics=evaluate_quality_split(
            learner, stories.corpus.valid, stories.metadata.valid
        ),
    )
    exclusive(
        output / f"evaluation-{learner.updates:06d}.json",
        item.model_dump_json(indent=2),
    )
    validation = (*progress.validation, item)
    progress = progress.model_copy(
        update={
            "validation": validation,
            "selected_update": select_update(validation)
            if learner.updates == progress.protocol.schedule.target_updates
            else None,
        }
    )
    _status("evaluation", learner.updates)
    return progress


def evaluate_test(
    learner: ARLearner,
    stories: StoryCorpus,
    output: Path,
    progress: QualityProgress,
    update: int,
) -> CheckpointResult:
    """Export deterministic panels and untouched test evidence for one checkpoint."""
    protocol = progress.protocol
    checkpoint = _checkpoint(output, update)
    load_checkpoint(learner, checkpoint, stories.corpus.fingerprint)
    draws = 3 if update == progress.selected_update else 1
    generation = generate_quality_panel(
        learner,
        stories.corpus,
        protocol.prompts,
        tuple(
            tuple(10_000 + 100 * i + j for j in range(draws))
            for i in range(len(protocol.prompts))
        ),
        protocol.generation,
    )
    previous_salt = TypeAdapter[str | None](str | None).validate_python(
        rcParams["svg.hashsalt"]
    )
    rcParams["svg.hashsalt"] = protocol.sha256
    try:
        activations = (
            tuple(
                export_activations(
                    learner,
                    stories.corpus,
                    protocol.prompts[index],
                    output / f"activations-{update:06d}" / f"prompt-{index:02d}",
                    ActivationOptions(length=protocol.generation.length),
                )
                .relative_to(output)
                .as_posix()
                for index in protocol.activation_prompt_indices
            )
            if update in {protocol.early_update, progress.selected_update}
            else ()
        )
    finally:
        rcParams["svg.hashsalt"] = previous_salt
    # Matplotlib SVG creation timestamps are presentation metadata, not evidence.
    for metadata in activations:
        for svg in (output / metadata).parent.glob("*.svg"):
            atomic_text(svg, re.sub(r"<dc:date>[^<]*</dc:date>", "", svg.read_text()))
    return CheckpointResult(
        update=update,
        checkpoint_sha256=file_digest(checkpoint),
        test=evaluate_quality_split(
            learner, stories.corpus.test, stories.metadata.test
        ),
        generation=generation,
        activations=activations,
        artifacts=tuple(
            ArtifactFile(
                path=path.relative_to(output).as_posix(), sha256=file_digest(path)
            )
            for metadata in activations
            for path in sorted((output / metadata).parent.iterdir())
            if path.is_file()
        ),
    )


def finish(
    learner: ARLearner, stories: StoryCorpus, output: Path, progress: QualityProgress
) -> QualityProgress:
    """Evaluate the deduplicated test set and publish every fixed draw and decision."""
    protocol, selected = progress.protocol, select_update(progress.validation)
    updates = sorted(
        {protocol.early_update, selected, protocol.schedule.target_updates}
    )
    for update in updates:
        if update in {item.update for item in progress.completed}:
            continue
        result = evaluate_test(learner, stories, output, progress, update)
        exclusive(
            output / f"result-{update:06d}.json", result.model_dump_json(indent=2)
        )
        progress = progress.model_copy(
            update={"completed": (*progress.completed, result)}
        )
        atomic_text(output / "progress.json", progress.model_dump_json(indent=2))
        _status("evaluation", update)
    chosen = next(item for item in progress.completed if item.update == selected)
    early = next(
        item for item in progress.completed if item.update == protocol.early_update
    )
    repeats = sum(
        has_repeated_word_block(stories.corpus.decode(list(record.token_ids)))
        for record in chosen.generation
        if record.mode == "sampled"
    )
    gain = (
        early.test.loss_sum / early.test.count
        - chosen.test.loss_sum / chosen.test.count
    )
    label = SuccessLabel.UNRELIABLE if gain > 0 else SuccessLabel.NOT_DEMONSTRATED
    if (
        protocol.profile == "quality"
        and progress.recovery_gates_complete
        and gain >= MINIMUM_GAIN
        and repeats <= MAXIMUM_COLLAPSE
    ):
        label = SuccessLabel.IMPROVED
    evidence: tuple[
        tuple[Literal["valid", "test"], tuple[tuple[int, QualityMetrics], ...]], ...
    ] = (
        ("valid", tuple((v.update, v.metrics) for v in progress.validation)),
        ("test", tuple((r.update, r.test) for r in progress.completed)),
    )
    summaries = tuple(
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
        for split, values in evidence
        for update, metrics in values
    )
    progress = progress.model_copy(
        update={
            "repeat_collapse_count": repeats,
            "success_label": label,
            "summaries": summaries,
        }
    )
    atomic_text(output / "progress.json", progress.model_dump_json(indent=2))
    atomic_text(output / "report.json", progress.model_dump_json(indent=2))
    _status("final", protocol.schedule.target_updates)
    return progress
