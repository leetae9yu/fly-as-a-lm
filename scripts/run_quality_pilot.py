"""Execute a frozen quality protocol: python -m scripts.run_quality_pilot."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import torch
import typer

from flyrl.ar_checkpoint import load_checkpoint, runtime_identity
from flyrl.ar_framework import optimizer_tensors
from flyrl.ar_learning import ARLearner
from flyrl.connectome import load_graph
from flyrl.language_runtime import graph_fingerprint
from flyrl.quality_pilot import evaluate_validation, exclusive, finish, persist_progress
from flyrl.quality_pilot_schema import (
    CheckpointResult,
    Evaluation,
    QualityProgress,
    QualityProtocol,
    select_update,
)
from flyrl.story_data import StoryCorpus, load_story_corpus
from scripts.connectome_source import file_digest

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _inputs(protocol: QualityProtocol) -> tuple[ARLearner, StoryCorpus]:
    for artifact in (protocol.graph, protocol.corpus):
        if file_digest(artifact.path) != artifact.sha256:
            message = "Protocol artifact file identity mismatch"
            raise ValueError(message)
    graph, stories = (
        load_graph(protocol.graph.path),
        load_story_corpus(protocol.corpus.path),
    )
    corpus = protocol.corpus
    if (
        graph_fingerprint(graph) != protocol.graph.fingerprint
        or (len(graph.node_ids), graph.source.size)
        != (protocol.graph.nodes, protocol.graph.edges)
        or stories.corpus.fingerprint != corpus.fingerprint
        or len(stories.corpus.vocabulary) != corpus.vocabulary
        or tuple(len(split.sha256) for split in stories.metadata.splits)
        != corpus.stories
        or tuple(int(tokens.size) for tokens in stories.streams) != corpus.tokens
        or any(not stories.corpus.encode(prompt) for prompt in protocol.prompts)
    ):
        message = "Protocol graph/corpus/count/prompt identity mismatch"
        raise ValueError(message)
    _ = stories.starts(stories.metadata.train, protocol.config.context)
    torch.set_num_threads(protocol.cpu_threads)
    learner = ARLearner(graph, protocol.config)
    if protocol.profile == "quality" and "T4" not in torch.cuda.get_device_name(
        learner.model.weight.device
    ):
        message = "Quality protocol requires an actual Tesla T4"
        raise ValueError(message)
    return learner, stories


def resume_progress(
    protocol: QualityProtocol, output: Path, learner: ARLearner
) -> QualityProgress:
    """Require matching checkpoint, runtime and immutable evaluation history."""
    stored = QualityProtocol.model_validate_json(
        (output / "protocol.json").read_text(), strict=True
    )
    progress = QualityProgress.model_validate_json(
        (output / "progress.json").read_text(), strict=True
    )
    if (
        stored != protocol
        or progress.protocol != protocol
        or progress.protocol_sha256 != protocol.sha256
    ):
        message = "Resume protocol hash mismatch"
        raise ValueError(message)
    checkpoint = output / "checkpoint-latest.npz"
    if file_digest(checkpoint) != progress.checkpoint_sha256:
        message = "Resume checkpoint identity mismatch"
        raise ValueError(message)
    load_checkpoint(learner, checkpoint, protocol.corpus.fingerprint)
    expected_lr = (
        protocol.schedule.rate(learner.updates + 1)
        if learner.updates < protocol.schedule.target_updates
        else None
    )
    expected_updates = tuple(
        u for u in protocol.evaluation_updates if u <= learner.updates
    )
    if (
        learner.updates != progress.updates
        or learner.updates > protocol.schedule.target_updates
        or progress.runtime != runtime_identity(learner)
        or progress.next_lr != expected_lr
        or tuple(item.update for item in progress.validation) != expected_updates
        or (progress.completed and learner.updates != protocol.schedule.target_updates)
        or progress.selected_update
        != (
            select_update(progress.validation)
            if learner.updates == protocol.schedule.target_updates
            else None
        )
    ):
        message = "Resume progress/runtime/evaluation history mismatch"
        raise ValueError(message)
    for item in progress.validation:
        stored_evaluation = Evaluation.model_validate_json(
            (output / f"evaluation-{item.update:06d}.json").read_text(), strict=True
        )
        if (
            stored_evaluation != item
            or file_digest(output / f"checkpoint-{item.update:06d}.npz")
            != item.checkpoint_sha256
        ):
            message = "Resume evaluation/checkpoint history mismatch"
            raise ValueError(message)
    _verify_results(progress, output)
    # The loader restores values in a different mapping order than fresh AdamW.
    # Normalize serialization order too, so exact continuation retains file hashes.
    for parameter, state in optimizer_tensors(learner.optimizer).items():
        learner.optimizer.state[parameter] = {
            key: state[key] for key in ("step", "exp_avg", "exp_avg_sq")
        }
    return progress


def _verify_results(progress: QualityProgress, output: Path) -> None:
    protocol = progress.protocol
    expected_test = (
        sorted(
            {
                protocol.early_update,
                select_update(progress.validation),
                protocol.schedule.target_updates,
            }
        )
        if progress.selected_update is not None
        else []
    )
    if [result.update for result in progress.completed] != expected_test[
        : len(progress.completed)
    ]:
        message = "Resume test evaluation history mismatch"
        raise ValueError(message)
    identities = {item.update: item.checkpoint_sha256 for item in progress.validation}
    for result in progress.completed:
        stored_result = CheckpointResult.model_validate_json(
            (output / f"result-{result.update:06d}.json").read_text(), strict=True
        )
        if (
            result != stored_result
            or result.checkpoint_sha256 != identities[result.update]
        ):
            message = "Resume completed evaluation history mismatch"
            raise ValueError(message)
        for artifact in result.artifacts:
            if file_digest(output / artifact.path) != artifact.sha256:
                message = "Resume activation artifact identity mismatch"
                raise ValueError(message)


def run(
    protocol: QualityProtocol,
    output: Path,
    *,
    resume: bool = False,
    stop_after: int | None = None,
    recovery_gates_complete: bool = False,
) -> QualityProgress:
    """Run the fixed budget or pause at an explicit durable boundary.

    External hardware/recovery gates require explicit attestation; a CPU smoke or
    an unattested run cannot receive the improvement label. A torn checkpoint and
    progress pair fails closed instead of silently replaying completed evaluations.
    """
    protocol = QualityProtocol.model_validate_json(
        protocol.model_dump_json(), strict=True
    )
    target = protocol.schedule.target_updates
    stop = target if stop_after is None else stop_after
    if not 0 <= stop <= target or (
        stop != target
        and stop % protocol.checkpoint_interval
        and stop not in protocol.evaluation_updates
    ):
        message = "stop_after must be a declared checkpoint/evaluation boundary"
        raise ValueError(message)
    if not resume and output.exists():
        raise FileExistsError(output)
    learner, stories = _inputs(protocol)
    if resume:
        progress = resume_progress(protocol, output, learner)
        if (
            progress.recovery_gates_complete != recovery_gates_complete
            or stop < learner.updates
        ):
            message = "Resume progress/recovery attestation mismatch"
            raise ValueError(message)
    else:
        output.mkdir(parents=True, exist_ok=False)
        exclusive(output / "protocol.json", protocol.model_dump_json(indent=2))
        progress = QualityProgress(
            protocol=protocol,
            protocol_sha256=protocol.sha256,
            runtime=runtime_identity(learner),
            updates=0,
            next_lr=protocol.schedule.rate(1),
            checkpoint_sha256="0" * 64,
            recovery_gates_complete=recovery_gates_complete,
        )
        progress = persist_progress(
            learner, output, evaluate_validation(learner, stories, output, progress)
        )
    starts = stories.starts(stories.metadata.train, protocol.config.context)
    while learner.updates < stop:
        for group in learner.optimizer.param_groups:
            group["lr"] = protocol.schedule.rate(learner.updates + 1)
        learner.train(stories.corpus.train, 1, starts=starts)
        if learner.updates in protocol.evaluation_updates:
            progress = evaluate_validation(learner, stories, output, progress)
        if (
            learner.updates % protocol.checkpoint_interval == 0
            or learner.updates in protocol.evaluation_updates
        ):
            progress = persist_progress(learner, output, progress)
    return finish(learner, stories, output, progress) if stop == target else progress


@app.command()
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """Train or resume without allowing command-line hyperparameter overrides."""

    protocol: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    output: Annotated[Path, typer.Option(file_okay=False)]
    resume: Annotated[bool, typer.Option()] = False
    stop_after: Annotated[int | None, typer.Option(min=0)] = None
    recovery_gates_complete: Annotated[bool, typer.Option()] = False

    def __post_init__(self) -> None:
        """Strictly parse all nested settings before executing the real learner."""
        protocol = QualityProtocol.model_validate_json(
            self.protocol.read_text(encoding="utf-8"), strict=True
        )
        _ = run(
            protocol,
            self.output,
            resume=self.resume,
            stop_after=self.stop_after,
            recovery_gates_complete=self.recovery_gates_complete,
        )


if __name__ == "__main__":
    app()
