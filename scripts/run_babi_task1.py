"""Run a sealed Task 1 experiment: uv run python -m scripts.run_babi_task1."""

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Annotated, Final

import torch
import typer

from flyrl.babi_checkpoint import publish, timed_copy
from flyrl.babi_data import BabiError
from flyrl.babi_learning import BabiLearner
from flyrl.babi_pilot import Experiment
from flyrl.babi_pilot_schema import (
    Manifest,
    Memorization,
    OperationalFailure,
    Preflight,
    Progress,
    Report,
    examples,
)
from flyrl.babi_protocol import (
    MEMORY_LIMIT,
    OPTIMIZATION_LIMIT,
    PANEL_SIZE,
    PROJECTION_LIMIT,
    WARMUPS,
    Allocation,
    BabiProtocol,
    synchronized_time,
    timing_fixture,
)
from flyrl.babi_recovery import record_event

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def preflight(experiment: Experiment, allocation: Allocation) -> Preflight:
    """Discard separate memorization/performance learners before creating main state."""
    protocol = experiment.protocol
    quality = protocol.profile == "quality"
    if quality:
        torch.cuda.reset_peak_memory_stats()
    panel = experiment.memorization_panel() if quality else experiment.corpus.train
    learner = BabiLearner(experiment.graph, protocol.config)
    diagnostic_updates = 400 if quality else 1
    for _ in range(diagnostic_updates):
        allocation.require_time()
        _ = learner.update(examples(panel.questions), learning_rate=0.003)
    evidence = experiment.evaluate(learner, panel)
    publish(
        experiment.output / "memorization.json",
        Memorization(
            protocol_sha256=protocol.sha256,
            trace=tuple(learner.trace),
            evidence=evidence,
        ),
    )
    correct = evidence.correct
    del learner
    if quality and correct != PANEL_SIZE:
        raise BabiError(reason=f"SANITY_FAILED: memorization {correct}/12")
    learner = BabiLearner(experiment.graph, protocol.config)
    train = examples(experiment.corpus.train.questions)
    timings: list[float] = []
    for update in range(1, 26):
        allocation.require_time()
        started = synchronized_time(quality)
        _ = learner.update(train, learning_rate=protocol.schedule.rate(update))
        elapsed = synchronized_time(quality) - started
        if update > WARMUPS:
            timings.append(elapsed)
    started = synchronized_time(quality)
    timing_fixture(learner, examples(experiment.corpus.valid.questions))
    evaluation = synchronized_time(quality) - started
    with TemporaryDirectory(dir=experiment.output) as temporary:
        checkpoint = Path(temporary) / "disposable.npz"
        state = Progress(
            protocol_sha256=protocol.sha256,
            updates=learner.updates,
            next_lr=protocol.schedule.rate(26),
            trace=tuple(learner.trace),
            exposures=tuple(sorted(learner.exposure_counts.items())),
            validation=(),
        )
        started = synchronized_time(quality)
        experiment.checkpoints.write_payload(learner, checkpoint, state)
        checkpoint_seconds = synchronized_time(quality) - started
        recovery_seconds = timed_copy(checkpoint, allocation.recovery)
    peak = torch.cuda.max_memory_allocated() if quality else 0
    del learner
    elapsed = allocation.elapsed
    projection = (
        elapsed
        + 1.3 * (12400 * mean(timings) + 9 * evaluation + 48 * checkpoint_seconds)
        + 600
    )
    passed = peak < MEMORY_LIMIT and projection <= PROJECTION_LIMIT
    record = Preflight(
        protocol_sha256=protocol.sha256,
        memorization_ids=tuple(e.example_id for e in examples(panel.questions)),
        memorization_updates=diagnostic_updates,
        memorization_correct=correct,
        warmups=5,
        timed_updates=tuple(timings),
        evaluation_seconds=evaluation,
        checkpoint_seconds=checkpoint_seconds,
        recovery_seconds=recovery_seconds,
        peak_bytes=peak,
        elapsed_seconds=elapsed,
        projected_seconds=projection,
        passed=passed,
    )
    publish(experiment.output / "preflight.json", record)
    if quality and not passed:
        raise BabiError(reason="INCOMPLETE_BUDGET_OR_INTERRUPTION: preflight budget")
    return record


@dataclass(frozen=True, slots=True)
class Launch:
    """Process launch identity, independent of the frozen training protocol."""

    resume: bool
    recovery: Path | None
    started: float

    def initialize(self, experiment: Experiment) -> tuple[Allocation, Preflight]:
        """Restore the original allocation clock or create disposable gate evidence."""
        frozen, output, recovery = experiment.protocol, experiment.output, self.recovery
        if self.resume:
            stored = BabiProtocol.model_validate_json(
                (output / "protocol.json").read_bytes()
            )
            allocation = Allocation.model_validate_json(
                (output / "allocation.json").read_bytes()
            )
            gates = Preflight.model_validate_json(
                (output / "preflight.json").read_bytes()
            )
            if (
                stored != frozen
                or gates.protocol_sha256 != frozen.sha256
                or (recovery is not None and recovery != allocation.recovery)
                or (frozen.profile == "quality" and not gates.passed)
            ):
                raise BabiError(reason="Resume protocol/preflight identity mismatch")
            allocation.require_time()
        else:
            if frozen.profile == "quality" and recovery is None:
                raise BabiError(
                    reason="Quality requires external verified recovery path"
                )
            destination = recovery or output.parent / (output.name + "-recovery")
            if destination.resolve().is_relative_to(output.resolve()):
                raise BabiError(reason="Recovery must be outside output")
            output.mkdir(parents=True)
            destination.mkdir(parents=True, exist_ok=True)
            allocation = Allocation(
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                started=self.started,
                recovery=destination,
            )
            publish(output / "protocol.json", frozen)
            record_event(output, frozen, "settings_frozen")
            publish(output / "allocation.json", allocation)
            gates = preflight(experiment, allocation)
        return allocation, gates


def run(
    protocol: BabiProtocol,
    output: Path,
    *,
    resume: bool = False,
    stop_after: int | None = None,
    recovery: Path | None = None,
) -> Report | Manifest:
    """Execute only frozen settings; explicit pauses are operational interruptions."""
    started = monotonic()
    frozen = BabiProtocol.model_validate_json(protocol.model_dump_json())
    stop = frozen.updates if stop_after is None else stop_after
    lower = (
        Manifest.model_validate_json(
            (output / "latest.json").read_bytes()
        ).progress.updates
        if resume
        else 0
    )
    if not lower <= stop <= frozen.updates:
        raise BabiError(reason="Invalid stop boundary")
    if not resume and output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    experiment = Experiment.open(frozen, output)
    allocation, gates = Launch(resume, recovery, started).initialize(experiment)
    learner = BabiLearner(experiment.graph, frozen.config)
    store = experiment.checkpoints
    if resume:
        manifest = store.load(learner)
        history = manifest.progress.validation
    else:
        history = (experiment.validate(learner),)
        manifest = store.save(learner, history)
    train = examples(experiment.corpus.train.questions)
    while learner.updates < stop:
        allocation.require_time(
            gates.evaluation_seconds + gates.checkpoint_seconds + 600
        )
        if allocation.elapsed >= OPTIMIZATION_LIMIT:
            _ = store.save(learner, history)
            _ = store.recover(allocation.recovery)
            raise BabiError(
                reason="INCOMPLETE_BUDGET_OR_INTERRUPTION: optimization deadline"
            )
        _ = learner.update(
            train, learning_rate=frozen.schedule.rate(learner.updates + 1)
        )
        if learner.updates in frozen.milestones:
            history = (*history, experiment.validate(learner))
        if (
            learner.updates % frozen.checkpoint_interval == 0
            or learner.updates in frozen.milestones
            or learner.updates == stop
        ):
            manifest = store.save(learner, history)
        if learner.updates % frozen.recovery_interval == 0:
            _ = store.recover(allocation.recovery)
    if stop != frozen.updates:
        return manifest
    allocation.require_time(4 * gates.evaluation_seconds + 600)
    report = experiment.finish(learner, history)
    _ = store.recover(allocation.recovery / "completed")
    allocation.require_time()
    return report


@app.command()
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """Run without command-line model, selection, decoding or budget overrides."""

    protocol: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    output: Annotated[Path, typer.Option(file_okay=False)]
    resume: Annotated[bool, typer.Option()] = False
    stop_after: Annotated[int | None, typer.Option(min=0)] = None
    recovery: Annotated[Path | None, typer.Option(file_okay=False)] = None

    def __post_init__(self) -> None:
        """Parse the sealed boundary and execute the same path exercised by tests."""
        try:
            protocol = BabiProtocol.model_validate_json(self.protocol.read_bytes())
            result = run(
                protocol,
                self.output,
                resume=self.resume,
                stop_after=self.stop_after,
                recovery=self.recovery,
            )
            typer.echo(result.model_dump_json())
        except (BabiError, RuntimeError, OSError, KeyboardInterrupt) as error:
            failure = OperationalFailure.from_error(error)
            if self.output.exists():
                publish(self.output / "failure.json", failure)
            typer.echo(failure.model_dump_json())
            raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
