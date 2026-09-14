"""Independent Task 1 selection and benchmark arithmetic from primary evidence."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Final, Literal

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from pydantic import Field

from flyrl.ar_framework import optimizer_tensors
from flyrl.babi_checkpoint import publish
from flyrl.babi_data import Question, Record
from flyrl.babi_pilot_schema import (
    Baselines,
    Decision,
    Evaluation,
    Manifest,
    PublishedPrediction,
    Report,
    Selection,
)
from flyrl.babi_protocol import BabiProtocol, Runtime
from flyrl.babi_recovery_validation import (
    EvaluationContext,
    Totals,
    checkpoint,
    evaluation,
    read_model,
    require,
    validation,
)
from scripts.connectome_source import file_digest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flyrl.babi_learning import BabiLearner
    from flyrl.babi_storage import Split, TrainValid
    from flyrl.bpe_tokenizer import BPETokenizer

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


@dataclass(frozen=True, slots=True)
class Inputs:
    """Authenticated prepared corpus and unchanged lexical decoder."""

    corpus: TrainValid
    test: Split
    tokenizer: BPETokenizer

    def decode(self, ids: tuple[int, ...]) -> str:
        """Decode without stripping, folding, or hiding generated special tokens."""
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)


def fitted_baselines(
    train: tuple[Question, ...], heldout: tuple[Question, ...]
) -> Baselines:
    """Independently fit answer/person counts on retained training questions only."""
    counts = Counter(q.answer for q in train)
    common = min(counts, key=lambda answer: (-counts[answer], answer))
    by_person = {
        person: Counter(q.answer for q in train if q.queried_person == person)
        for person in {q.queried_person for q in train}
    }
    priors = {
        person: min(histogram, key=lambda answer: (-histogram[answer], answer))
        for person, histogram in by_person.items()
    }
    return Baselines(
        questions=len(heldout),
        majority=sum(q.answer == common for q in heldout),
        person_prior=sum(
            q.answer == priors.get(q.queried_person, common) for q in heldout
        ),
        final_fact=sum(q.answer == q.facts[-1].location for q in heldout),
        oracle=sum(
            q.answer
            == tuple(f.location for f in q.facts if f.person == q.queried_person)[-1]
            for q in heldout
        ),
    )


def score_band(correct: int) -> str:
    """Apply the frozen integer thresholds, not a rounded accuracy."""
    return next(
        (
            band
            for minimum, band in ((950, "solved"), (800, "meaningful"), (500, "weak"))
            if correct >= minimum
        ),
        "insufficient",
    )


class RecoveryReport(Record):
    """Locally recomputed claims; remote evidence never becomes local CUDA replay."""

    protocol_sha256: str
    selected_update: int
    selected_checkpoint_sha256: str
    ordinary: Totals
    removed: Totals
    train_novel_correct: int
    train_novel_questions: int
    baseline_valid: Baselines
    baseline_test: Baselines
    baseline_maximum: float
    ordinary_accuracy: float
    removed_accuracy: float
    input_evidence_passed: bool
    raw_score_band: str
    final_label: str
    local_cuda_continuation_verified: Literal[False] = False
    topology_causality_demonstrated: Literal[False] = False


class Artifact(Record):
    """Exact bytes of a named regular result artifact."""

    path: str
    size: Annotated[int, Field(ge=0)]
    sha256: Digest


class ArtifactManifest(Record):
    """All result files except this manifest's own non-self-referential bytes."""

    protocol_sha256: Digest
    artifacts: tuple[Artifact, ...]


Phase = Literal["settings_frozen", "selection_frozen", "test_opened", "completed"]
PHASES: Final[tuple[Phase, ...]] = (
    "settings_frozen",
    "selection_frozen",
    "test_opened",
    "completed",
)


class Event(Record):
    """One exclusive immutable event chained to its predecessor's physical bytes."""

    phase: Phase
    protocol_sha256: Digest
    artifact_sha256: Digest
    previous_sha256: Digest


class EventChain(Record):
    """Published ordered copies of the four separately immutable event records."""

    events: tuple[Event, ...]


class StateSnapshot(Record):
    """Parameter, Adam, private RNG and non-consuming continuation preview."""

    model_sha256: Digest
    adam_sha256: Digest
    rng_sha256: Digest
    next_ids: tuple[str, ...]
    next_lr: float | None


class Restoration(Record):
    """Same-runtime selected-state comparison plus post-restore generated evidence."""

    checkpoint_sha256: Digest
    runtime: Runtime
    before: StateSnapshot
    after: StateSnapshot
    ordinary: tuple[PublishedPrediction, ...]
    removed: tuple[PublishedPrediction, ...]


def retained_updates(protocol: BabiProtocol) -> tuple[int, ...]:
    """Derive milestone retention plus the latest two complete scheduled saves."""
    saved = sorted(
        set(range(0, protocol.updates + 1, protocol.checkpoint_interval))
        | set(protocol.milestones)
    )
    return tuple(sorted(set(protocol.milestones) | set(saved[-2:])))


def verify_artifacts(root: Path, protocol: BabiProtocol) -> None:
    """Require a complete safe namespace and independently hash every result file."""
    manifest = read_model(ArtifactManifest, root / "artifacts.json")
    require(
        manifest.protocol_sha256 == protocol.sha256,
        "Artifact protocol mismatch",
    )
    names = tuple(item.path for item in manifest.artifacts)
    require(
        names == tuple(sorted(set(names))),
        "Artifact paths must be unique and sorted",
    )
    for name in names:
        path = PurePosixPath(name)
        require(
            bool(name)
            and not path.is_absolute()
            and ".." not in path.parts
            and "\\" not in name
            and str(path) == name,
            "Artifact path is unsafe",
        )
    paths = tuple(root.rglob("*"))
    require(
        root.is_dir()
        and not root.is_symlink()
        and all(not p.is_symlink() for p in paths),
        "Artifact namespace contains a symlink",
    )
    actual = {
        path.relative_to(root).as_posix()
        for path in paths
        if path.is_file() and path.name != "artifacts.json"
    }
    expected = set(names)
    require(actual == expected, "Artifact membership mismatch")
    expected_directories = {
        str(parent)
        for name in names
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    actual_directories = {
        path.relative_to(root).as_posix() for path in paths if path.is_dir()
    }
    require(
        actual_directories == expected_directories,
        "Artifact directory membership mismatch",
    )
    for item in manifest.artifacts:
        path = root / item.path
        require(
            path.stat().st_size == item.size and file_digest(path) == item.sha256,
            "Artifact size or hash mismatch",
        )


def verify_events(root: Path, protocol: BabiProtocol) -> None:
    """Recompute the immutable selection-before-test event chain."""
    chain = read_model(EventChain, root / "events.json")
    require(
        tuple(event.phase for event in chain.events) == PHASES,
        "Event phase ordering mismatch",
    )
    for index, event in enumerate(chain.events):
        path = root / f"event-{index}.json"
        require(
            read_model(Event, path) == event
            and event.protocol_sha256 == protocol.sha256
            and event.previous_sha256
            == (file_digest(root / f"event-{index - 1}.json") if index else "0" * 64),
            "Event chain identity mismatch",
        )
        expected = {
            "settings_frozen": protocol.sha256,
            "selection_frozen": file_digest(root / "selection.json"),
            "test_opened": protocol.corpus.test_sha256,
            "completed": file_digest(root / "report.json"),
        }[event.phase]
        require(event.artifact_sha256 == expected, "Event artifact mismatch")


def checkpoint_snapshot(
    root: Path,
    protocol: BabiProtocol,
    selection: Selection,
    training_ids: tuple[str, ...],
) -> StateSnapshot:
    """Derive parameter, Adam, RNG and next-sample identity from selected bytes."""
    path = root / f"update-{selection.update:06d}" / "checkpoint.npz"
    with (
        path.open("rb") as stream,
        NpzFile[np.generic](stream, allow_pickle=False) as data,
    ):
        model = {
            name: torch.tensor(data[name])
            for name in data.files
            if name.startswith("model.")
        }
        adam = {
            name: torch.tensor(data[name])
            for name in data.files
            if name.startswith("adam.")
        }
        state = data["rng"]
    rng = torch.Generator(device="cpu").set_state(torch.tensor(state))
    return StateSnapshot(
        model_sha256=state_digest(model),
        adam_sha256=state_digest(adam),
        rng_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
        next_ids=tuple(
            training_ids[int(index)]
            for index in torch.randint(
                len(training_ids),
                (protocol.config.batch_size,),
                generator=rng,
            )
        ),
        next_lr=protocol.next_lr(selection.update),
    )


def verify_restoration(
    root: Path,
    protocol: BabiProtocol,
    selection: Selection,
    report: Report,
    training_ids: tuple[str, ...],
) -> None:
    """Bind same-runtime restoration and both 32-question output reproductions."""
    record = read_model(Restoration, root / "restoration.json")
    expected = checkpoint_snapshot(root, protocol, selection, training_ids)
    require(
        record.checkpoint_sha256 == selection.checkpoint_sha256
        and record.runtime == protocol.runtime
        and record.before == record.after == expected,
        "Restoration state mismatch",
    )
    require(
        record.ordinary == report.ordinary.predictions[:32]
        and record.removed == report.removed.predictions[:32],
        "Restoration output reproduction mismatch",
    )


def verify(root: Path, protocol: BabiProtocol, inputs: Inputs) -> RecoveryReport:
    """Recompute every milestone, retained transaction, test metric and decision."""
    verify_artifacts(root, protocol)
    verify_events(root, protocol)
    report = read_model(Report, root / "report.json")
    selection = read_model(Selection, root / "selection.json")
    require(
        read_model(BabiProtocol, root / "protocol.json") == protocol
        and report.profile == protocol.profile
        and selection == report.selection
        and selection.protocol_sha256 == protocol.sha256,
        "Protocol or selection identity mismatch",
    )
    train, valid, test = (
        inputs.corpus.train.questions,
        inputs.corpus.valid.questions,
        inputs.test.questions,
    )
    ids = tuple(f"{q.episode_id:04d}:{q.question_line:02d}" for q in train)
    history = tuple(
        validation(
            update,
            evaluation(
                read_model(Evaluation, root / f"validation-{update:06d}.json"),
                valid,
                EvaluationContext("0" * 64, removed=False, decode=inputs.decode),
            ),
        )
        for update in protocol.milestones
    )
    chosen = min((-v.correct, v.nll, v.update) for v in history if v.update)[2]
    require(
        selection.validation == history and selection.update == chosen,
        "Validation-only selection mismatch",
    )
    previous = ()
    selected_hash = ""
    latest = read_model(Manifest, root / "latest.json")
    for update in retained_updates(protocol):
        manifest = checkpoint(root / f"update-{update:06d}", protocol, ids)
        progress = manifest.progress
        require(
            progress.validation == tuple(v for v in history if v.update <= update)
            and progress.validation_sha256
            == tuple(
                file_digest(root / f"validation-{v.update:06d}.json")
                for v in history
                if v.update <= update
            )
            and progress.trace[: len(previous)] == previous,
            "Checkpoint history prefix mismatch",
        )
        previous = progress.trace
        if update == chosen:
            selected_hash = manifest.checkpoint_sha256
        if update == protocol.updates:
            require(manifest == latest, "Latest checkpoint mismatch")
    require(
        selection.checkpoint_sha256 == selected_hash,
        "Selected checkpoint hash mismatch",
    )
    verify_restoration(root, protocol, selection, report, ids)
    ordinary = evaluation(
        report.ordinary,
        test,
        EvaluationContext(selected_hash, removed=False, decode=inputs.decode),
    )
    removed = evaluation(
        report.removed,
        test,
        EvaluationContext(selected_hash, removed=True, decode=inputs.decode),
    )
    require(
        read_model(Evaluation, root / "ordinary.json") == report.ordinary
        and read_model(Evaluation, root / "removed.json") == report.removed,
        "Published evaluation mismatch",
    )
    train_prompts = {q.prompt_hash for q in train}
    novel = tuple(
        p
        for q, p in zip(test, report.ordinary.predictions, strict=True)
        if q.prompt_hash not in train_prompts
    )
    novel_correct = sum(
        p.terminated and inputs.decode(p.generated_ids[:-1]) == " " + p.gold_answer
        for p in novel
    )
    baseline_valid, baseline_test = (
        fitted_baselines(train, valid),
        fitted_baselines(train, test),
    )
    require(
        report.baseline_valid == baseline_valid
        and report.baseline_test == baseline_test
        and report.train_novel_correct == novel_correct
        and report.train_novel_questions == len(novel)
        and report.splits
        == (inputs.corpus.train.report, inputs.corpus.valid.report, inputs.test.report),
        "Baseline, train-novel or split summary mismatch",
    )
    require(
        protocol.profile != "quality"
        or (ordinary.questions, removed.questions, len(novel)) == (1000, 1000, 984),
        "Official test membership mismatch",
    )
    count = ordinary.questions
    baseline = max(
        Fraction(1, 6),
        Fraction(baseline_test.majority, count),
        Fraction(baseline_test.person_prior, count),
    )
    e, r = Fraction(ordinary.correct, count), Fraction(removed.correct, count)
    passed = e - r >= Fraction(1, 5) and r <= baseline + Fraction(1, 10)
    band = score_band(ordinary.correct)
    label = "READING_NOT_DEMONSTRATED" if not passed else band.upper()
    if protocol.profile == "smoke":
        band = label = "SMOKE_NOT_BENCHMARK_EVIDENCE"
    require(
        report.raw_score_band == band
        and report.input_evidence_passed == passed
        and read_model(Decision, root / "decision.json")
        == Decision(
            protocol_sha256=protocol.sha256,
            checkpoint_sha256=selected_hash,
            raw_score_band=band,
            input_evidence_passed=passed,
            benchmark_evidence=protocol.profile == "quality",
        ),
        "Final decision mismatch",
    )
    return RecoveryReport(
        protocol_sha256=protocol.sha256,
        selected_update=chosen,
        selected_checkpoint_sha256=selected_hash,
        ordinary=ordinary,
        removed=removed,
        train_novel_correct=novel_correct,
        train_novel_questions=len(novel),
        baseline_valid=baseline_valid,
        baseline_test=baseline_test,
        baseline_maximum=float(baseline),
        ordinary_accuracy=float(e),
        removed_accuracy=float(r),
        input_evidence_passed=passed,
        raw_score_band=band,
        final_label=label,
    )


def state_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    """Hash sorted fully qualified tensor names followed by contiguous CPU bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def capture_state(
    learner: BabiLearner, training_ids: tuple[str, ...], next_lr: float | None
) -> StateSnapshot:
    """Capture a restoration comparator without consuming the learner's sampler."""
    rng = torch.Generator(device="cpu").set_state(learner.window_rng.get_state())
    states = optimizer_tensors(learner.optimizer)
    model = {
        f"model.{name}": parameter
        for name, parameter in learner.model.named_parameters()
    }
    adam = {
        f"adam.{name}.{key}": tensor
        for name, parameter in learner.model.named_parameters()
        if parameter in states
        for key, tensor in states[parameter].items()
    }
    return StateSnapshot(
        model_sha256=state_digest(model),
        adam_sha256=state_digest(adam),
        rng_sha256=hashlib.sha256(rng.get_state().numpy().tobytes()).hexdigest(),
        next_ids=tuple(
            training_ids[int(i)]
            for i in torch.randint(
                len(training_ids), (learner.config.batch_size,), generator=rng
            )
        ),
        next_lr=next_lr,
    )


def record_event(root: Path, protocol: BabiProtocol, phase: Phase) -> None:
    """Publish one ordered event; test-open cannot precede immutable selection."""
    index = PHASES.index(phase)
    previous = root / f"event-{index - 1}.json"
    require(index == 0 or previous.is_file(), "Missing predecessor event")
    hashes = {
        "settings_frozen": protocol.sha256,
        "test_opened": protocol.corpus.test_sha256,
    }
    artifact = (
        hashes[phase]
        if phase in hashes
        else file_digest(
            root / ("selection.json" if phase == "selection_frozen" else "report.json")
        )
    )
    entry = Event(
        phase=phase,
        protocol_sha256=protocol.sha256,
        artifact_sha256=artifact,
        previous_sha256=file_digest(previous) if index else "0" * 64,
    )
    publish(root / f"event-{index}.json", entry)
    if phase == "completed":
        publish(
            root / "events.json",
            EventChain(
                events=tuple(
                    read_model(Event, root / f"event-{i}.json")
                    for i in range(len(PHASES))
                )
            ),
        )


def seal_artifacts(root: Path, protocol: BabiProtocol) -> ArtifactManifest:
    """Publish a sorted size/hash inventory only after all evidence is complete."""
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p != root / "artifacts.json"
    )
    require(all(not p.is_symlink() for p in root.rglob("*")), "Unsafe artifact symlink")
    manifest = ArtifactManifest(
        protocol_sha256=protocol.sha256,
        artifacts=tuple(
            Artifact(
                path=p.relative_to(root).as_posix(),
                size=p.stat().st_size,
                sha256=file_digest(p),
            )
            for p in files
        ),
    )
    publish(root / "artifacts.json", manifest)
    return manifest
