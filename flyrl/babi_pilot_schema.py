"""Typed immutable transactions, selection records and independently scored evidence."""

from collections import Counter
from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, computed_field, model_validator

from flyrl.babi_data import BabiError, Question, Record
from flyrl.babi_learning import BabiExample, BabiUpdate
from flyrl.babi_metrics import AnswerNLL, Prediction
from flyrl.babi_protocol import BabiProtocol
from flyrl.babi_storage import SHA, SplitReport

Count = Annotated[int, Field(ge=0)]
Seconds = Annotated[float, Field(ge=0)]


def examples(
    questions: tuple[Question, ...], *, removed: bool = False
) -> tuple[BabiExample, ...]:
    """Adapt immutable source questions to the learner's token-only interface."""
    return tuple(
        BabiExample(
            example_id=f"{q.episode_id:04d}:{q.question_line:02d}",
            prompt_ids=q.removed_prompt_ids if removed else q.prompt_ids,
            answer_ids=q.answer_ids,
        )
        for q in questions
    )


class Validation(Record):
    """Selection inputs from the entire validation split, never official test."""

    update: Count
    correct: Count
    questions: Annotated[int, Field(gt=0)]
    nll: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def counts(self) -> Self:
        """Reject impossible correct-answer counts."""
        if self.correct > self.questions:
            raise BabiError(reason="Validation count exceeds questions")
        return self


def select_update(history: tuple[Validation, ...]) -> int:
    """Choose exact count, then question-mean NLL, then earliest nonzero update."""
    candidates = tuple((-v.correct, v.nll, v.update) for v in history if v.update)
    if not candidates:
        raise BabiError(reason="Selection requires nonzero validation")
    return min(candidates)[2]


class Selection(Record):
    """Exclusive publication of the decision before the official test is opened."""

    protocol_sha256: SHA
    update: Count
    checkpoint_sha256: SHA
    validation: tuple[Validation, ...]


class Progress(Record):
    """All mutable learner progress captured inside one immutable transaction."""

    protocol_sha256: SHA
    updates: Count
    next_lr: float | None
    trace: tuple[BabiUpdate, ...]
    exposures: tuple[tuple[str, Count], ...]
    validation: tuple[Validation, ...]
    validation_sha256: tuple[SHA, ...] = ()

    def verify(self, protocol: BabiProtocol, ids: tuple[str, ...]) -> None:
        """Reject every progress disagreement before restoring any learner tensor."""
        expected = tuple(u for u in protocol.milestones if u <= self.updates)
        counts = Counter(identity for row in self.trace for identity in row.sampled_ids)
        if (
            self.protocol_sha256 != protocol.sha256
            or self.updates > protocol.updates
            or self.next_lr != protocol.next_lr(self.updates)
            or (
                protocol.profile == "quality"
                and len(self.validation_sha256) != len(self.validation)
            )
            or tuple(row.update for row in self.trace)
            != tuple(range(1, self.updates + 1))
            or tuple(v.update for v in self.validation) != expected
            or any(v.questions != protocol.corpus.counts[1] for v in self.validation)
            or self.exposures != tuple((identity, counts[identity]) for identity in ids)
            or any(identity not in ids for identity in counts)
            or any(
                len(row.sampled_ids) != protocol.config.batch_size
                or row.learning_rate != protocol.schedule.rate(row.update)
                or not isfinite(row.nll)
                or not isfinite(row.gradient_norm)
                for row in self.trace
            )
        ):
            raise BabiError(reason="Checkpoint progress identity mismatch")


class Manifest(Record):
    """Checkpoint bytes plus complete progress; NPZ repeats this progress binding."""

    schema_version: Literal[1] = 1
    checkpoint_sha256: SHA
    progress: Progress


class Preflight(Record):
    """Disposable gate evidence; timings never masquerade as benchmark accuracy."""

    protocol_sha256: SHA
    memorization_ids: tuple[str, ...]
    memorization_updates: Count
    memorization_correct: Count
    warmups: Count
    timed_updates: tuple[Seconds, ...]
    evaluation_seconds: Seconds
    checkpoint_seconds: Seconds
    recovery_seconds: Seconds
    peak_bytes: Count
    elapsed_seconds: Seconds
    projected_seconds: Seconds
    passed: bool


class PublishedPrediction(Prediction):
    """Source identity and removal evidence for every unmodified generated answer."""

    episode_id: Count
    question_line: Count
    removed_line_ids: tuple[int, ...]
    checkpoint_sha256: SHA

    exact: bool | None = None

    @model_validator(mode="after")
    def literal_exact(self) -> Self:
        """Reject a cached exact status disagreeing with literal decoded evidence."""
        if self.exact is not None and self.exact != self.correct:
            raise BabiError(reason="Prediction exact status mismatch")
        return self


class Evaluation(Record):
    """Full generated and teacher-forced evidence in source question order."""

    predictions: tuple[PublishedPrediction, ...]
    answer_nll: AnswerNLL

    @property
    def correct(self) -> int:
        """Count literal terminated answers without altering multiplicity."""
        return sum(p.correct for p in self.predictions)


class Memorization(Record):
    """Complete disposable trace and unrestricted predictions, never main state."""

    protocol_sha256: SHA
    trace: tuple[BabiUpdate, ...]
    evidence: Evaluation


class Decision(Record):
    """Machine-consumed score and evidence-use decision, separate from operations."""

    protocol_sha256: SHA
    checkpoint_sha256: SHA
    raw_score_band: str
    input_evidence_passed: bool
    benchmark_evidence: bool


class OperationalFailure(Record):
    """An interrupted/invalid operation is not a completed benchmark failure."""

    status: Literal["INVALID", "SANITY_FAILED", "INCOMPLETE_BUDGET_OR_INTERRUPTION"]
    reason: str

    @classmethod
    def from_error(
        cls, error: BabiError | RuntimeError | OSError | KeyboardInterrupt
    ) -> "OperationalFailure":
        """Map only operational exceptions to operational status codes."""
        reason = str(error)
        status = "SANITY_FAILED" if reason.startswith("SANITY_FAILED") else "INVALID"
        if reason.startswith("INCOMPLETE_BUDGET_OR_INTERRUPTION") or isinstance(
            error, KeyboardInterrupt
        ):
            status = "INCOMPLETE_BUDGET_OR_INTERRUPTION"
        return cls(status=status, reason=reason)


class Baselines(Record):
    """Train-fitted predictors evaluated without held-out fitting."""

    questions: int
    majority: int
    person_prior: int
    final_fact: int
    oracle: int
    uniform: float = 1 / 6


def baselines(train: tuple[Question, ...], heldout: tuple[Question, ...]) -> Baselines:
    """Fit lexical-tie-broken majority and person priors on retained train only."""

    def majority(questions: tuple[Question, ...]) -> str:
        counts = Counter(q.answer for q in questions)
        return min(counts, key=lambda answer: (-counts[answer], answer))

    common = majority(train)
    priors = {
        person: majority(tuple(q for q in train if q.queried_person == person))
        for person in {q.queried_person for q in train}
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
            == next(
                f.location for f in reversed(q.facts) if f.person == q.queried_person
            )
            for q in heldout
        ),
    )


class Report(Record):
    """Completed evidence; operational failure labels never describe a low score."""

    profile: Literal["quality", "smoke"]
    selection: Selection
    ordinary: Evaluation
    removed: Evaluation
    train_novel_correct: Count
    train_novel_questions: Count
    baseline_valid: Baselines
    baseline_test: Baselines
    splits: tuple[SplitReport, SplitReport, SplitReport]

    @property
    def decision(self) -> Decision:
        """Serialize the recomputed score and evidence-use result separately."""
        return Decision(
            protocol_sha256=self.selection.protocol_sha256,
            checkpoint_sha256=self.selection.checkpoint_sha256,
            raw_score_band=self.raw_score_band,
            input_evidence_passed=self.input_evidence_passed,
            benchmark_evidence=self.benchmark_evidence,
        )

    @computed_field
    @property
    def raw_score_band(self) -> str:
        """Use exact official counts; smoke is never benchmark evidence."""
        if self.profile == "smoke":
            return "SMOKE_NOT_BENCHMARK_EVIDENCE"
        correct = self.ordinary.correct
        for minimum, band in ((950, "solved"), (800, "meaningful"), (500, "weak")):
            if correct >= minimum:
                return band
        return "insufficient"

    @computed_field
    @property
    def input_evidence_passed(self) -> bool:
        """Evaluate both inequalities using rational counts, including exact 1/6."""
        count = len(self.ordinary.predictions)
        ordinary, removed = (
            Fraction(self.ordinary.correct, count),
            Fraction(self.removed.correct, count),
        )
        baseline = max(
            Fraction(1, 6),
            Fraction(self.baseline_test.majority, count),
            Fraction(self.baseline_test.person_prior, count),
        )
        return ordinary - removed >= Fraction(1, 5) and removed <= baseline + Fraction(
            1, 10
        )

    @computed_field
    @property
    def benchmark_evidence(self) -> bool:
        """Explicitly prevent CPU smoke publication as a quality result."""
        return self.profile == "quality"
