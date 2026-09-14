"""Literal recovery scoring, selection, baselines and publication boundary tests."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from flyrl.babi_data import Question, encode_question, load_tokenizer, parse_episodes
from flyrl.babi_metrics import AnswerNLL, QuestionNLL
from flyrl.babi_pilot_schema import Evaluation, PublishedPrediction, Report
from flyrl.babi_protocol import BabiProtocol
from flyrl.babi_recovery import (
    ArtifactManifest,
    EventChain,
    Inputs,
    RecoveryReport,
    Restoration,
    fitted_baselines,
    score_band,
    verify,
)
from flyrl.babi_recovery_validation import EvaluationContext, evaluation
from flyrl.babi_storage import Split, TrainValid
from scripts.recover_babi_task1 import app
from scripts.run_babi_task1 import run
from tests.test_babi_pilot import smoke_protocol


def evidence() -> tuple[tuple[Question, ...], Evaluation, EvaluationContext]:
    tokenizer = load_tokenizer(Path("data/tinystories_quality/tokenizer.json"))
    questions = tuple(
        encode_question(q, tokenizer)
        for episode in parse_episodes(
            b"".join(
                (
                    b"1 Mary moved to the hallway.\n2 John went to the garden.\n",
                    b"3 Where is Mary?\thallway\t1\n",
                    b"1 Mary went to the garden.\n2 John moved to the hallway.\n",
                    b"3 Where is Mary?\tgarden\t1\n",
                )
            )
        )
        for q in episode
    )
    predictions = tuple(
        PublishedPrediction(
            example_id=f"{q.episode_id:04d}:{q.question_line:02d}",
            generated_ids=q.answer_ids,
            terminated=True,
            decoded_answer=" " + q.answer,
            gold_answer=q.answer,
            episode_id=q.episode_id,
            question_line=q.question_line,
            removed_line_ids=(),
            checkpoint_sha256="a" * 64,
        )
        for q in questions
    )
    item = Evaluation(
        predictions=predictions,
        answer_nll=AnswerNLL(
            per_question=tuple(
                QuestionNLL(
                    example_id=p.example_id, loss_sum=loss, count=len(q.answer_ids)
                )
                for p, q, loss in zip(predictions, questions, (3.0, 6.0), strict=True)
            )
        ),
    )
    return (
        questions,
        item,
        EvaluationContext(
            "a" * 64,
            removed=False,
            decode=lambda ids: tokenizer.decode(list(ids), skip_special_tokens=False),
        ),
    )


def test_evaluation_when_literal_tokens_recomputes_additive_and_question_mean() -> None:
    # Given: a three-token hallway answer and a two-token garden answer.
    questions, item, context = evidence()
    # When: independently aggregate primary evidence.
    result = evaluation(item, questions, context)
    # Then: equal-question NLL is not token-weighted NLL.
    assert (
        result.correct,
        result.questions,
        result.loss_sum,
        result.tokens,
        result.nll,
    ) == (2, 2, 9.0, 5, 2.0)


@pytest.mark.parametrize(
    "field",
    [
        "ordering",
        "identity",
        "generated",
        "terminator",
        "text",
        "gold",
        "removal",
        "checkpoint",
        "denominator",
    ],
)
def test_evaluation_when_primary_evidence_corrupt_rejected(field: str) -> None:
    # Given: honest decoded evidence and one corrupted machine-consumed field.
    questions, item, context = evidence()
    changes = {
        "identity": {"example_id": "forged"},
        "generated": {"generated_ids": (851, 199)},
        "terminator": {"terminated": False},
        "text": {"decoded_answer": "hallway"},
        "gold": {"gold_answer": "garden"},
        "removal": {"removed_line_ids": (1,)},
        "checkpoint": {"checkpoint_sha256": "b" * 64},
    }
    if field in changes:
        corrupted = item.model_copy(
            update={
                "predictions": (
                    item.predictions[0].model_copy(update=changes[field]),
                    item.predictions[1],
                )
            }
        )
    elif field == "ordering":
        corrupted = item.model_copy(
            update={"predictions": tuple(reversed(item.predictions))}
        )
    else:
        corrupted = item.model_copy(
            update={
                "answer_nll": AnswerNLL(
                    per_question=(
                        item.answer_nll.per_question[0].model_copy(update={"count": 2}),
                        item.answer_nll.per_question[1],
                    )
                )
            }
        )
    # When/Then: no cached correctness can bypass primary-evidence checks.
    with pytest.raises(ValueError, match=r"mismatch|identity"):
        _ = evaluation(corrupted, questions, context)


def test_baselines_when_train_tie_are_train_fitted() -> None:
    # Given: two train answers tied; lexical garden wins and last facts are wrong.
    questions, _, _ = evidence()
    # When: fit only on those training questions.
    result = fitted_baselines(questions, questions)
    # Then: heldout majority is not substituted and person priors preserve ties.
    assert (result.majority, result.person_prior, result.final_fact, result.oracle) == (
        1,
        1,
        0,
        2,
    )


@pytest.mark.parametrize(
    ("correct", "band"),
    [
        (0, "insufficient"),
        (499, "insufficient"),
        (500, "weak"),
        (799, "weak"),
        (800, "meaningful"),
        (949, "meaningful"),
        (950, "solved"),
        (1000, "solved"),
    ],
)
def test_score_band_when_exact_count_boundary(correct: int, band: str) -> None:
    # Given/When/Then: fixed official count thresholds, never rounded ratios.
    assert score_band(correct) == band


@pytest.fixture
def completed(tmp_path: Path) -> tuple[Path, Inputs]:
    protocol = smoke_protocol(tmp_path)
    root = tmp_path / "run"
    _ = run(protocol, root)
    return root, Inputs(
        TrainValid.model_validate_json((tmp_path / "train_valid.json").read_bytes()),
        Split.model_validate_json((tmp_path / "test.json").read_bytes()),
        load_tokenizer(tmp_path / "tokenizer.json"),
    )


def test_verify_when_real_cpu_bundle_recomputes_without_changes(
    completed: tuple[Path, Inputs],
) -> None:
    # Given: the actual runner's completed CPU output, not a mocked learner.
    root, inputs = completed
    protocol = BabiProtocol.model_validate_json((root / "protocol.json").read_bytes())
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    # When: independent primary-evidence recovery.
    result = verify(root, protocol, inputs)
    # Then: evidence is recomputed but no source byte changes or CUDA claim appears.
    assert result.ordinary.questions == result.removed.questions == 2
    assert not result.local_cuda_continuation_verified
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("field", ["selection", "baseline", "aggregate", "split"])
def test_verify_when_cached_claim_is_corrupt_rejects(
    completed: tuple[Path, Inputs], field: str
) -> None:
    # Given: real primary evidence but a changed cached claim.
    root, inputs = completed
    protocol = BabiProtocol.model_validate_json((root / "protocol.json").read_bytes())
    report = Report.model_validate_json((root / "report.json").read_bytes())
    changes = {
        "selection": {"selection": report.selection.model_copy(update={"update": 0})},
        "baseline": {
            "baseline_test": report.baseline_test.model_copy(update={"majority": 999})
        },
        "aggregate": {"train_novel_correct": 999},
        "split": {"splits": tuple(reversed(report.splits))},
    }
    altered = report.model_copy(update=changes[field])
    _ = (root / "report.json").write_text(
        altered.model_dump_json(exclude_computed_fields=True)
    )
    if field == "selection":
        _ = (root / "selection.json").write_text(altered.selection.model_dump_json())
    # When/Then: independently recovered arithmetic or selection exposes the change.
    with pytest.raises(ValueError, match=r"mismatch"):
        _ = verify(root, protocol, inputs)


@pytest.mark.parametrize("sidecar", ["artifacts", "events", "restoration"])
def test_verify_when_recovery_sidecar_is_corrupt_rejects(
    completed: tuple[Path, Inputs], sidecar: str
) -> None:
    # Given: one independently parseable but false completed-bundle sidecar.
    root, inputs = completed
    protocol = BabiProtocol.model_validate_json((root / "protocol.json").read_bytes())
    if sidecar == "artifacts":
        path = root / "artifacts.json"
        record = ArtifactManifest.model_validate_json(path.read_bytes())
        changed = record.model_copy(
            update={
                "artifacts": (
                    record.artifacts[0].model_copy(
                        update={"size": record.artifacts[0].size + 1}
                    ),
                    *record.artifacts[1:],
                )
            }
        )
    elif sidecar == "events":
        path = root / "events.json"
        record = EventChain.model_validate_json(path.read_bytes())
        changed = record.model_copy(
            update={
                "events": (
                    record.events[0].model_copy(update={"artifact_sha256": "f" * 64}),
                    *record.events[1:],
                )
            }
        )
    else:
        path = root / "restoration.json"
        record = Restoration.model_validate_json(path.read_bytes())
        changed = record.model_copy(
            update={
                "after": record.after.model_copy(
                    update={"next_ids": ("forged", *record.after.next_ids[1:])}
                )
            }
        )
    _ = path.write_text(changed.model_dump_json())
    # When/Then: primary metric validity cannot bypass recovery-gate corruption.
    with pytest.raises(ValueError, match=r"[Aa]rtifact|[Ee]vent|[Rr]estor"):
        _ = verify(root, protocol, inputs)


def test_recovery_cli_when_bundle_complete_writes_new_verified_report(
    completed: tuple[Path, Inputs], tmp_path: Path
) -> None:
    # Given: a completed immutable run and a distinct absent report destination.
    root, _ = completed
    destination = tmp_path / "recovery-report.json"
    # When: a user drives the read-only verifier through its real CLI.
    result = CliRunner().invoke(
        app,
        [
            "--protocol",
            str(root / "protocol.json"),
            "--root",
            str(root),
            "--output",
            str(destination),
        ],
    )
    # Then: verification succeeds visibly and publishes only the new report.
    assert result.exit_code == 0, result.output
    assert "BABI_TASK1_RECOVERY_VERIFIED" in result.output
    recovered = RecoveryReport.model_validate_json(destination.read_bytes())
    assert not recovered.local_cuda_continuation_verified
