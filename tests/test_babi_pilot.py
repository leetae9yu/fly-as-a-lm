"""Real CPU CLI fresh/resumed execution and the official-test access boundary."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from flyrl.babi_data import BabiError, encode_question, load_tokenizer, parse_episodes
from flyrl.babi_learning import BabiLearner
from flyrl.babi_pilot import Experiment
from flyrl.babi_pilot_schema import Evaluation, Preflight, Report
from flyrl.babi_protocol import BabiProtocol, CorpusIdentity
from flyrl.babi_recovery import ArtifactManifest, EventChain, Restoration
from flyrl.babi_storage import Split, TrainValid
from scripts.connectome_source import file_digest
from scripts.run_babi_task1 import app, run
from tests.test_babi_checkpoint import fixture


def smoke_protocol(tmp_path: Path) -> BabiProtocol:
    protocol, _, _ = fixture(tmp_path)
    tokenizer = load_tokenizer(Path("data/tinystories_quality/tokenizer.json"))
    source = (
        b"1 Mary moved to the bathroom.\n2 Where is Mary?\tbathroom\t1\n"
        b"3 John went to the kitchen.\n4 Where is John?\tkitchen\t3\n"
        b"1 Sandra moved to the garden.\n2 Where is Sandra?\tgarden\t1\n"
        b"3 Daniel went to the office.\n4 Where is Daniel?\toffice\t3\n"
        b"1 Mary moved to the bedroom.\n2 Where is Mary?\tbedroom\t1\n"
        b"3 John went to the hallway.\n4 Where is John?\thallway\t3\n"
    )
    splits = tuple(
        Split(questions=tuple(encode_question(q, tokenizer) for q in episode))
        for episode in parse_episodes(source)
    )
    corpus = TrainValid(train=splits[0], valid=splits[1])
    _ = (tmp_path / "train_valid.json").write_text(corpus.model_dump_json())
    _ = (tmp_path / "test.json").write_text(splits[2].model_dump_json())
    _ = (tmp_path / "tokenizer.json").write_bytes(
        Path("data/tinystories_quality/tokenizer.json").read_bytes()
    )
    return BabiProtocol.model_validate_json(
        protocol.model_copy(
            update={
                "config": protocol.config.model_copy(update={"context": 64}),
                "corpus": CorpusIdentity(
                    path=tmp_path,
                    train_valid_sha256=file_digest(tmp_path / "train_valid.json"),
                    test_sha256=file_digest(tmp_path / "test.json"),
                    train_valid_fingerprint=corpus.fingerprint,
                    counts=(2, 2, 2),
                ),
            }
        ).model_dump_json()
    )


def test_runner_when_resumed_matches_uninterrupted(tmp_path: Path) -> None:
    # Given: sealed small CPU fixtures and two independent output directories.
    protocol = smoke_protocol(tmp_path)
    full, partial = tmp_path / "full", tmp_path / "partial"
    complete = run(protocol, full)
    _ = run(protocol, partial, stop_after=1)
    # When: continuing the interrupted transaction through selection and test.
    resumed = run(protocol, partial, resume=True)
    # Then: every prediction, selected checkpoint and deterministic state is equal.
    assert resumed == complete
    assert (full / "latest.json").read_bytes() == (partial / "latest.json").read_bytes()
    assert run(protocol, partial, resume=True) == resumed
    assert isinstance(resumed, Report)
    assert resumed.benchmark_evidence is False


def test_official_test_when_selection_not_frozen_stays_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a read boundary observing selection, not an evaluation mock.
    protocol = smoke_protocol(tmp_path)
    output = tmp_path / "run"
    original = Path.read_bytes
    observed: list[bool] = []

    def guarded(path: Path) -> bytes:
        if path == tmp_path / "test.json":
            observed.append((output / "selection.json").exists())
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    # When: the real runner trains, selects and opens the held-out artifact.
    _ = run(protocol, output)
    # Then: every official-test byte read is after durable selection publication.
    assert observed
    assert all(observed)


def test_cli_when_fresh_then_resume(tmp_path: Path) -> None:
    # Given: a strict JSON protocol accepted by the real module entrypoint.
    protocol = smoke_protocol(tmp_path)
    path, output = tmp_path / "protocol.json", tmp_path / "cli"
    _ = path.write_text(protocol.model_dump_json())
    runner = CliRunner()
    first = runner.invoke(
        app, ["--protocol", str(path), "--output", str(output), "--stop-after", "1"]
    )
    assert first.exit_code == 0, first.output
    # When: CLI resumes without any hyperparameter overrides.
    result = runner.invoke(
        app, ["--protocol", str(path), "--output", str(output), "--resume"]
    )
    # Then: immutable selected evidence and final report are published.
    assert result.exit_code == 0, result.output
    assert (output / "selection.json").exists()
    assert (output / "report.json").exists()


def test_gate_when_memorization_panel_is_incomplete(tmp_path: Path) -> None:
    # Given: the smoke fixture cannot supply the frozen six-answer 12-example panel.
    protocol = smoke_protocol(tmp_path)
    experiment = Experiment.open(protocol, tmp_path / "run")
    # When/Then: quality panel construction fails instead of silently shrinking it.
    with pytest.raises(BabiError):
        _ = experiment.memorization_panel()


def test_published_evidence_when_complete_has_gate_trace_and_exact(
    tmp_path: Path,
) -> None:
    # Given/When: a real completed CPU experiment.
    protocol = smoke_protocol(tmp_path)
    output = tmp_path / "published"
    _ = run(protocol, output)
    # Then: machine publication retains diagnostic trace and literal exact status.
    assert (output / "memorization.json").exists()
    assert (output / "decision.json").exists()
    assert '"exact":' in (output / "ordinary.json").read_text()


def test_validation_tampering_when_resume_rejects_before_mutation(
    tmp_path: Path,
) -> None:
    # Given: changed generated evidence with unchanged aggregate counts and NLL.
    protocol = smoke_protocol(tmp_path)
    output = tmp_path / "tampered"
    _ = run(protocol, output, stop_after=1)
    experiment = Experiment.open(protocol, output)
    path = output / "validation-000000.json"
    evidence = Evaluation.model_validate_json(path.read_bytes())
    altered = evidence.model_copy(
        update={"predictions": tuple(reversed(evidence.predictions))}
    )
    _ = path.write_text(altered.model_dump_json(exclude_computed_fields=True))
    learner = BabiLearner(experiment.graph, protocol.config)
    # When/Then: rejection precedes restoration of the saved update.
    with pytest.raises(BabiError):
        _ = experiment.checkpoints.load(learner)
    assert learner.updates == 0


def test_projection_when_measured_uses_frozen_mean_inequality(tmp_path: Path) -> None:
    # Given/When: a real disposable performance fixture publishes its timings.
    protocol = smoke_protocol(tmp_path)
    output = tmp_path / "projection"
    _ = run(protocol, output, stop_after=0)
    record = Preflight.model_validate_json((output / "preflight.json").read_bytes())
    # Then: independently recompute the frozen conservative arithmetic.
    expected = (
        record.elapsed_seconds
        + 1.3
        * (
            12400 * (sum(record.timed_updates) / 20)
            + 9 * record.evaluation_seconds
            + 48 * record.checkpoint_seconds
        )
        + 600
    )
    assert record.projected_seconds == pytest.approx(expected)
    assert record.warmups == 5
    assert len(record.timed_updates) == 20


def test_completed_run_when_published_seals_recovery_evidence(tmp_path: Path) -> None:
    # Given: a complete run through the real CPU smoke surface.
    protocol = smoke_protocol(tmp_path)
    output = tmp_path / "sealed"
    report = run(protocol, output)
    assert isinstance(report, Report)
    # When: the completed bundle is parsed at its recovery boundaries.
    events = EventChain.model_validate_json((output / "events.json").read_bytes())
    restoration = Restoration.model_validate_json(
        (output / "restoration.json").read_bytes()
    )
    artifacts = ArtifactManifest.model_validate_json(
        (output / "artifacts.json").read_bytes()
    )
    # Then: ordering, same-state restore, replay and exact membership are sealed.
    assert tuple(event.phase for event in events.events) == (
        "settings_frozen",
        "selection_frozen",
        "test_opened",
        "completed",
    )
    assert restoration.before == restoration.after
    assert restoration.ordinary == report.ordinary.predictions[:32]
    assert restoration.removed == report.removed.predictions[:32]
    expected = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifacts.json"
    }
    assert {artifact.path for artifact in artifacts.artifacts} == expected
