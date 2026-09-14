"""Real CPU quality-runner transactions, evidence and deterministic continuation."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile
from pydantic import ValidationError
from typer.testing import CliRunner

from flyrl.activations import ActivationMetadata
from flyrl.ar_checkpoint import load_checkpoint, save_checkpoint
from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.bpe_data import save_bpe_corpus
from flyrl.connectome import Graph, load_graph, save_graph
from flyrl.language_checkpoint import atomic_text
from flyrl.language_runtime import graph_fingerprint
from flyrl.quality_corpus import QualityPreparation, prepare_quality_corpus
from flyrl.quality_generation import QualityGenerationOptions, generate_quality_panel
from flyrl.quality_metrics import QualityMetrics, StoryEvidence
from flyrl.quality_pilot import (
    evaluate_test,
    evaluate_validation,
    exclusive,
    persist_progress,
)
from flyrl.quality_pilot_schema import (
    ArtifactIdentity,
    CorpusIdentity,
    Evaluation,
    QualityProtocol,
    Schedule,
    select_update,
)
from flyrl.story_data import load_story_corpus
from scripts.connectome_source import file_digest
from scripts.run_quality_pilot import app, run

if TYPE_CHECKING:
    from numpy import generic


@pytest.fixture
def protocol(tmp_path: Path) -> QualityProtocol:
    train, heldout = tmp_path / "train.txt", tmp_path / "heldout.txt"
    _ = train.write_text("A cat ran home.\n<|endoftext|>\nA dog found a ball.")
    _ = heldout.write_text("A bird sang softly.\n<|endoftext|>\nA fox danced happily.")
    stories = prepare_quality_corpus(
        QualityPreparation(
            raw_train=train,
            raw_valid=heldout,
            source="synthetic",
            revision="1",
            license_text="CC0",
            train_stories=2,
            valid_stories=1,
            test_stories=1,
        )
    )
    corpus_path, graph_path = tmp_path / "corpus.npz", tmp_path / "graph.npz"
    save_bpe_corpus(stories.corpus, corpus_path)
    source, target = np.nonzero(np.ones((6, 6)) - np.eye(6))
    graph = Graph(
        tuple(str(i) for i in range(6)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size),
        "synthetic",
    )
    save_graph(graph, graph_path)
    return QualityProtocol(
        profile="smoke",
        graph=ArtifactIdentity(
            path=graph_path,
            sha256=file_digest(graph_path),
            fingerprint=graph_fingerprint(graph),
            nodes=6,
            edges=30,
        ),
        corpus=CorpusIdentity(
            path=corpus_path,
            sha256=file_digest(corpus_path),
            fingerprint=stories.corpus.fingerprint,
            stories=(2, 1, 1),
            tokens=(
                int(stories.corpus.train.size),
                int(stories.corpus.valid.size),
                int(stories.corpus.test.size),
            ),
            vocabulary=len(stories.corpus.vocabulary),
        ),
        config=ARConfig(
            alphabet_size=len(stories.corpus.vocabulary),
            tokenization="bpe",
            trainable_codes=True,
            context=2,
            batch_size=2,
            readout_neurons=3,
        ),
        schedule=Schedule(target_updates=4, warmup_updates=2),
        evaluation_updates=(0, 1, 2, 4),
        early_update=1,
        checkpoint_interval=1,
        prompts=("A cat", "A fox"),
        activation_prompt_indices=(0,),
        generation=QualityGenerationOptions(length=2),
    )


def test_lr_boundaries() -> None:
    schedule = Schedule()
    assert schedule.rate(1) == 0.003 / 500
    assert schedule.rate(500) == 0.003
    assert schedule.rate(15_250) == pytest.approx(0.00165)
    assert schedule.rate(30_000) == 0.0003
    assert schedule.rate(501) < schedule.rate(500)
    for update in (0, 30_001):
        with pytest.raises(ValueError, match="update"):
            _ = schedule.rate(update)


def test_selection_excludes_initial_and_breaks_ties() -> None:
    metrics = tuple(
        Evaluation(
            update=update,
            checkpoint_sha256="a" * 64,
            metrics=QualityMetrics(
                per_story=(
                    StoryEvidence(
                        sha256="b" * 64,
                        loss_sum=loss,
                        count=2,
                        correct=0,
                    ),
                )
            ),
        )
        for update, loss in ((0, 0.1), (1, 8.0), (2, 6.0), (4, 6.0))
    )
    assert select_update(metrics) == 2
    assert select_update(tuple(reversed(metrics))) == 2


@pytest.mark.parametrize("pause", [0, 1, 2, 3])
def test_resume_equals_uninterrupted(
    protocol: QualityProtocol, tmp_path: Path, pause: int
) -> None:
    full, split = tmp_path / "full", tmp_path / "split"
    complete = run(protocol, full)
    partial = run(protocol, split, stop_after=pause)
    assert partial.updates == pause
    assert partial.next_lr == protocol.schedule.rate(pause + 1)
    preserved = (split / "evaluation-000000.json").stat().st_mtime_ns
    resumed = run(protocol, split, resume=True)
    assert resumed == complete
    assert (split / "evaluation-000000.json").stat().st_mtime_ns == preserved
    left: NpzFile[generic] = NpzFile(
        (full / "checkpoint-latest.npz").open("rb"), own_fid=True, allow_pickle=False
    )
    right: NpzFile[generic] = NpzFile(
        (split / "checkpoint-latest.npz").open("rb"), own_fid=True, allow_pickle=False
    )
    with left, right:
        assert left.files == right.files
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])
    assert {r.update for r in resumed.completed} == {1, resumed.selected_update, 4}
    assert [v.update for v in resumed.validation] == [0, 1, 2, 4]
    assert run(protocol, split, resume=True) == resumed
    for update in protocol.evaluation_updates:
        assert (split / f"checkpoint-{update:06d}.npz").exists()


def test_generation_and_activation_identity(
    protocol: QualityProtocol,
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    report = run(protocol, output)
    learner = ARLearner(load_graph(protocol.graph.path), protocol.config)
    corpus = load_story_corpus(protocol.corpus.path).corpus
    for result in report.completed:
        load_checkpoint(
            learner, output / f"checkpoint-{result.update:06d}.npz", corpus.fingerprint
        )
        draws = 3 if result.update == report.selected_update else 1
        expected = generate_quality_panel(
            learner,
            corpus,
            protocol.prompts,
            tuple(tuple(10_000 + 100 * i + j for j in range(draws)) for i in range(2)),
            protocol.generation,
        )
        assert result.generation == expected
        for path in result.activations:
            metadata = ActivationMetadata.model_validate_json(
                (output / path).read_text()
            )
            assert metadata.updates == result.update
            assert metadata.corpus_fingerprint == corpus.fingerprint
            assert metadata.graph_fingerprint == protocol.graph.fingerprint
            assert metadata.generated_ids == expected[0].token_ids
            arrays: NpzFile[generic] = NpzFile(
                ((output / path).parent / "activations.npz").open("rb"),
                own_fid=True,
                allow_pickle=False,
            )
            with arrays:
                assert arrays["states"].shape == (2, 6)
                assert arrays["states"].dtype == np.float32
    assert report.success_label != "language-quality improvement demonstrated"


def test_reject_protocol_hash_and_collisions(
    protocol: QualityProtocol, tmp_path: Path
) -> None:
    output = tmp_path / "run"
    _ = run(protocol, output, stop_after=2)
    with pytest.raises(FileExistsError):
        _ = run(protocol, output)
    changed = protocol.model_copy(update={"prompts": ("Different prompt", "A fox")})
    with pytest.raises(ValueError, match="protocol"):
        _ = run(changed, output, resume=True)
    path = output / "progress.json"
    data = cast("dict[str, object]", json.loads(path.read_text()))
    data["updates"] = 1
    _ = path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="progress"):
        _ = run(protocol, output, resume=True)
    with pytest.raises(ValueError, match="identity"):
        _ = run(
            protocol.model_copy(
                update={"graph": protocol.graph.model_copy(update={"sha256": "0" * 64})}
            ),
            tmp_path / "bad",
        )


def test_strict_protocol_and_cli(protocol: QualityProtocol, tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    _ = path.write_text(protocol.model_dump_json())
    loaded = QualityProtocol.model_validate_json(path.read_text(), strict=True)
    assert loaded == protocol
    for field, value in (("unknown", 1), ("checkpoint_interval", "1")):
        data = cast("dict[str, object]", json.loads(path.read_text()))
        data[field] = value
        with pytest.raises(ValidationError):
            _ = QualityProtocol.model_validate_json(json.dumps(data), strict=True)
    result = CliRunner().invoke(
        app, ["--protocol", str(path), "--output", str(tmp_path / "cli")]
    )
    assert result.exit_code == 0, result.output
    lines = [
        cast("dict[str, object]", json.loads(line))
        for line in result.stdout.splitlines()
    ]
    assert lines[-1]["event"] == "final"
    assert all(line["event"] in {"evaluation", "checkpoint", "final"} for line in lines)
    assert torch.get_num_threads() == 1


def test_every_update_uses_schedule_and_legal_windows(
    protocol: QualityProtocol,
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    report = run(protocol, output)
    stories = load_story_corpus(protocol.corpus.path)
    manual = ARLearner(load_graph(protocol.graph.path), protocol.config)
    for rate in (0.0015, 0.003, 0.00165, 0.0003):
        for group in manual.optimizer.param_groups:
            group["lr"] = rate
        manual.train(
            stories.corpus.train, 1, starts=stories.starts(stories.metadata.train, 2)
        )
    restored = ARLearner(manual.graph, manual.config)
    load_checkpoint(
        restored, output / "checkpoint-latest.npz", stories.corpus.fingerprint
    )
    assert restored.trace == manual.trace
    assert torch.equal(restored.window_rng.get_state(), manual.window_rng.get_state())
    for left, right in zip(
        restored.model.parameters(), manual.model.parameters(), strict=True
    ):
        assert torch.equal(left, right)
    assert report.selected_update == select_update(report.validation)
    for item in report.validation:
        assert (
            item.metrics.count == protocol.corpus.tokens[1] - protocol.corpus.stories[1]
        )
    for result in report.completed:
        assert (
            result.test.count == protocol.corpus.tokens[2] - protocol.corpus.stories[2]
        )
    assert len(report.summaries) == len(report.validation) + len(report.completed)


@pytest.mark.parametrize(
    "field", ["next_lr", "runtime", "validation", "selected_update"]
)
def test_reject_progress_corruption(
    protocol: QualityProtocol,
    tmp_path: Path,
    field: str,
) -> None:
    output = tmp_path / "run"
    progress = run(protocol, output, stop_after=2)
    changes: dict[str, object] = {
        "next_lr": 0.7,
        "runtime": "another runtime",
        "validation": (),
        "selected_update": 1,
    }
    bad = progress.model_copy(update={field: changes[field]})
    atomic_text(output / "progress.json", bad.model_dump_json())
    with pytest.raises(ValueError, match="progress"):
        _ = run(protocol, output, resume=True)


@pytest.mark.parametrize("mismatch", ["config", "corpus", "runtime"])
def test_existing_checkpoint_compatibility_is_enforced(
    protocol: QualityProtocol,
    tmp_path: Path,
    mismatch: str,
) -> None:
    output = tmp_path / "run"
    progress = run(protocol, output, stop_after=2)
    config = (
        protocol.config.model_copy(update={"seed": 1})
        if mismatch == "config"
        else protocol.config
    )
    learner = ARLearner(load_graph(protocol.graph.path), config)
    if mismatch == "runtime":
        torch.set_num_threads(2)
    checkpoint = output / "checkpoint-latest.npz"
    save_checkpoint(
        learner,
        checkpoint,
        "0" * 64 if mismatch == "corpus" else protocol.corpus.fingerprint,
    )
    bad = progress.model_copy(update={"checkpoint_sha256": file_digest(checkpoint)})
    atomic_text(output / "progress.json", bad.model_dump_json())
    with pytest.raises(ValueError, match="compatibility"):
        _ = run(protocol, output, resume=True)


@pytest.mark.parametrize(
    "field", ["nodes", "edges", "stories", "tokens", "fingerprint"]
)
def test_counts_and_fingerprints_are_verified(
    protocol: QualityProtocol,
    tmp_path: Path,
    field: str,
) -> None:
    if field in {"nodes", "edges"}:
        changed = protocol.model_copy(
            update={"graph": protocol.graph.model_copy(update={field: 7})}
        )
    else:
        changed = protocol.model_copy(
            update={
                "corpus": protocol.corpus.model_copy(
                    update={field: "0" * 64 if field == "fingerprint" else (3, 2, 2)}
                )
            }
        )
    with pytest.raises(ValueError, match="identity"):
        _ = run(changed, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_resume_finalization_preserves_completed_test(
    protocol: QualityProtocol,
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    progress = run(protocol, output, stop_after=3)
    learner = ARLearner(load_graph(protocol.graph.path), protocol.config)
    stories = load_story_corpus(protocol.corpus.path)
    load_checkpoint(
        learner, output / "checkpoint-latest.npz", stories.corpus.fingerprint
    )
    for group in learner.optimizer.param_groups:
        group["lr"] = protocol.schedule.rate(4)
    learner.train(
        stories.corpus.train, 1, starts=stories.starts(stories.metadata.train, 2)
    )
    progress = persist_progress(
        learner, output, evaluate_validation(learner, stories, output, progress)
    )
    result = evaluate_test(learner, stories, output, progress, 1)
    exclusive(output / "result-000001.json", result.model_dump_json())
    progress = progress.model_copy(update={"completed": (result,)})
    atomic_text(output / "progress.json", progress.model_dump_json())
    unchanged = {
        p: p.stat().st_mtime_ns
        for p in (output / "activations-000001").rglob("*")
        if p.is_file()
    }
    report = run(protocol, output, resume=True)
    assert report.completed[0] == result
    assert all(p.stat().st_mtime_ns == before for p, before in unchanged.items())
    artifact = output / report.completed[0].artifacts[0].path
    _ = artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact identity"):
        _ = run(protocol, output, resume=True)


@pytest.mark.parametrize(
    "field", ["architecture", "context", "profile", "activation_prompt_indices"]
)
def test_reject_nonconnectome_illegal_windows_and_quality_smoke_confusion(
    protocol: QualityProtocol,
    tmp_path: Path,
    field: str,
) -> None:
    changes: dict[str, object] = {
        "architecture": protocol.config.model_copy(update={"control": "shuffled"}),
        "context": protocol.config.model_copy(update={"context": 10_000}),
        "profile": "quality",
        "activation_prompt_indices": (2,),
    }
    key = "config" if field in {"architecture", "context"} else field
    with pytest.raises(
        ValueError, match=r"connectome|enough tokens|Quality profile|panel"
    ):
        _ = run(protocol.model_copy(update={key: changes[field]}), tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
