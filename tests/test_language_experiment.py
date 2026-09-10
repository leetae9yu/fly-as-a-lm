"""Real CLI and experiment integration using clearly labeled synthetic text."""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from flyrl import language
from flyrl import language_experiment as experiment
from flyrl.connectome import Graph, save_graph
from flyrl.language_data import (
    Corpus,
    CorpusLimits,
    TextSplits,
    make_corpus,
    save_corpus,
)
from flyrl.language_models import LanguageConfig, LanguageError


@pytest.fixture
def graph() -> Graph:
    source, target = np.nonzero(np.ones((9, 9)) - np.eye(9))
    return Graph(
        tuple(f"synthetic:{i}" for i in range(9)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size, dtype=np.float64),
        "synthetic integration graph; not biological",
    )


@pytest.fixture
def corpus() -> Corpus:
    return make_corpus(
        TextSplits("ab" * 64, "ba" * 32, "ab" * 32, "synthetic integration text"),
        CorpusLimits(max_symbols=3),
    )


def test_controls_when_experiment_runs(
    graph: Graph, corpus: Corpus, tmp_path: Path
) -> None:
    # Given paired topology/plasticity controls with a fixed final update.
    config = experiment.Experiment(
        output=tmp_path,
        updates=3,
        seeds=(0,),
        learner=LanguageConfig(
            alphabet_size=3, context=2, batch_size=8, sample_length=8
        ),
    )
    # When the full experiment runs.
    summary = experiment.run_experiment(graph, corpus, config)
    # Then all controls use identical ports, and frozen weights cannot change.
    assert {record.control for record in summary.runs} == {"real", "shuffled", "frozen"}
    assert len({record.sensory for record in summary.runs}) == 1
    assert len({record.output for record in summary.runs}) == 1
    frozen = next(record for record in summary.runs if record.control == "frozen")
    assert frozen.initial_weights == frozen.final_weights
    assert all(record.updates == 3 for record in summary.runs)
    assert all(record.final.test.windows == 21 for record in summary.runs)
    assert all(len(record.continuation) == 10 for record in summary.runs)
    assert (tmp_path / "summary.json").exists()


def test_test_split_when_labels_change(
    graph: Graph, corpus: Corpus, tmp_path: Path
) -> None:
    # Given equal training/validation data but a different test split.
    changed = make_corpus(
        TextSplits("ab" * 64, "ba" * 32, "aa" * 32, "synthetic integration text"),
        CorpusLimits(max_symbols=3),
    )
    shared = LanguageConfig(alphabet_size=3, context=2, batch_size=8, sample_length=0)
    # When the fixed-budget experiment runs independently on both corpora.
    first = experiment.run_experiment(
        graph,
        corpus,
        experiment.Experiment(
            output=tmp_path / "a",
            updates=2,
            seeds=(0,),
            learner=shared,
        ),
    )
    second = experiment.run_experiment(
        graph,
        changed,
        experiment.Experiment(
            output=tmp_path / "b",
            updates=2,
            seeds=(0,),
            learner=shared,
        ),
    )
    # Then weights and train/validation metrics are unaffected by test labels.
    for a, b in zip(first.runs, second.runs, strict=True):
        assert a.final_weights == b.final_weights
        assert a.final.train == b.final.train
        assert a.final.valid == b.final.valid


def test_resume_when_budget_is_extended(
    graph: Graph, corpus: Corpus, tmp_path: Path
) -> None:
    # Given completed checkpoints and a larger prespecified update target.
    config = experiment.Experiment(
        output=tmp_path / "resumed",
        updates=2,
        seeds=(0,),
        learner=LanguageConfig(
            alphabet_size=3, context=2, batch_size=8, sample_length=0
        ),
    )
    _ = experiment.run_experiment(graph, corpus, config)
    # When resuming to the same budget as an uninterrupted reference.
    resumed = experiment.run_experiment(
        graph,
        corpus,
        config.model_copy(
            update={"updates": 5, "resume": True},
        ),
    )
    full = experiment.run_experiment(
        graph,
        corpus,
        config.model_copy(
            update={"updates": 5, "output": tmp_path / "full"},
        ),
    )
    # Then final metrics, samples, reward histories and weights are exactly equal.
    for a, b in zip(resumed.runs, full.runs, strict=True):
        assert a.final == b.final
        assert a.final_weights == b.final_weights
        assert a.training_rewards == b.training_rewards


def test_existing_checkpoint_when_resume_is_omitted(
    graph: Graph,
    corpus: Corpus,
    tmp_path: Path,
) -> None:
    # Given an existing completed run.
    config = experiment.Experiment(
        output=tmp_path,
        updates=0,
        seeds=(0,),
        learner=LanguageConfig(alphabet_size=3, context=2, sample_length=0),
    )
    _ = experiment.run_experiment(graph, corpus, config)
    # When rerunning without explicit resume.
    # Then accidental overwrites fail.
    with pytest.raises(LanguageError, match="resume"):
        _ = experiment.run_experiment(graph, corpus, config)


def test_cli_when_cpu_run_is_requested(
    graph: Graph, corpus: Corpus, tmp_path: Path
) -> None:
    # Given portable graph and corpus files.
    graph_path, corpus_path = tmp_path / "graph.npz", tmp_path / "corpus.npz"
    save_graph(graph, graph_path)
    save_corpus(corpus, corpus_path)
    output = tmp_path / "output"
    # When the actual CLI receives its documented options.
    result = CliRunner().invoke(
        language.app,
        [
            "--graph",
            str(graph_path),
            "--corpus",
            str(corpus_path),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--updates",
            "2",
            "--batch-size",
            "8",
            "--context",
            "2",
            "--seeds",
            "0",
            "--eval-windows",
            "8",
            "--sample-length",
            "4",
        ],
    )
    # Then the report exposes actual placement and all three paired controls.
    assert result.exit_code == 0, result.output
    summary = experiment.Summary.model_validate_json(
        (output / "summary.json").read_text()
    )
    assert len(summary.runs) == 3
    assert all(run.device.tensor_device == "cpu" for run in summary.runs)
    assert all(run.final.valid.windows == 8 for run in summary.runs)
