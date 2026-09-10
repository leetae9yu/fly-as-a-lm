"""Checkpoint and CLI integration regression tests."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from flyrl.checkpoint import Checkpoint, atomic_write, load_checkpoint, save_checkpoint
from flyrl.cli import app
from flyrl.connectome import Graph, save_graph
from flyrl.experiment import Experiment, Summary, run_experiment
from flyrl.learning import ExperimentError, Learner
from flyrl.models import Config, Task


@pytest.fixture
def synthetic_graph() -> Graph:
    return Graph(
        tuple(f"synthetic:{i}" for i in range(4)),
        np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64),
        np.array([1, 2, 2, 3, 3, 0, 0, 1], dtype=np.int64),
        np.arange(1, 9, dtype=np.float64),
        "synthetic regression fixture; not biological data",
    )


@pytest.mark.parametrize("task", list(Task))
def test_resume_when_split_at_episode_boundary(
    synthetic_graph: Graph,
    tmp_path: Path,
    task: Task,
) -> None:
    # Given an uninterrupted reference and a partially trained matching run.
    full = Learner(synthetic_graph, Config(seed=9, task=task))
    split = Learner(synthetic_graph, Config(seed=9, task=task))
    full.train(100)
    split.train(37)
    checkpoint = tmp_path / "checkpoint.json"
    # When persisted state resumes for the remaining episodes.
    save_checkpoint(split, checkpoint)
    restored = load_checkpoint(checkpoint)
    restored.train(63)
    # Then state, RNG and evaluation match exactly, not approximately.
    np.testing.assert_array_equal(full.weights, restored.weights)
    np.testing.assert_array_equal(full.eligibility, restored.eligibility)
    assert full.baseline == restored.baseline
    assert full.rewards == restored.rewards
    assert full.rng.bit_generator.state == restored.rng.bit_generator.state
    assert full.evaluate() == restored.evaluate()


def test_cli_when_running_and_resuming(synthetic_graph: Graph, tmp_path: Path) -> None:
    # Given a prepared synthetic NPZ graph.
    graph_path = tmp_path / "synthetic.npz"
    save_graph(synthetic_graph, graph_path)
    output = tmp_path / "cli"
    command = [
        "--graph",
        str(graph_path),
        "--output",
        str(output),
        "--episodes",
        "12",
        "--seeds",
        "0,1",
        "--eval-trials",
        "16",
        "--checkpoint-every",
        "5",
    ]
    # When the real CLI runs and then resumes to the same target.
    first = CliRunner().invoke(app, command, catch_exceptions=False)
    second = CliRunner().invoke(app, [*command, "--resume"], catch_exceptions=False)
    summary = Summary.model_validate_json(first.stdout)
    # Then reports are identical and all task/control/seed combinations exist.
    assert first.exit_code == second.exit_code == 0
    assert Summary.model_validate_json(second.stdout) == summary
    assert len(summary.runs) == 12
    assert len(summary.aggregates) == 6
    assert all(run.episodes == 12 for run in summary.runs)
    assert (
        len(
            {
                (run.ports.sensory, run.ports.output)
                for run in summary.runs
                if run.config.seed == 0
            }
        )
        == 1
    )


def test_experiment_resume_when_target_increases(
    synthetic_graph: Graph,
    tmp_path: Path,
) -> None:
    # Given a partial suite and an uninterrupted reference.
    full = run_experiment(
        synthetic_graph,
        Experiment(
            output=tmp_path / "full",
            episodes=31,
            seeds=(4,),
            delay=2,
            eval_trials=16,
        ),
    )
    _ = run_experiment(
        synthetic_graph,
        Experiment(
            output=tmp_path / "split",
            episodes=9,
            seeds=(4,),
            delay=2,
            eval_trials=16,
        ),
    )
    # When the complete suite resumes.
    resumed = run_experiment(
        synthetic_graph,
        Experiment(
            output=tmp_path / "split",
            resume=True,
            episodes=31,
            seeds=(4,),
            delay=2,
            eval_trials=16,
        ),
    )
    # Then JSON machine-consumed results equal the uninterrupted suite.
    assert resumed == full


def test_checkpoint_when_config_is_incompatible(
    synthetic_graph: Graph,
    tmp_path: Path,
) -> None:
    # Given persisted training with one seed.
    path = tmp_path / "checkpoint.json"
    save_checkpoint(Learner(synthetic_graph, Config(seed=3)), path)
    expected = Learner(synthetic_graph, Config(seed=4))
    # When resumption requests different dynamics.
    with pytest.raises(ExperimentError):
        _ = load_checkpoint(path, expected)
    # Then no mismatching run is returned.


def test_initial_weights_when_topology_is_shuffled(
    synthetic_graph: Graph,
    tmp_path: Path,
) -> None:
    # Given the controls at zero episodes, isolating initialization.
    summary = run_experiment(
        synthetic_graph,
        Experiment(
            output=tmp_path,
            episodes=0,
            seeds=(2,),
            eval_trials=16,
        ),
    )
    # When the initialized real/shuffle checkpoints are read.
    real = load_checkpoint(tmp_path / summary.runs[0].checkpoint)
    shuffled = load_checkpoint(tmp_path / summary.runs[1].checkpoint)
    # Then only topology, not edge-ordered initial weights, was intervened on.
    np.testing.assert_array_equal(real.weights, shuffled.weights)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weights", (0.1,)),
        ("eligibility", (0.0,)),
        ("weights", (-1.0,) * 8),
        ("weights", (float("nan"),) * 8),
        ("numpy_version", "incompatible"),
    ],
)
def test_checkpoint_when_payload_is_invalid(
    synthetic_graph: Graph,
    tmp_path: Path,
    field: str,
    value: tuple[float, ...] | str,
) -> None:
    # Given an artifact with one deliberately malformed machine field.
    path = tmp_path / "state.json"
    save_checkpoint(Learner(synthetic_graph, Config()), path)
    saved = Checkpoint.model_validate_json(path.read_bytes())
    corrupted = saved.model_copy(update={field: value})
    atomic_write(path, corrupted.model_dump_json())
    # When parsing untrusted checkpoint data.
    with pytest.raises((ExperimentError, ValidationError)):
        _ = load_checkpoint(path)
    # Then invalid state cannot enter the engine.


def test_evaluation_when_checkpointed_state_is_compared(
    synthetic_graph: Graph,
    tmp_path: Path,
) -> None:
    # Given a learner with nonzero eligibility and an advanced RNG.
    learner = Learner(synthetic_graph, Config())
    learner.train(11)
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    save_checkpoint(learner, before)
    # When evaluation is interleaved.
    _ = learner.evaluate()
    save_checkpoint(learner, after)
    # Then every serialized bit of training state is untouched.
    assert before.read_bytes() == after.read_bytes()


def test_atomic_write_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given an existing checkpoint and a failing filesystem rename.
    path = tmp_path / "checkpoint.json"
    atomic_write(path, '{"state":1}')

    def fail_replace(_source: Path, target: Path) -> Path:
        raise PermissionError(13, "simulated rename failure", str(target))

    monkeypatch.setattr(Path, "replace", fail_replace)
    # When saving fails before the atomic replacement.
    with pytest.raises(PermissionError):
        atomic_write(path, '{"state":2}')
    # Then the original survives and the temporary artifact is cleaned up.
    assert path.read_text() == '{"state":1}'
    assert tuple(tmp_path.iterdir()) == (path,)


def test_cli_when_seed_is_invalid(tmp_path: Path, synthetic_graph: Graph) -> None:
    # Given a valid graph but malformed user input.
    path = tmp_path / "synthetic.npz"
    save_graph(synthetic_graph, path)
    # When the CLI parses a noninteger seed.
    result = CliRunner().invoke(app, ["--graph", str(path), "--seeds", "bad"])
    # Then it exits nonzero rather than running with a fallback seed.
    assert result.exit_code != 0
