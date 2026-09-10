"""Maximum likelihood, causality, controls and exact update-boundary resume."""

from pathlib import Path

import numpy as np
import pytest
import torch

from flyrl.ar_checkpoint import load_checkpoint, save_checkpoint
from flyrl.ar_config import ARConfig
from flyrl.ar_framework import optimizer_tensors
from flyrl.ar_learning import ARLearner
from flyrl.connectome import Graph


@pytest.fixture
def graph() -> Graph:
    source, target = np.nonzero(np.ones((12, 12)) - np.eye(12))
    return Graph(
        tuple(f"synthetic:{i}" for i in range(12)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size, dtype=np.float64),
        "synthetic test circuit, not anatomy",
    )


def test_causality_and_all_position_loss(graph: Graph) -> None:
    learner = ARLearner(graph, ARConfig(alphabet_size=2, context=3, batch_size=2))
    first = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    changed = first.clone()
    changed[:, -1] = 1 - changed[:, -1]
    a = learner.model.forward(first[:, :-1])
    b = learner.model.forward(changed[:, :-1])
    assert torch.equal(a, b)
    changed[:, 1] = 1 - changed[:, 1]
    assert torch.equal(a[:, :1], learner.model.forward(changed[:, :-1])[:, :1])
    expected = torch.nn.functional.cross_entropy(
        a.reshape(-1, 2), first[:, 1:].reshape(-1)
    )
    assert torch.equal(learner.loss(first), expected)


def test_likelihood_learning_beats_unigram(graph: Graph) -> None:
    learner = ARLearner(
        graph,
        ARConfig(alphabet_size=2, context=4, batch_size=8, learning_rate=0.03),
    )
    text = np.array([0, 1] * 100, dtype=np.int64)
    before = learner.evaluate(text)
    learner.train(text, 45)
    after = learner.evaluate(text)
    assert after.nll < before.nll
    assert after.nll < 0.3  # Unigram NLL is log(2).
    assert after.greedy_accuracy == 1.0


def test_train_frozen_and_shuffled_invariants(graph: Graph) -> None:
    real = ARLearner(graph, ARConfig(alphabet_size=2, context=3))
    frozen = ARLearner(graph, real.config.model_copy(update={"control": "frozen"}))
    shuffled = ARLearner(graph, real.config.model_copy(update={"control": "shuffled"}))
    original = real.model.weight.detach().clone()
    bias = frozen.model.bias.detach().clone()
    head = frozen.model.readout.detach().clone()
    assert torch.equal(real.model.weight, shuffled.model.weight)
    assert torch.equal(real.model.codes, shuffled.model.codes)
    assert torch.equal(real.model.ports, shuffled.model.ports)
    assert not torch.equal(real.model.edges, shuffled.model.edges)
    text = np.array([0, 1] * 40, dtype=np.int64)
    real.train(text, 2)
    frozen.train(text, 2)
    assert not torch.equal(real.model.weight, original)
    assert torch.equal(frozen.model.weight, original)
    assert torch.equal(frozen.model.bias, bias)
    assert not torch.equal(frozen.model.readout, head)


def test_evaluation_isolation_and_exact_resume(graph: Graph, tmp_path: Path) -> None:
    config = ARConfig(alphabet_size=2, context=3, batch_size=4)
    full, partial = ARLearner(graph, config), ARLearner(graph, config)
    text = np.array([0, 1] * 40, dtype=np.int64)
    full.train(text, 5)
    partial.train(text, 2)
    state = partial.window_rng.get_state().clone()
    metrics = partial.evaluate(text)
    assert partial.evaluate(text) == metrics
    _ = partial.generate([0, 1], 5)
    assert torch.equal(state, partial.window_rng.get_state())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(partial, path, "fixture")
    resumed = ARLearner(graph, config)
    load_checkpoint(resumed, path, "fixture")
    resumed.train(text, 3)
    assert resumed.trace == full.trace
    assert resumed.updates == full.updates
    assert torch.equal(resumed.window_rng.get_state(), full.window_rng.get_state())
    for name, value in full.model.named_parameters():
        assert torch.equal(value, resumed.model.get_parameter(name))
    for name, value in full.model.named_buffers():
        assert torch.equal(value, resumed.model.get_buffer(name))
    for left, right in zip(
        optimizer_tensors(full.optimizer).values(),
        optimizer_tensors(resumed.optimizer).values(),
        strict=True,
    ):
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key])
    with pytest.raises(ValueError, match="compatibility"):
        load_checkpoint(resumed, path, "wrong-corpus")


def test_cuda_never_falls_back(graph: Graph, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        _ = ARLearner(graph, ARConfig(alphabet_size=2, device="cuda"))
