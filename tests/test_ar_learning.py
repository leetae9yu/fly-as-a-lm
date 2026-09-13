"""Maximum likelihood, causality, controls and exact update-boundary resume."""

from pathlib import Path

import numpy as np
import pytest
import torch

from flyrl.ar_checkpoint import load_checkpoint, save_checkpoint
from flyrl.ar_config import ARConfig
from flyrl.ar_framework import optimizer_tensors
from flyrl.ar_learning import ARLearner
from flyrl.ar_prepared import prepare_recurrence
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


def test_explicit_port_configuration_drives_declared_neurons(graph: Graph) -> None:
    # Given: exact, disjoint sensory and readout graph indices.
    config = ARConfig(
        alphabet_size=2,
        readout_neurons=2,
        port_policy="random_random",
        port_manifest_sha256="a" * 64,
        sensory_indices=(1, 3),
        readout_indices=(8, 9),
    )
    # When: the anatomical model is constructed.
    learner = ARLearner(graph, config)
    # Then: no internal random port choice replaces the frozen protocol.
    assert torch.equal(learner.model.sensory, torch.tensor([1, 3]))
    assert torch.equal(learner.model.ports, torch.tensor([8, 9]))
    with pytest.raises(ValueError, match="disjoint"):
        _ = ARConfig(
            alphabet_size=2,
            readout_neurons=2,
            port_policy="random_random",
            port_manifest_sha256="a" * 64,
            sensory_indices=(1, 3),
            readout_indices=(3, 9),
        )


def test_explicit_port_policies_share_initial_trainable_tensors(graph: Graph) -> None:
    # Given: two policies with different node IDs but identical port counts and seed.
    anatomy = ARConfig(
        alphabet_size=2,
        readout_neurons=2,
        seed=7,
        port_policy="alpn_mbon",
        port_manifest_sha256="a" * 64,
        sensory_indices=(1, 3),
        readout_indices=(8, 9),
    )
    random = anatomy.model_copy(
        update={
            "port_policy": "random_random",
            "sensory_indices": (2, 4),
            "readout_indices": (10, 11),
        }
    )
    # When: both paired conditions initialize independently.
    left, right = ARLearner(graph, anatomy), ARLearner(graph, random)
    # Then: only port buffers differ; all learned tensors and edges start identically.
    assert not torch.equal(left.model.sensory, right.model.sensory)
    assert not torch.equal(left.model.ports, right.model.ports)
    assert torch.equal(left.model.edges, right.model.edges)
    for name, parameter in left.model.named_parameters():
        assert torch.equal(parameter, right.model.get_parameter(name))


def test_selected_states_match_step_and_chunked_extraction(graph: Graph) -> None:
    # Given: one frozen model, ordered selected neurons and a token batch.
    learner = ARLearner(
        graph,
        ARConfig(alphabet_size=2, context=4, batch_size=2),
    )
    model = learner.model
    tokens = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    selected = torch.tensor([7, 2, 10])
    prepared = prepare_recurrence(model.weight, model.topology, need_reverse=False)
    # When: states are recorded in one pass and across two carried-state chunks.
    features, final_state = model.selected_states(tokens, selected, prepared=prepared)
    first, carried = model.selected_states(
        tokens[:, :2],
        selected,
        prepared=prepared,
    )
    second, chunked_final = model.selected_states(
        tokens[:, 2:],
        selected,
        state=carried,
        prepared=prepared,
    )
    repeated: list[torch.Tensor] = []
    state = model.weight.new_zeros((model.nodes, tokens.shape[0]))
    for token in tokens.unbind(dim=1):
        _, state = model.step(token, state)
        repeated.append(state[selected].T)
    # Then: selected order, causal alignment and carried state exactly match step().
    assert features.shape == (2, 4, 3)
    assert torch.equal(features, torch.stack(repeated, dim=1))
    assert torch.equal(features, torch.cat((first, second), dim=1))
    assert torch.equal(final_state, state)
    assert torch.equal(chunked_final, final_state)


def test_explicit_port_checkpoint_requires_exact_manifest_and_indices(
    graph: Graph, tmp_path: Path
) -> None:
    # Given: a trained explicit-port learner with self-contained port provenance.
    config = ARConfig(
        alphabet_size=2,
        readout_neurons=2,
        port_policy="alpn_mbon",
        port_manifest_sha256="a" * 64,
        sensory_indices=(1, 3),
        readout_indices=(8, 9),
    )
    learner = ARLearner(graph, config)
    learner.train(np.array([0, 1] * 20, dtype=np.int64), 1)
    checkpoint = tmp_path / "explicit.npz"
    save_checkpoint(learner, checkpoint, "fixture")
    # When: the exact condition restores and a changed manifest attempts restoration.
    restored = ARLearner(graph, config)
    load_checkpoint(restored, checkpoint, "fixture")
    changed = ARLearner(
        graph,
        config.model_copy(update={"port_manifest_sha256": "b" * 64}),
    )
    # Then: exact ports resume while any provenance change is incompatible.
    assert restored.trace == learner.trace
    assert torch.equal(restored.model.sensory, learner.model.sensory)
    with pytest.raises(ValueError, match="compatibility"):
        load_checkpoint(changed, checkpoint, "fixture")


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
