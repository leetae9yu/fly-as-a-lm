"""Reward-only language behavior on explicitly synthetic, nonbiological graphs."""

from pathlib import Path

import numpy as np
import pytest
import torch

from flyrl import language_learning as learning
from flyrl.connectome import Graph, shuffled_graph
from flyrl.language_checkpoint import load_checkpoint, save_checkpoint


@pytest.fixture
def graph() -> Graph:
    # Complete signed synthetic graph makes every fixed port assignment reachable.
    source, target = np.nonzero(np.ones((7, 7)) - np.eye(7))
    weight = np.ones(source.size, dtype=np.float64)
    weight[source == 6] = -1
    return Graph(
        tuple(f"synthetic:{i}" for i in range(7)),
        source.astype(np.int64),
        target.astype(np.int64),
        weight,
        "synthetic test circuit; not biological",
    )


def test_ports_and_initialization_when_topology_changes(graph: Graph) -> None:
    # Given paired real/shuffled controls with edge-independent assignments.
    config = learning.LanguageConfig(alphabet_size=2, seed=4)
    sparse = Graph(
        graph.node_ids,
        np.arange(7, dtype=np.int64),
        np.array([1, 2, 3, 4, 5, 6, 0], dtype=np.int64),
        np.arange(1, 8, dtype=np.float64),
        graph.provenance,
    )
    real = learning.CharacterLearner(sparse, config)
    control_graph = shuffled_graph(sparse, 2)
    # When another topology initializes.
    control = learning.CharacterLearner(control_graph, config)
    # Then roles and source-normalized edge weights match exactly.
    assert any(
        sparse.target.item(i) != control_graph.target.item(i)
        for i in range(sparse.target.size)
    )
    assert real.ports == control.ports
    assert set(real.ports.sensory).isdisjoint(real.ports.output)
    assert torch.equal(
        real.weights[sparse.source.copy(), sparse.target.copy()],
        control.weights[control_graph.source.copy(), control_graph.target.copy()],
    )


def test_invariants_when_reward_updates_signed_graph(graph: Graph) -> None:
    # Given a signed masked network and clamped sensory columns.
    learner = learning.CharacterLearner(
        graph, learning.LanguageConfig(alphabet_size=2, context=3, batch_size=16)
    )
    original = learner.weights.clone()
    # When terminal rewards drive learning.
    learner.train(np.array([0, 1] * 100, dtype=np.int64), 10)
    # Then topology, signs, bounds, and sensory incoming weights are invariant.
    assert torch.equal(torch.sign(learner.weights), torch.sign(original))
    assert torch.all(learner.weights.abs() <= learner.config.max_weight)
    assert torch.equal(
        learner.weights[:, learner.ports.sensory],
        original[:, learner.ports.sensory],
    )
    assert not learner.weights.requires_grad


def test_frozen_when_training_runs(graph: Graph) -> None:
    # Given disabled plasticity.
    learner = learning.CharacterLearner(
        graph, learning.LanguageConfig(alphabet_size=2, frozen=True, context=2)
    )
    original = learner.weights.clone()
    # When training runs.
    learner.train(np.array([0, 1] * 50, dtype=np.int64), 4)
    # Then weights remain bit-identical although progress advances.
    assert torch.equal(original, learner.weights)
    assert learner.updates == 4


def test_predictions_when_only_targets_change(graph: Graph) -> None:
    # Given paired models, identical contexts, and opposite target labels.
    config = learning.LanguageConfig(alphabet_size=2, context=2, batch_size=32)
    first = learning.CharacterLearner(graph, config)
    second = learning.CharacterLearner(graph, config)
    contexts = torch.zeros((32, 2), dtype=torch.int64)
    # When one reward step sees different labels after the decision.
    a = first.learn(contexts, torch.zeros(32, dtype=torch.int64))
    b = second.learn(contexts, torch.ones(32, dtype=torch.int64))
    # Then sampled decisions and probabilities are identical, rewards complementary.
    assert torch.equal(a.actions, b.actions)
    assert torch.equal(a.probabilities, b.probabilities)
    assert a.reward + b.reward == pytest.approx(1.0)
    assert not torch.equal(first.weights, second.weights)


def test_learning_when_tiny_text_has_deterministic_next_character(graph: Graph) -> None:
    # Given alternating synthetic text, equally frequent symbols, and one-step input.
    config = learning.LanguageConfig(
        alphabet_size=2,
        context=1,
        batch_size=64,
        learning_rate=0.3,
        seed=7,
    )
    learner = learning.CharacterLearner(graph, config)
    text = np.array([0, 1] * 256, dtype=np.int64)
    before = learner.evaluate(text)
    # When reward alone trains next-character decisions.
    learner.train(text, 160)
    after = learner.evaluate(text)
    # Then conditional reward beats both initialization and the unigram ceiling.
    assert after.expected_reward > before.expected_reward + 0.25
    assert after.expected_reward > 0.8
    assert after.greedy_accuracy == 1.0


def test_evaluation_when_inserted_between_updates(graph: Graph) -> None:
    # Given independent equal training streams.
    config = learning.LanguageConfig(alphabet_size=2, context=3, batch_size=8)
    first = learning.CharacterLearner(graph, config)
    second = learning.CharacterLearner(graph, config)
    train = np.array([0, 1] * 64, dtype=np.int64)
    heldout = np.zeros(100, dtype=np.int64)
    # When heldout evaluation is inserted for one learner.
    metrics = first.evaluate(heldout)
    assert metrics == first.evaluate(heldout)
    first.train(train, 3)
    second.train(train, 3)
    # Then no heldout data, weights, RNG, or progress leak into training.
    assert torch.equal(first.weights, second.weights)
    assert torch.equal(first.policy_rng.get_state(), second.policy_rng.get_state())
    assert first.rewards == second.rewards


def test_resume_when_checkpoint_is_loaded(graph: Graph, tmp_path: Path) -> None:
    # Given an interrupted run and uninterrupted reference.
    config = learning.LanguageConfig(alphabet_size=2, context=3, batch_size=8)
    full = learning.CharacterLearner(graph, config)
    partial = learning.CharacterLearner(graph, config)
    text = np.array([0, 1] * 64, dtype=np.int64)
    full.train(text, 9)
    partial.train(text, 4)
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(partial, path, "corpus-fixture")
    # When a fresh learner resumes at an update boundary.
    resumed = learning.CharacterLearner(graph, config)
    load_checkpoint(resumed, path, "corpus-fixture")
    resumed.train(text, 5)
    # Then all mutable training state matches bit-for-bit.
    assert torch.equal(full.weights, resumed.weights)
    assert torch.equal(full.policy_rng.get_state(), resumed.policy_rng.get_state())
    assert torch.equal(full.window_rng.get_state(), resumed.window_rng.get_state())
    assert full.rewards == resumed.rewards
    assert full.baseline == resumed.baseline
    assert full.updates == resumed.updates


def test_checkpoint_when_corpus_is_incompatible(graph: Graph, tmp_path: Path) -> None:
    # Given a valid checkpoint for another corpus.
    learner = learning.CharacterLearner(graph, learning.LanguageConfig(alphabet_size=2))
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(learner, path, "original")
    # When the checkpoint is requested with a different corpus.
    # Then compatibility validation rejects it before mutation.
    with pytest.raises(learning.LanguageError, match="compatib"):
        load_checkpoint(learner, path, "different")


def test_checkpoint_when_file_is_malformed(graph: Graph, tmp_path: Path) -> None:
    # Given an invalid checkpoint container.
    learner = learning.CharacterLearner(graph, learning.LanguageConfig(alphabet_size=2))
    path = tmp_path / "checkpoint.npz"
    _ = path.write_bytes(b"not a checkpoint")
    # When parsing it.
    # Then it fails clearly instead of executing pickle or silently resetting.
    with pytest.raises(learning.LanguageError):
        load_checkpoint(learner, path, "fixture")


def test_cuda_when_unavailable(graph: Graph, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a host with unavailable CUDA (the availability boundary only is mocked).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # When GPU execution is explicitly requested.
    # Then CPU fallback is forbidden.
    with pytest.raises(learning.LanguageError, match="CUDA"):
        _ = learning.CharacterLearner(
            graph, learning.LanguageConfig(alphabet_size=2, device="cuda")
        )


def test_nll_when_bounded_weights_make_action_probability_underflow() -> None:
    # Given a saturated but valid 40-neuron positive synthetic circuit.
    source, target = np.nonzero(np.ones((40, 40)) - np.eye(40))
    circuit = Graph(
        tuple(f"synthetic:{i}" for i in range(40)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size, dtype=np.float64),
        "synthetic numeric stability fixture",
    )
    learner = learning.CharacterLearner(
        circuit,
        learning.LanguageConfig(
            alphabet_size=2,
            context=2,
            eval_windows=4,
        ),
    )
    learner.weights = 4 * learner.signs
    learner.weights[:, learner.ports.output[1]] *= 1e-9
    # When targets choose the extremely unlikely category.
    metrics = learner.evaluate(np.ones(30, dtype=np.int64))
    # Then stable log-softmax retains finite NLL instead of log(underflowed zero).
    assert np.isfinite(metrics.nll)
    assert metrics.nll > 100


def test_checkpoint_when_frozen_weights_were_changed(
    graph: Graph, tmp_path: Path
) -> None:
    # Given a checkpoint whose frozen weights changed while retaining their signs.
    config = learning.LanguageConfig(alphabet_size=2, frozen=True)
    damaged = learning.CharacterLearner(graph, config)
    damaged.weights *= 1 + damaged.plastic
    path = tmp_path / "frozen.npz"
    save_checkpoint(damaged, path, "fixture")
    fresh = learning.CharacterLearner(graph, config)
    # When a fresh frozen learner loads it.
    # Then valid checksums cannot conceal a violation of the frozen condition.
    with pytest.raises(learning.LanguageError):
        load_checkpoint(fresh, path, "fixture")


def test_eligibility_when_hidden_trajectory_is_sampled(graph: Graph) -> None:
    # Given one two-character trajectory, replay only its first hidden sample.
    learner = learning.CharacterLearner(graph, learning.LanguageConfig(alphabet_size=2))
    contexts = torch.tensor([[0, 1]], dtype=torch.int64)
    replay = torch.Generator().manual_seed(42)
    first = torch.zeros((1, 7))
    first[0, learner.ports.sensory[0]] = 1
    hidden = torch.bernoulli((first @ learner.weights).sigmoid(), generator=replay)
    last = hidden.clone()
    last[:, learner.sensory] = 0
    last[0, learner.ports.sensory[1]] = 1
    recurrent = torch.ones(7, dtype=torch.bool)
    recurrent[learner.sensory] = False
    # When the actual engine collects its local score for that same trajectory.
    decision = learner.rollout(contexts, torch.Generator().manual_seed(42))

    def trajectory_log_probability(weights: torch.Tensor) -> float:
        probability = (first.double() @ weights).sigmoid()
        log_hidden = hidden * probability.log() + (1 - hidden) * (1 - probability).log()
        log_action = (last.double() @ weights)[:, learner.output].log_softmax(dim=1)
        return float(
            (
                log_hidden[:, recurrent].sum()
                + log_action[0, int(decision.actions.item())]
            ).item()
        )

    # Then each permitted shared-edge score equals a finite-difference likelihood
    # derivative with the observed hidden path and action held fixed, not labels.
    delta = 1e-5
    numerical = torch.zeros_like(learner.weights)
    for source in range(7):
        for target in range(7):
            plus, minus = learner.weights.double(), learner.weights.double()
            plus[source, target] += delta
            minus[source, target] -= delta
            numerical[source, target] = (
                trajectory_log_probability(plus) - trajectory_log_probability(minus)
            ) / (2 * delta)
    assert torch.allclose(
        decision.eligibility[0],
        numerical * learner.plastic,
        atol=1e-6,
        rtol=1e-5,
    )
