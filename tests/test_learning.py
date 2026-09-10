"""Local score-rule regression tests using explicitly synthetic networks."""

import numpy as np
import pytest

from flyrl.connectome import Graph
from flyrl.learning import Learner, eligibility_increment
from flyrl.models import Config, Control, Ports, Task


def test_score_when_bernoulli_spike_is_observed() -> None:
    # Given a presynaptic spike and a sampled postsynaptic spike.
    pre, post, probability = 1.0, 1.0, 0.25
    # When the local score is computed.
    score = eligibility_increment(pre, post, probability)
    # Then it equals the analytic log-likelihood derivative.
    assert score == pytest.approx(0.75)


def test_score_when_presynaptic_neuron_is_silent() -> None:
    # Given no causal presynaptic input.
    # When the local score is computed.
    score = eligibility_increment(0.0, 1.0, 0.25)
    # Then the synapse has zero eligibility.
    assert score == 0.0


@pytest.fixture
def synthetic_graph() -> Graph:
    return Graph(
        ("synthetic:input", "synthetic:output", "synthetic:hidden"),
        np.array([0, 1, 2, 1], dtype=np.int64),
        np.array([1, 2, 1, 1], dtype=np.int64),
        np.array([1.0, 0.3, 0.3, 0.3], dtype=np.float64),
        "synthetic regression fixture; not biological data",
    )


def test_training_when_reward_is_available(synthetic_graph: Graph) -> None:
    # Given a direct sensory-output circuit and fixed evaluation stream.
    learner = Learner(synthetic_graph, Config(seed=7), Ports(sensory=0, output=1))
    before = learner.evaluate()
    original = learner.weights.copy()
    # When reward-modulated learning runs through the public API.
    learner.train(2500)
    after = learner.evaluate()
    # Then policy accuracy improves and only permitted synapses change.
    assert after.accuracy > before.accuracy + 0.10
    assert after.accuracy > 0.70
    assert np.not_equal(original, learner.weights).any()
    assert (learner.weights > 0).all()


def test_frozen_when_training_is_requested(synthetic_graph: Graph) -> None:
    # Given disabled plasticity.
    learner = Learner(synthetic_graph, Config(control=Control.FROZEN))
    original = learner.weights.copy()
    # When episodes execute.
    learner.train(50)
    # Then synaptic weights remain bit-identical.
    np.testing.assert_array_equal(original, learner.weights)


def test_evaluation_when_called_repeatedly(synthetic_graph: Graph) -> None:
    # Given identically initialized training streams.
    evaluated = Learner(synthetic_graph, Config(seed=12))
    untouched = Learner(synthetic_graph, Config(seed=12))
    # When held-out evaluation is inserted between training calls.
    first = evaluated.evaluate()
    second = evaluated.evaluate()
    evaluated.train(20)
    untouched.train(20)
    # Then evaluation is deterministic and cannot perturb training.
    assert first == second
    np.testing.assert_array_equal(evaluated.weights, untouched.weights)
    assert evaluated.rewards == untouched.rewards
    assert evaluated.rng.bit_generator.state == untouched.rng.bit_generator.state


def test_memory_when_cues_differ(synthetic_graph: Graph) -> None:
    # Given delayed memory trials with opposite cues.
    learner = Learner(synthetic_graph, Config(task=Task.MEMORY, delay=4))
    # When fixed sensory sequences are produced.
    zero, one = learner.stimulus(0), learner.stimulus(1)
    # Then all post-cue inputs, including terminal input, are identical.
    assert zero[0] != one[0]
    np.testing.assert_array_equal(zero[1:], one[1:])
    assert len(zero) == 6


def test_signed_synapses_when_learning_runs(synthetic_graph: Graph) -> None:
    # Given an explicitly synthetic inhibitory synapse.
    signed = synthetic_graph.weight.copy()
    signed[1] = -0.3
    graph = Graph(
        synthetic_graph.node_ids,
        synthetic_graph.source,
        synthetic_graph.target,
        signed,
        synthetic_graph.provenance,
    )
    learner = Learner(graph, Config())
    # When rewards update synapses.
    learner.train(100)
    # Then no initial sign can change.
    np.testing.assert_array_equal(np.sign(learner.weights), np.sign(signed))
