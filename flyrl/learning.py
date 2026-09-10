"""Bernoulli recurrent neurons and an episodic local likelihood-ratio rule.

At each step p_j = sigmoid(b + sum_i w_ij z_i), z'_j ~ Bernoulli(p_j).
The sensory neuron is clamped, so its incoming edges have zero score.
Eligibility sums z_i (z'_j - p_j); terminal reward uses the preceding
episode baseline. This is a score-function gradient, not backpropagation.
Projection keeps every original edge and sign; no bias or decoder is learned.
"""

from dataclasses import dataclass
from typing import TypeAlias, assert_never, overload

import numpy as np
from typing_extensions import override

from flyrl.connectome import Graph
from flyrl.models import Config, Control, Evaluation, Ports, Task

Vector: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]


@dataclass(frozen=True, slots=True)
class ExperimentError(ValueError):
    """Invalid experiment or checkpoint input."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the boundary failure reason."""
        return self.reason


@overload
def eligibility_increment(pre: float, post: float, probability: float) -> float: ...


@overload
def eligibility_increment(
    pre: Vector,
    post: Vector,
    probability: Vector,
) -> Vector: ...


def eligibility_increment(
    pre: float | Vector,
    post: float | Vector,
    probability: float | Vector,
) -> float | Vector:
    """Compute a local Bernoulli log-probability derivative."""
    return pre * (post - probability)


def select_ports(graph: Graph, seed: int) -> Ports:
    """Select an observed non-self edge, avoiding unrelated disconnected nodes.

    Selection is on the real graph before any control intervention. It imposes
    no anatomical cell-role claim and deliberately favors a one-hop task.
    """
    edges = [
        i
        for i, (source, target) in enumerate(
            zip(graph.source, graph.target, strict=True)
        )
        if source != target
    ]
    if not edges:
        raise ExperimentError(reason="a sensory/output pair needs a non-self edge")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    edge = edges[int(rng.integers(len(edges)))]
    return Ports(sensory=graph.source.item(edge), output=graph.target.item(edge))


class Learner:
    """Mutable episode-boundary state; neural state resets on every trial."""

    graph: Graph
    config: Config
    ports: Ports
    weights: Vector
    eligibility: Vector
    baseline: float
    rewards: list[int]
    rng: np.random.Generator

    def __init__(
        self,
        graph: Graph,
        config: Config,
        ports: Ports | None = None,
    ) -> None:
        """Initialize fixed graph-normalized weights and a training-only RNG."""
        self.graph = graph
        self.config = config
        self.ports = select_ports(graph, config.seed) if ports is None else ports
        if (
            max(self.ports.sensory, self.ports.output) >= len(graph.node_ids)
            or self.ports.sensory == self.ports.output
        ):
            raise ExperimentError(reason="ports must be distinct graph node indices")
        outgoing = np.bincount(
            graph.source,
            weights=np.abs(graph.weight),
            minlength=len(graph.node_ids),
        )
        # Source normalization preserves identical edge weights under rewiring.
        self.weights = config.initial_gain * graph.weight / outgoing[graph.source]
        self.eligibility = np.zeros_like(self.weights)
        self.baseline = 0.5
        self.rewards = []
        self.rng = np.random.default_rng(np.random.SeedSequence([config.seed, 23]))

    def stimulus(self, cue: int) -> Vector:
        """Encode binary cue; memory has delay+1 identical zero-input steps."""
        if cue not in (0, 1):
            raise ExperimentError(reason="cue must be binary")
        match self.config.task:
            case Task.ASSOCIATION:
                return np.full(2, cue, dtype=np.float64)
            case Task.MEMORY:
                stimulus = np.zeros(self.config.delay + 2, dtype=np.float64)
                stimulus[0] = cue
                return stimulus
            case _:
                assert_never(self.config.task)

    def rollout(
        self,
        cue: int,
        rng: np.random.Generator,
    ) -> tuple[int, Vector]:
        """Sample a trial without mutating any training state.

        Clamp sensory at time t, then synchronously sample other neurons at
        t+1. The output's final binary state is the action, without a decoder.
        """
        graph = self.graph
        state: Vector = np.zeros(len(graph.node_ids), dtype=np.float64)
        eligibility: Vector = np.zeros_like(self.weights)
        plastic = np.not_equal(graph.target, self.ports.sensory)
        for sensory in self.stimulus(cue):
            state[self.ports.sensory] = sensory
            potential = self.config.bias + np.bincount(
                graph.target,
                weights=self.weights * state[graph.source],
                minlength=len(graph.node_ids),
            )
            # Stable sigmoid, with no clipping of the score-function model.
            decay = np.exp(-np.abs(potential))
            probability: Vector = decay / (1 + decay)
            positive = potential >= 0
            probability[positive] = 1 / (1 + decay[positive])
            post: Vector = (rng.random(len(state)) < probability).astype(np.float64)
            eligibility += plastic * eligibility_increment(
                state[graph.source],
                post[graph.target],
                probability[graph.target],
            )
            state = post
        return int(state.item(self.ports.output)), eligibility

    def train(self, episodes: int) -> None:
        """Run additional episodes with terminal binary reward and local updates."""
        if episodes < 0:
            raise ExperimentError(reason="additional episode count must be nonnegative")
        for _ in range(episodes):
            cue = int(self.rng.integers(2))
            action, self.eligibility = self.rollout(cue, self.rng)
            reward = int(action == cue)
            if self.config.control != Control.FROZEN:
                update = self.config.learning_rate * (reward - self.baseline)
                signed = np.sign(self.graph.weight)
                proposed = signed * (self.weights + update * self.eligibility)
                self.weights = signed * np.minimum(
                    np.maximum(proposed, 1e-9), self.config.max_weight
                )
            self.baseline += self.config.baseline_rate * (reward - self.baseline)
            self.rewards.append(reward)

    def evaluate(self) -> Evaluation:
        """Evaluate a balanced held-out RNG stream reproducibly, without mutation.

        Deterministic refers to seeded evaluation, not thresholding neurons.
        Cues are exhaustive binary task conditions; randomness is held out,
        not previously unseen cue identities. Odd trial counts are rejected.
        """
        trials = self.config.eval_trials
        if trials % 2:
            raise ExperimentError(
                reason="evaluation trials must be even for balanced cues"
            )
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, 47]))
        correct = [0, 0]
        for index in range(trials):
            cue = index % 2
            action, _ = self.rollout(cue, rng)
            correct[cue] += int(action == cue)
        total = sum(correct)
        return Evaluation(
            correct=total,
            trials=trials,
            accuracy=total / trials,
            per_cue=(correct[0] * 2 / trials, correct[1] * 2 / trials),
        )
