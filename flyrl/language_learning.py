"""Batched local likelihood-ratio learning; no autograd or target gradients.

For intermediate steps, p=sigmoid(z W), h~Bernoulli(p), E += z (h-p)^T.
At the final step pi=softmax((z W)[output]), a~Categorical(pi), and
E[:,output] += z (one_hot(a)-pi)^T. Incoming sensory scores are zero.
W <- project(W + eta mean_batch[(1[a=y]-b_previous) E]).
The EMA baseline changes only AFTER the update. Scores sum across time,
not a decayed surrogate, so the estimator is the trajectory score gradient.
"""

from dataclasses import dataclass
from math import log

import numpy as np
import torch

from flyrl.connectome import Graph
from flyrl.language_data import IntVector, evaluation_starts
from flyrl.language_models import LanguageConfig, LanguageError, Metrics, Ports
from flyrl.language_runtime import configure_device, select_ports

__all__ = ["CharacterLearner", "LanguageConfig", "LanguageError"]


@dataclass(frozen=True, slots=True)
class Decision:
    """Sampled action, conditional probabilities, and local trajectory scores."""

    actions: torch.Tensor
    probabilities: torch.Tensor
    eligibility: torch.Tensor
    log_probabilities: torch.Tensor


@dataclass(frozen=True, slots=True)
class LearningStep:
    """Observed reward from decisions made before target comparison."""

    actions: torch.Tensor
    probabilities: torch.Tensor
    reward: float


class CharacterLearner:
    """Mutable update-boundary state with private window and policy RNGs.

    Dense masked matmul implements the sparse topology, on the requested device.
    Eligibility storage is O(batch*N*N), independent of context length.
    No global RNG is consumed; both persistent RNGs are checkpointed.
    """

    graph: Graph
    config: LanguageConfig
    device: torch.device
    ports: Ports
    sensory: torch.Tensor
    output: torch.Tensor
    weights: torch.Tensor
    signs: torch.Tensor
    plastic: torch.Tensor
    policy_rng: torch.Generator
    window_rng: torch.Generator
    baseline: float
    updates: int
    rewards: list[float]

    def __init__(self, graph: Graph, config: LanguageConfig) -> None:
        """Initialize source-normalized, sign-preserving graph weights."""
        self.graph, self.config = graph, config
        self.device = configure_device(config.device)
        nodes = len(graph.node_ids)
        self.ports = select_ports(nodes, config.alphabet_size, config.seed)
        self.sensory = torch.tensor(self.ports.sensory, device=self.device)
        self.output = torch.tensor(self.ports.output, device=self.device)
        outgoing = np.bincount(
            graph.source, weights=np.abs(graph.weight), minlength=nodes
        )
        dense = np.zeros((nodes, nodes), dtype=np.float32)
        dense[graph.source, graph.target] = (
            config.initial_gain * graph.weight / outgoing[graph.source]
        )
        self.weights = torch.tensor(dense, device=self.device)
        self.signs = self.weights.sign()
        self.plastic = self.signs.ne(0).to(torch.float32)
        self.plastic[:, self.sensory] = 0
        self.policy_rng = torch.Generator(device=self.device).manual_seed(
            config.seed + 23
        )
        self.window_rng = torch.Generator(device=self.device).manual_seed(
            config.seed + 37
        )
        self.baseline = 1 / config.alphabet_size
        self.updates = 0
        self.rewards = []

    def rollout(self, contexts: torch.Tensor, rng: torch.Generator) -> Decision:
        """Reset and sample teacher-forced contexts; no target is accepted here."""
        batch, length = contexts.shape
        nodes = self.weights.shape[0]
        state = torch.zeros((batch, nodes), device=self.device)
        eligibility = torch.zeros((batch, nodes, nodes), device=self.device)
        rows = torch.arange(batch, device=self.device)
        potential = torch.zeros_like(state)
        for step in range(length):
            state[:, self.sensory] = 0
            state[rows, self.sensory[contexts[:, step]]] = 1
            potential = state @ self.weights
            if step + 1 < length:
                probability = potential.sigmoid()
                post = torch.bernoulli(probability, generator=rng)
                residual = post - probability
                residual[:, self.sensory] = 0
                eligibility += state.unsqueeze(2) * residual.unsqueeze(1)
                state = post
        log_probabilities = potential[:, self.output].log_softmax(dim=1)
        probabilities = log_probabilities.exp()
        actions = torch.multinomial(probabilities, 1, generator=rng).squeeze(1)
        score = -probabilities.clone()
        score[rows, actions] += 1
        eligibility[:, :, self.output] += state.unsqueeze(2) * score.unsqueeze(1)
        eligibility *= self.plastic
        return Decision(actions, probabilities, eligibility, log_probabilities)

    def learn(self, contexts: torch.Tensor, targets: torch.Tensor) -> LearningStep:
        """Compare sampled actions with labels only to compute terminal correctness."""
        decision = self.rollout(contexts, self.policy_rng)
        rewards = decision.actions.eq(targets).to(torch.float32)
        reward = float(rewards.mean().item())
        if not self.config.frozen:
            advantage = rewards - self.baseline
            update = (advantage[:, None, None] * decision.eligibility).mean(dim=0)
            proposed = self.weights + self.config.learning_rate * update
            projected = self.signs * (self.signs * proposed).clamp(
                min=1e-9, max=self.config.max_weight
            )
            self.weights = torch.where(self.plastic.bool(), projected, self.weights)
        self.baseline += self.config.baseline_rate * (reward - self.baseline)
        self.rewards.append(reward)
        self.updates += 1
        return LearningStep(decision.actions, decision.probabilities, reward)

    def train(self, tokens: IntVector, updates: int) -> None:
        """Draw contiguous context/target windows exclusively from this split."""
        if updates < 0 or tokens.size <= self.config.context:
            raise LanguageError(reason="need nonnegative updates and context+1 tokens")
        if (tokens < 0).any() or (tokens >= self.config.alphabet_size).any():
            raise LanguageError(reason="token outside the configured alphabet")
        data = torch.tensor(tokens.copy(), device=self.device)
        offsets = torch.arange(self.config.context, device=self.device)
        for _ in range(updates):
            starts = torch.randint(
                tokens.size - self.config.context,
                (self.config.batch_size,),
                device=self.device,
                generator=self.window_rng,
            )
            contexts = data[starts[:, None] + offsets]
            _ = self.learn(contexts, data[starts + self.config.context])

    def evaluate(self, tokens: IntVector) -> Metrics:
        """Evaluate fixed disjoint windows without changing training RNG or state.

        NLL is mean conditional NLL over one seeded hidden trajectory per window,
        not the intractable marginal hidden-state language likelihood. Expected
        reward integrates the categorical action, sampled_reward samples it.
        """
        starts = evaluation_starts(
            tokens, self.config.context, self.config.eval_windows
        )
        rng = torch.Generator(device=self.device).manual_seed(self.config.seed + 47)
        data = torch.tensor(tokens.copy(), device=self.device)
        offsets = torch.arange(self.config.context, device=self.device)
        totals = torch.zeros(4, dtype=torch.float64, device=self.device)
        for chunk in range(0, starts.size, self.config.batch_size):
            indices = torch.tensor(
                starts[chunk : chunk + self.config.batch_size].copy(),
                device=self.device,
            )
            result = self.rollout(data[indices[:, None] + offsets], rng)
            targets = data[indices + self.config.context]
            probability = result.probabilities.gather(1, targets[:, None]).squeeze(1)
            totals += torch.stack(
                (
                    result.probabilities.argmax(dim=1).eq(targets).sum(),
                    probability.double().sum(),
                    result.actions.eq(targets).sum(),
                    -result.log_probabilities.gather(1, targets[:, None])
                    .double()
                    .sum(),
                )
            )
        values = totals / starts.size
        return Metrics(
            windows=int(starts.size),
            greedy_accuracy=float(values[0].item()),
            expected_reward=float(values[1].item()),
            sampled_reward=float(values[2].item()),
            nll=float(values[3].item()),
            bits_per_character=float(values[3].item()) / log(2),
        )

    def generate(self, prompt: tuple[int, ...], length: int) -> tuple[int, ...]:
        """Autoregress with generated input, resetting over the sliding context.

        This is the same recurrent within-window policy used in evaluation, not
        a learned decoder. Sampling uses a separate reproducible RNG stream.
        """
        if not prompt or length < 0:
            raise LanguageError(
                reason="generation needs a prompt and nonnegative length"
            )
        rng = torch.Generator(device=self.device).manual_seed(self.config.seed + 59)
        continuation = list(prompt)
        for _ in range(length):
            context = torch.tensor(
                [continuation[-self.config.context :]], device=self.device
            )
            continuation.append(int(self.rollout(context, rng).actions.item()))
        return tuple(continuation)
