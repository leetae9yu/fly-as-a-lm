"""Continuous anatomical recurrence with fixed sensory codes and a small readout.

h' = (1-leak) h + leak tanh(W_anatomy (h + code) + bias + code).
The fixed sensory code is injected before anatomical propagation, so even the
first next-token prediction can use its current input.
Only existing edges, neuron biases and the linear readout are trainable. The
sensory and readout neurons are disjoint, sampled without looking at edges.
Graph weights are importer-provided strengths (normally log1p synapse counts).
Random functional signs avoid an unsupported all-excitatory assumption. Original
target L1 fan-in normalization bounds the initial real graph infinity norm by
initial_gain <= 1. Shuffled edges retain EXACTLY these edge-order initial weights,
not a renormalization that would change the paired initialization distribution.
The shuffled condition permutes target stubs, preserving directed degrees but
allowing parallel edges and self-loops: a directed configuration multigraph, not
the slower simple-graph double-edge-swap control used by the old RL experiment.
"""

from math import sqrt
from typing import Final, assert_never

import numpy as np
import torch
from typing_extensions import override

from flyrl.ar_config import ARConfig
from flyrl.ar_sparse import sparse_recur
from flyrl.ar_topology import SparseTopology
from flyrl.connectome import Graph

BPE_SENSORY_NEURONS: Final = 192


def requested_device(requested: str) -> torch.device:
    """Validate the execution boundary; an unavailable GPU is an error."""
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        message = "Only cpu and cuda devices are supported"
        raise ValueError(message)
    if device.type == "cuda" and not torch.cuda.is_available():
        message = "CUDA requested but unavailable; CPU fallback is forbidden"
        raise ValueError(message)
    return device


class ConnectomeLM(torch.nn.Module):
    """Strictly causal next-token logits, with zero state per training window."""

    config: ARConfig
    nodes: int
    topology: SparseTopology
    sensory: torch.Tensor
    ports: torch.Tensor
    codes: torch.Tensor
    weight: torch.nn.Parameter
    bias: torch.nn.Parameter
    readout: torch.nn.Parameter
    output_bias: torch.nn.Parameter

    def __init__(self, graph: Graph, config: ARConfig) -> None:
        """Build O(E + N + vocabulary*ports) tensors with private initialization RNG."""
        super().__init__()
        self.config, self.nodes = config, len(graph.node_ids)
        device = requested_device(config.device)
        rng = np.random.default_rng(config.seed)
        order = np.arange(self.nodes, dtype=np.int64)
        rng.shuffle(order)
        match config.tokenization:
            case "character":
                sensory_budget = config.alphabet_size * 4
            case "bpe":
                sensory_budget = BPE_SENSORY_NEURONS
            case _:
                assert_never(config.tokenization)
        sensory_count = min(self.nodes // 2, sensory_budget)
        head_count = min(config.readout_neurons, self.nodes - sensory_count)
        self.register_buffer("sensory", torch.tensor(order[:sensory_count]))
        self.register_buffer("ports", torch.tensor(order[-head_count:]))
        codes = 2 * rng.integers(2, size=(config.alphabet_size, sensory_count)) - 1
        self.register_buffer("codes", torch.tensor(codes, dtype=torch.float32))
        strength = np.abs(graph.weight)
        fan_in = np.bincount(graph.target, weights=strength, minlength=self.nodes)
        initial = config.initial_gain * strength / fan_in[graph.target]
        initial *= 2 * rng.integers(2, size=graph.source.size) - 1
        target = graph.target.copy()
        if config.control == "shuffled":
            np.random.default_rng(config.seed + 104729).shuffle(target)
        self.topology = SparseTopology(
            torch.stack((torch.tensor(target), torch.tensor(graph.source.copy()))),
        )
        self.sensory = self.get_buffer("sensory")
        self.ports = self.get_buffer("ports")
        self.codes = self.get_buffer("codes")
        train_core = config.control != "frozen"
        self.weight = torch.nn.Parameter(
            torch.tensor(initial, dtype=torch.float32), requires_grad=train_core
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(self.nodes), requires_grad=train_core
        )
        self.readout = torch.nn.Parameter(
            torch.tensor(
                rng.normal(
                    0, 0.1 / sqrt(head_count), (head_count, config.alphabet_size)
                ),
                dtype=torch.float32,
            )
        )
        self.output_bias = torch.nn.Parameter(torch.zeros(config.alphabet_size))
        _ = self.to(device)

    @property
    def edges(self) -> torch.Tensor:
        """Expose original parameter-order edges without duplicating device storage."""
        return self.topology.edges

    def step(
        self, tokens: torch.Tensor, state: torch.Tensor, *, zero_recurrent: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume only the current character, returning its next-character logits."""
        drive = torch.zeros_like(state)
        drive[self.sensory] = self.codes[tokens].T
        recurrent = (
            torch.zeros_like(state)
            if zero_recurrent
            else sparse_recur(
                self.weight, state + drive, self.topology, self.config.edge_chunk
            )
        )
        state = (1 - self.config.leak) * state + self.config.leak * torch.tanh(
            recurrent + self.bias[:, None] + drive
        )
        return state[self.ports].T @ self.readout + self.output_bias, state

    @override
    def forward(
        self, tokens: torch.Tensor, *, zero_recurrent: bool = False
    ) -> torch.Tensor:
        """Map [batch, time] input tokens to [batch, time, vocabulary] logits."""
        state = self.weight.new_zeros((self.nodes, tokens.shape[0]))
        logits: list[torch.Tensor] = []
        for character in tokens.unbind(dim=1):
            output, state = self.step(character, state, zero_recurrent=zero_recurrent)
            logits.append(output)
        return torch.stack(logits, dim=1)
