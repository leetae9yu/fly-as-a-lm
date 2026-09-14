"""Anatomical recurrence: h' = (1-leak)h + leak*tanh(W(h+code) + bias + code).

Codes precede propagation even on token one; they are fixed unless opted into
learning. Edges, biases and readout learn. Sensory/readout ports are disjoint and
sampled without edges. Imported strengths (normally log1p synapses) receive random
functional signs and original-target L1 normalization, bounding initial real gain
by one. Shuffling target stubs preserves directed degrees and EXACT edge-order
initial weights without renormalizing; parallel edges and self-loops are allowed.
"""

from math import sqrt
from typing import Final, Protocol, TypeAlias, assert_never

import numpy as np
import torch
from numpy.typing import NDArray
from typing_extensions import override

from flyrl.ar_config import ARConfig
from flyrl.ar_prepared import PreparedRecurrence, prepare_recurrence
from flyrl.ar_sparse import sparse_recur, sparse_recur_prepared
from flyrl.ar_topology import SparseTopology
from flyrl.connectome import Graph

BPE_SENSORY_NEURONS: Final = 192
IndexArray: TypeAlias = NDArray[np.int64]


class OutgoingIntervention(Protocol):
    """Read-only calibrated source signals, independent of evaluation machinery."""

    @property
    def indices(self) -> tuple[int, ...]:
        """Return the distinct outgoing source neuron indices."""
        ...

    @property
    def training_mean(self) -> tuple[float, ...]:
        """Return the full-node old-training mean vector."""
        ...


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


def _port_indices(
    config: ARConfig,
    nodes: int,
    sensory_budget: int,
    order: IndexArray,
) -> tuple[IndexArray, IndexArray]:
    """Resolve legacy or explicit disjoint ports without changing RNG consumption."""
    match config.port_policy:
        case "legacy_random":
            sensory_count = min(nodes // 2, sensory_budget)
            head_count = min(config.readout_neurons, nodes - sensory_count)
            return order[:sensory_count], order[-head_count:]
        case "alpn_mbon" | "alpn_random" | "random_mbon" | "random_random":
            if config.sensory_indices is None or config.readout_indices is None:
                message = "Explicit port configuration is incomplete"
                raise ValueError(message)
            sensory = np.asarray(config.sensory_indices, dtype=np.int64)
            readout = np.asarray(config.readout_indices, dtype=np.int64)
            if (
                sensory.size + readout.size > nodes
                or bool((sensory >= nodes).any())
                or bool((readout >= nodes).any())
            ):
                message = "Explicit port index exceeds graph bounds"
                raise ValueError(message)
            return sensory, readout
        case _:
            assert_never(config.port_policy)


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
        sensory_indices, readout_indices = _port_indices(
            config, self.nodes, sensory_budget, order
        )
        sensory_count = sensory_indices.size
        head_count = readout_indices.size
        self.register_buffer("sensory", torch.tensor(sensory_indices))
        self.register_buffer("ports", torch.tensor(readout_indices))
        codes = 2 * rng.integers(2, size=(config.alphabet_size, sensory_count)) - 1
        code_tensor = torch.tensor(codes, dtype=torch.float32)
        if config.trainable_codes:
            self.codes = torch.nn.Parameter(code_tensor)
        else:
            self.register_buffer("codes", code_tensor)
        strength = np.abs(graph.weight)
        fan_in = np.bincount(graph.target, weights=strength, minlength=self.nodes)
        initial = config.initial_gain * strength / fan_in[graph.target]
        initial *= 2 * rng.integers(2, size=graph.source.size) - 1
        target = graph.target.copy()
        if config.control == "shuffled":
            np.random.default_rng(config.seed + 104729).shuffle(target)
        self.topology = SparseTopology(
            torch.stack((torch.tensor(target), torch.tensor(graph.source.copy()))),
            nodes=self.nodes,
        )
        self.sensory = self.get_buffer("sensory")
        self.ports = self.get_buffer("ports")
        if not config.trainable_codes:
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
        """Consume the current token and project its state for online generation."""
        state = self._advance(tokens, state, zero_recurrent=zero_recurrent)
        return state[self.ports].T @ self.readout + self.output_bias, state

    @torch.no_grad()
    def selected_states(
        self,
        tokens: torch.Tensor,
        indices: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        prepared: PreparedRecurrence | None = None,
        intervention: OutgoingIntervention | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Record states; only state=None starts a story, even for a zero prefix."""
        outgoing: tuple[torch.Tensor, torch.Tensor] | None = None
        if intervention is not None:
            if len(intervention.training_mean) != self.nodes:
                message = "Training mean must contain one value per model neuron"
                raise ValueError(message)
            if intervention.indices:
                group = torch.tensor(intervention.indices, device=self.weight.device)
                mean = self.weight.new_tensor(intervention.training_mean)[group, None]
                outgoing = group, mean
        current = (
            self.weight.new_zeros((self.nodes, tokens.shape[0]))
            if state is None
            else state
        )
        recurrence = (
            prepare_recurrence(self.weight, self.topology, need_reverse=False)
            if prepared is None
            else prepared
        )
        selected: list[torch.Tensor] = []
        for position, token in enumerate(tokens.unbind(dim=1)):
            current = self._advance(
                token,
                current,
                zero_recurrent=False,
                prepared=recurrence,
                outgoing=outgoing if state is not None or position > 0 else None,
            )
            selected.append(current[indices].T)
        return torch.stack(selected, dim=1), current

    def _advance(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor,
        *,
        zero_recurrent: bool,
        prepared: PreparedRecurrence | None = None,
        outgoing: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Update local state without modifying the history used by the leak term."""
        drive = torch.zeros_like(state)
        drive[self.sensory] = self.codes[tokens].T
        signal = state + drive
        if outgoing is not None:
            group, mean = outgoing
            signal[group] = mean + drive[group]
        if zero_recurrent:
            recurrent = torch.zeros_like(state)
        elif prepared is None:
            recurrent = sparse_recur(
                self.weight, signal, self.topology, self.config.edge_chunk
            )
        else:
            recurrent = sparse_recur_prepared(
                self.weight,
                signal,
                prepared,
                self.topology,
                self.config.edge_chunk,
            )
        return (1 - self.config.leak) * state + self.config.leak * torch.tanh(
            recurrent + self.bias[:, None] + drive
        )

    @override
    def forward(
        self, tokens: torch.Tensor, *, zero_recurrent: bool = False
    ) -> torch.Tensor:
        """Project all causal readout states together as one dense matrix product."""
        state = self.weight.new_zeros((self.nodes, tokens.shape[0]))
        readouts: list[torch.Tensor] = []
        prepared = (
            None
            if zero_recurrent
            else prepare_recurrence(
                self.weight,
                self.topology,
                need_reverse=torch.is_grad_enabled(),
            )
        )
        for character in tokens.unbind(dim=1):
            state = self._advance(
                character,
                state,
                zero_recurrent=zero_recurrent,
                prepared=prepared,
            )
            readouts.append(state[self.ports].T)
        return torch.stack(readouts, dim=1) @ self.readout + self.output_bias
