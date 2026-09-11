"""Sequence projection must be batched without changing the anatomical recurrence."""

from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pytest
import torch
from pydantic import ConfigDict, TypeAdapter

from flyrl.ar_config import ARConfig
from flyrl.ar_model import ConnectomeLM
from flyrl.connectome import Graph


@runtime_checkable
class _ProfilerEvent(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def count(self) -> int: ...


@pytest.fixture
def graph() -> Graph:
    source, target = np.nonzero(np.ones((12, 12)) - np.eye(12))
    return Graph(
        tuple(f"synthetic:{i}" for i in range(12)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size, dtype=np.float64),
        "synthetic readout equivalence fixture",
    )


def test_sequence_uses_one_dense_readout_projection(graph: Graph) -> None:
    # Given: four timesteps and a distinct vocabulary/readout shape.
    model = ConnectomeLM(graph, ARConfig(alphabet_size=7, readout_neurons=5))
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    # When: executing an actual sequence forward pass under the CPU profiler.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True
    ) as trace:
        _ = model.forward(tokens)
    # Then: dense matrix multiplication is batched instead of launched per token.
    events = TypeAdapter(
        list[_ProfilerEvent], config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(trace.key_averages())
    projections = sum(event.count for event in events if event.key == "aten::mm")
    assert projections == 1


@pytest.mark.parametrize("control", ["real", "shuffled", "frozen"])
@pytest.mark.parametrize("zero_recurrent", [False, True])
def test_sequence_logits_and_gradients_match_dense_recurrence(
    graph: Graph,
    control: Literal["real", "shuffled", "frozen"],
    *,
    zero_recurrent: bool,
) -> None:
    # Given: a double precision circuit and an independent dense equation reference.
    model = ConnectomeLM(
        graph,
        ARConfig(
            alphabet_size=7, readout_neurons=5, trainable_codes=True, control=control
        ),
    ).double()
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    dense = model.weight.new_zeros((12, 12)).index_put(
        (model.edges[0], model.edges[1]), model.weight, accumulate=True
    )
    state = model.weight.new_zeros((12, 2))
    outputs: list[torch.Tensor] = []
    for token in tokens.unbind(dim=1):
        drive = torch.zeros_like(state)
        drive[model.sensory] = model.codes[token].T
        recurrence = (
            torch.zeros_like(state) if zero_recurrent else dense @ (state + drive)
        )
        state = (1 - model.config.leak) * state + model.config.leak * torch.tanh(
            recurrence + model.bias[:, None] + drive
        )
        outputs.append(state[model.ports].T @ model.readout + model.output_bias)
    expected = torch.stack(outputs, dim=1)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    expected_grads = torch.autograd.grad(
        expected.square().mean(), parameters, materialize_grads=True
    )
    # When: evaluating the sequence implementation and its derivatives.
    actual = model.forward(tokens, zero_recurrent=zero_recurrent)
    actual_grads = torch.autograd.grad(
        actual.square().mean(), parameters, materialize_grads=True
    )
    # Then: every logit and trained-parameter derivative matches the equation.
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    for left, right in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)
