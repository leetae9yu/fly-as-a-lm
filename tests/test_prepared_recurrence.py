"""Sequence-scoped sparse values must preserve original parameter semantics."""

from typing import Protocol

import numpy as np
import pytest
import torch

from flyrl.ar_config import ARConfig
from flyrl.ar_model import ConnectomeLM
from flyrl.ar_prepared import prepare_recurrence
from flyrl.ar_sparse import sparse_recur_prepared
from flyrl.ar_topology import SparseTopology
from flyrl.connectome import Graph


class _Prepare(Protocol):
    def __call__(
        self, topology: SparseTopology, weights: torch.Tensor, /
    ) -> torch.Tensor: ...


def test_topology_csr_layout_includes_empty_and_trailing_rows() -> None:
    # Given: unordered edges with a duplicate, an empty middle row and trailing node.
    topology = SparseTopology(
        torch.tensor([[2, 0, 2, 0], [0, 3, 0, 3]]),
        nodes=5,
    )
    # When: reading its canonical forward and reverse CSR layouts.
    forward = topology.csr_layout()
    reverse = topology.csr_layout(transpose=True)
    # Then: row pointers cover every node and columns follow coalesced edge order.
    assert torch.equal(forward[0], torch.tensor([0, 1, 1, 2, 2, 2]))
    assert torch.equal(forward[1], torch.tensor([3, 0]))
    assert torch.equal(reverse[0], torch.tensor([0, 1, 1, 1, 2, 2]))
    assert torch.equal(reverse[1], torch.tensor([2, 0]))
    assert (
        not {
            "forward_crow",
            "forward_col",
            "reverse_crow",
            "reverse_col",
        }
        & topology.state_dict().keys()
    )


@pytest.mark.parametrize("duplicate", [False, True])
def test_prepared_recurrence_preserves_multistep_gradients(duplicate: bool) -> None:
    # Given: a cyclic graph with optional parallel parameters on the same pair.
    edges = torch.tensor(
        [[0, 1, 2, 0 if duplicate else 2], [1, 2, 0, 1 if duplicate else 2]]
    )
    topology = SparseTopology(edges, nodes=4)
    weights = torch.tensor([0.3, -0.2, 0.7, 0.1], dtype=torch.float64)
    state = torch.arange(8, dtype=torch.float64).reshape(4, 2) / 7
    _ = weights.requires_grad_()
    _ = state.requires_grad_()
    prepared = prepare_recurrence(weights, topology, need_reverse=True)
    # When: reusing one immutable preparation over three recurrent timesteps.
    actual = state
    for _ in range(3):
        actual = sparse_recur_prepared(weights, actual, prepared, topology, 2)
    # Then: outputs and original parameter derivatives match a dense recurrence.
    dense = torch.zeros((4, 4), dtype=torch.float64).index_put(
        (edges[0], edges[1]), weights, accumulate=True
    )
    expected = state
    for _ in range(3):
        expected = dense @ expected
    actual_grads = torch.autograd.grad(actual.square().sum(), (weights, state))
    expected_grads = torch.autograd.grad(expected.square().sum(), (weights, state))
    torch.testing.assert_close(actual, expected)
    for left, right in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(left, right)


def test_model_prepares_sparse_values_once_per_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a four-token sequence on a small real execution path.
    source, target = np.nonzero(np.ones((12, 12)) - np.eye(12))
    graph = Graph(
        tuple(f"synthetic:{i}" for i in range(12)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size, dtype=np.float64),
        "synthetic preparation-count fixture",
    )
    model = ConnectomeLM(graph, ARConfig(alphabet_size=7))
    original: _Prepare = SparseTopology.ordered_values
    calls = 0

    def counted(topology: SparseTopology, weights: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(topology, weights)

    monkeypatch.setattr(SparseTopology, "ordered_values", counted)
    # When: one training-style forward and backward processes all four tokens.
    output = model.forward(torch.tensor([[0, 1, 2, 3]]))
    torch.autograd.backward(output.square().sum())
    # Then: canonical weight ordering and duplicate coalescing occur once.
    assert calls == 1


def test_separate_forwards_own_separate_weight_snapshots() -> None:
    # Given: one topology and two parameter snapshots before any backward.
    topology = SparseTopology(torch.tensor([[0, 1], [1, 0]]), nodes=2)
    weights = torch.tensor([0.2, 0.3], dtype=torch.float64, requires_grad=True)
    state = torch.ones((2, 1), dtype=torch.float64)
    first = prepare_recurrence(weights, topology, need_reverse=True)
    first_output = sparse_recur_prepared(weights, state, first, topology, 2)
    second = prepare_recurrence(weights, topology, need_reverse=True)
    second_output = sparse_recur_prepared(weights, state * 2, second, topology, 2)
    # When: the branched losses are differentiated together.
    gradient = torch.autograd.grad(first_output.sum() + second_output.sum(), weights)[0]
    # Then: both sequence snapshots contribute their original-edge derivatives.
    torch.testing.assert_close(gradient, torch.tensor([3.0, 3.0], dtype=torch.float64))
