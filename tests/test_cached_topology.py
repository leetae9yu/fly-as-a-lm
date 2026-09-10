import pytest
import torch

from flyrl.ar_sparse import sparse_recur
from flyrl.ar_topology import SparseTopology


def test_cached_layout_reuses_current_values_and_handles_duplicates() -> None:
    # Given: unordered directed edges, including a duplicated pair.
    edges = torch.tensor([[2, 0, 2, 1], [0, 2, 0, 1]])
    topology = SparseTopology(edges)
    # When: assembling the pattern twice with different live parameter values.
    first = topology.matrix_parts(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    second = topology.matrix_parts(torch.tensor([5.0, 6.0, 7.0, 8.0]))
    # Then: the static pattern is canonical but weights were not cached.
    assert torch.equal(first[0], torch.tensor([[0, 1, 2], [2, 1, 0]]))
    assert torch.equal(first[1], torch.tensor([2.0, 4.0, 4.0]))
    assert torch.equal(second[1], torch.tensor([6.0, 8.0, 12.0]))


def test_transpose_is_sorted_and_preserves_mathematical_operator() -> None:
    # Given: a non-symmetric pattern whose transpose needs another order.
    edges = torch.tensor([[1, 0, 2], [0, 2, 1]])
    topology = SparseTopology(edges)
    # When: obtaining the transposed representation.
    indices, values = topology.matrix_parts(
        torch.tensor([3.0, 5.0, 7.0]), transpose=True
    )
    # Then: row/source order and the associated values move together.
    assert torch.equal(indices, torch.tensor([[0, 1, 2], [1, 2, 0]]))
    assert torch.equal(values, torch.tensor([3.0, 7.0, 5.0]))


@pytest.mark.parametrize("duplicate", [False, True])
def test_cached_layout_preserves_weight_gradients(duplicate: bool) -> None:
    # Given: a differentiable set of scalar edge weights.
    edges = torch.tensor([[2, 0, 2 if duplicate else 1], [0, 2, 0]])
    topology = SparseTopology(edges)
    weights = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64, requires_grad=True)
    # When: differentiating a function of the coalesced weights.
    _, values = topology.matrix_parts(weights)
    derivative = torch.autograd.grad(values.square().sum(), weights)[0]
    # Then: duplicate parameters receive the sum-operator derivative.
    expected = [1.2, 0.6, 1.2] if duplicate else [0.4, 0.6, 0.8]
    assert torch.allclose(derivative, torch.tensor(expected, dtype=torch.float64))


def test_cached_sparse_operator_matches_dense_gradients() -> None:
    # Given: a fixed pattern with repeated and unordered edges.
    edges = torch.tensor([[2, 0, 2, 1], [0, 2, 0, 1]])
    topology = SparseTopology(edges)
    weights = torch.tensor([0.2, 0.3, 0.4, -0.1], dtype=torch.float64)
    state = torch.arange(6, dtype=torch.float64).reshape(3, 2) / 7
    _ = weights.requires_grad_()
    _ = state.requires_grad_()
    # When: using the cached pattern inside the custom backward path.
    actual = sparse_recur(weights, state, topology, 2)
    dense = torch.zeros((3, 3), dtype=torch.float64).index_put(
        (edges[0], edges[1]), weights, accumulate=True
    )
    expected = dense @ state
    # Then: both parameter and input gradients retain the same operator.
    assert torch.allclose(actual, expected)
    observed = torch.autograd.grad(actual.square().sum(), (weights, state))
    reference = torch.autograd.grad(expected.square().sum(), (weights, state))
    assert all(
        torch.allclose(one, two) for one, two in zip(observed, reference, strict=True)
    )
