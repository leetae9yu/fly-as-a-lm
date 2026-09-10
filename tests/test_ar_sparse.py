"""Sparse kernels must preserve dense reference derivatives without dense state."""

import pytest
import torch

from flyrl.ar_sparse import sparse_recur


def test_sparse_forward_and_backward() -> None:
    edges = torch.tensor([[0, 2, 1, 2], [1, 0, 2, 2]])
    weights = torch.tensor([0.3, -0.2, 0.7, 0.1], dtype=torch.float64)
    state = torch.arange(6, dtype=torch.float64).reshape(3, 2) / 7
    _ = weights.requires_grad_()
    _ = state.requires_grad_()
    actual = sparse_recur(weights, state, edges, 2)
    dense = torch.zeros(3, 3, dtype=torch.float64).index_put(
        (edges[0], edges[1]), weights
    )
    expected = dense @ state
    assert torch.allclose(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), (weights, state))
    expected_grads = torch.autograd.grad(expected.square().sum(), (weights, state))
    assert all(
        torch.allclose(a, b) for a, b in zip(actual_grads, expected_grads, strict=True)
    )

    def operation(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return sparse_recur(w, x, edges, 2)

    assert torch.autograd.gradcheck(operation, (weights, state))


def test_duplicate_control_edges_and_no_quadratic_saved_tensor() -> None:
    edges = torch.tensor([[0, 0, 999], [1, 1, 998]])
    weights = torch.ones(3, requires_grad=True)
    state = torch.ones(1000, 2, requires_grad=True)
    shapes: list[tuple[int, ...]] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        shapes.append(tuple(tensor.shape))
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        result = sparse_recur(weights, state, edges, 1)
        torch.autograd.backward(result.sum())
    assert result[0, 0].item() == 2
    assert weights.grad is not None
    assert torch.equal(weights.grad, torch.full((3,), 2.0))
    assert (1000, 1000) not in shapes
    assert max(torch.Size(shape).numel() for shape in shapes) <= 2000


def test_higher_order_gradients_are_explicitly_unsupported() -> None:
    edges = torch.tensor([[0, 1], [1, 0]])
    weights = torch.ones(2, requires_grad=True)
    state = torch.ones(2, 1, requires_grad=True)
    output = sparse_recur(weights, state, edges, 1)
    with pytest.raises(RuntimeError, match="Higher-order"):
        _ = torch.autograd.grad(output.sum(), weights, create_graph=True)
