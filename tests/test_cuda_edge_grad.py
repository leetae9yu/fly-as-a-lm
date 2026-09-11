"""Original-edge derivatives for the opt-in fused CUDA candidate."""

from importlib import import_module
from importlib.util import find_spec
from typing import Protocol, runtime_checkable

import pytest
import torch
from pydantic import ConfigDict, TypeAdapter

from flyrl.ar_sparse import sparse_recur


@runtime_checkable
class _Operator(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def count(self) -> int: ...


@runtime_checkable
class _EdgeGradient(Protocol):
    def __call__(
        self, grad: torch.Tensor, state: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor: ...


def _candidate() -> _EdgeGradient:
    assert find_spec("flyrl.cuda_edge_grad") is not None, "CUDA candidate is missing"
    module = import_module("flyrl.cuda_edge_grad")
    assert "edge_weight_gradient" in vars(module), "CUDA candidate API is missing"
    return TypeAdapter(
        _EdgeGradient, config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(vars(module)["edge_weight_gradient"])


def test_candidate_is_importable_on_cpu() -> None:
    # Given a host that need not have Triton or CUDA installed.
    # When importing the callable, no GPU backend should be loaded.
    candidate = _candidate()
    # Then the optional backend exposes its CPU-importable API.
    assert callable(candidate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sparse_backward_avoids_materialized_edge_gathers() -> None:
    # Given: original-order CUDA edges and an independently differentiable state.
    edges = torch.tensor([[0, 2, 1, 2], [1, 0, 2, 2]], device="cuda")
    weights = torch.tensor([0.3, -0.2, 0.7, 0.1], device="cuda", requires_grad=True)
    state = torch.arange(6, dtype=torch.float32, device="cuda").reshape(3, 2)
    _ = state.requires_grad_()
    output = sparse_recur(weights, state, edges, 2)
    # When: differentiating through the actual sparse recurrence entry point.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True
    ) as trace:
        _ = torch.autograd.grad(output.sum(), (weights, state))
    # Then: the CUDA backward no longer launches separate edge-gather operations.
    events = TypeAdapter(
        list[_Operator], config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(trace.key_averages())
    assert sum(event.count for event in events if event.key == "aten::index") == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("batch", [1, 3, 8, 17, 33, 65, 129])
@pytest.mark.parametrize("strided", [False, True])
def test_original_edge_gradient_matches_dense_derivative(
    batch: int, *, strided: bool
) -> None:
    # Given unsorted original edges with parallel edges, loops, and a tile tail.
    nodes, count = 19, 263
    index = torch.arange(count, dtype=torch.int64)
    edges = torch.stack(((index * 7 + 3) % nodes, (index * 11 + 5) % nodes))
    edges[:, 0] = torch.tensor([4, 4])
    edges[:, 1] = edges[:, 0]
    edges[:, -1] = torch.tensor([0, 18])
    values = torch.arange(nodes * batch, dtype=torch.float32).reshape(nodes, batch)
    grad = torch.sin(values * 0.37) + 0.13
    state = torch.cos(values * 0.19) - 0.21
    dense_derivative = grad.double() @ state.double().T
    expected = dense_derivative[edges[0], edges[1]].float()
    device_grad, device_state, device_edges = (
        tensor.cuda() for tensor in (grad, state, edges)
    )
    if strided:
        grad_storage = torch.empty((batch * 2 + 1, nodes * 2 + 1), device="cuda")
        device_grad = grad_storage[1::2, 1::2].T
        _ = device_grad.copy_(grad)
        state_storage = torch.empty((nodes * 2 + 1, batch * 3 + 1), device="cuda")
        device_state = state_storage[1::2, 1::3]
        _ = device_state.copy_(state)
        edge_storage = torch.empty((count * 2 + 1, 5), dtype=torch.int64, device="cuda")
        device_edges = edge_storage[1::2, 1::2].T
        _ = device_edges.copy_(edges)
    # When gathering and reducing through the real fused backend.
    actual = _candidate()(device_grad, device_state, device_edges)
    # Then every original edge retains its independent, correctly ordered gradient.
    assert actual.shape == (count,)
    assert actual.dtype == torch.float32
    assert actual.device == device_grad.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(("batch", "count"), [(0, 3), (3, 0)])
def test_empty_dimensions_return_original_edge_shape(batch: int, count: int) -> None:
    # Given a valid empty reduction or empty graph.
    grad = torch.empty((2, batch), dtype=torch.float32, device="cuda")
    state = torch.empty_like(grad)
    edges = torch.zeros((2, count), dtype=torch.int64, device="cuda")
    # When evaluating the candidate.
    actual = _candidate()(grad, state, edges)
    # Then an empty sum is zero, and an empty graph has no gradients.
    torch.testing.assert_close(actual, torch.zeros(count, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_broadcast_strides_use_the_current_cuda_stream() -> None:
    # Given zero-stride views and inputs produced on a non-default stream.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        grad = torch.arange(7, dtype=torch.float32, device="cuda")[:, None].expand(7, 5)
        row = torch.arange(5, dtype=torch.float32, device="cuda")
        state = row[None, :].expand(7, 5)
        edges = torch.tensor([[6, 1, 3, 6], [0, 5, 3, 0]], device="cuda")
        # When running immediately after input production on that same stream.
        actual = _candidate()(grad, state, edges)
        complete = stream.record_event()
    complete.synchronize()
    # Then launch ordering and zero strides preserve the edge dot products.
    torch.testing.assert_close(actual.cpu(), torch.tensor([60.0, 10.0, 30.0, 60.0]))
