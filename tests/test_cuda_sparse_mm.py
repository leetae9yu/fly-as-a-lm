"""Isolated CSR CUDA candidate: dense-reference semantics and backend boundaries."""

import subprocess
import sys
from importlib import import_module, reload
from importlib.util import find_spec
from types import ModuleType
from typing import Final, Protocol, runtime_checkable

import pytest
import torch
from pydantic import ConfigDict, TypeAdapter

from flyrl import ar_prepared, ar_sparse
from flyrl.ar_prepared import prepare_recurrence
from flyrl.ar_topology import SparseTopology

_CUDA: Final = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


@runtime_checkable
class _SparseMatrixState(Protocol):
    def __call__(
        self,
        crow_indices: torch.Tensor,
        col_indices: torch.Tensor,
        values: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor: ...


def _candidate() -> _SparseMatrixState:
    assert find_spec("flyrl.cuda_sparse_mm") is not None, "CUDA candidate is missing"
    module = import_module("flyrl.cuda_sparse_mm")
    assert "sparse_matrix_state" in vars(module), "CUDA candidate API is missing"
    return TypeAdapter(
        _SparseMatrixState, config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(vars(module)["sparse_matrix_state"])


def _inputs(device: str = "cuda") -> list[torch.Tensor]:
    return [
        torch.tensor([0, 1, 1, 2], dtype=torch.int64, device=device),
        torch.tensor([0, 2], dtype=torch.int64, device=device),
        torch.tensor([0.5, -1.5], dtype=torch.float32, device=device),
        torch.arange(24, dtype=torch.float32, device=device).reshape(3, 8),
    ]


def test_import_is_lazy_on_cpu() -> None:
    # Given a fresh interpreter, even on hosts with Triton installed.
    assert callable(_candidate())
    code = (
        "import sys; import flyrl.cuda_sparse_mm; "
        "assert not any(k == 'triton' or k.startswith('triton.') for k in sys.modules)"
    )
    # When importing the public module without invoking its CUDA operator.
    result = subprocess.run(  # noqa: S603 - fixed interpreter and source, no user input.
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    # Then importing has no optional backend dependency.
    assert result.returncode == 0, result.stderr


def test_cpu_inputs_are_rejected_without_fallback() -> None:
    # Given structurally valid CPU tensors.
    inputs = _inputs("cpu")
    candidate = _candidate()
    # When requesting the CUDA-only operator, then it rejects the device.
    with pytest.raises(ValueError, match="CUDA"):
        _ = candidate(*inputs)


@_CUDA
@pytest.mark.parametrize("index", range(4))
def test_mixed_devices_are_rejected(index: int) -> None:
    # Given one CPU tensor among otherwise valid CUDA inputs.
    inputs = _inputs()
    inputs[index] = inputs[index].cpu()
    # When invoking the operator, then device mismatches fail at the boundary.
    with pytest.raises(ValueError, match="CUDA"):
        _ = _candidate()(*inputs)


@_CUDA
@pytest.mark.parametrize(
    ("index", "dtype"),
    [(0, torch.int32), (1, torch.int32), (2, torch.float64), (3, torch.float16)],
)
def test_wrong_dtypes_are_rejected(index: int, dtype: torch.dtype) -> None:
    # Given an otherwise valid input with an unsupported scalar type.
    inputs = _inputs()
    inputs[index] = inputs[index].to(dtype=dtype)
    # When invoking the operator, then no implicit conversion occurs.
    with pytest.raises(ValueError, match=r"float32|int64"):
        _ = _candidate()(*inputs)


@_CUDA
@pytest.mark.parametrize(
    ("index", "shape"),
    [
        (0, (2, 2)),
        (1, (1, 2)),
        (2, (1, 2)),
        (3, (24,)),
        (0, (3,)),
        (1, (1,)),
        (2, (1,)),
        (3, (4, 6)),
    ],
)
def test_wrong_shapes_are_rejected(index: int, shape: tuple[int, ...]) -> None:
    # Given rank or CSR/state size mismatches (contents are never dereferenced).
    inputs = _inputs()
    inputs[index] = inputs[index].new_zeros(shape)
    # When invoking the operator, then incompatible metadata is rejected.
    with pytest.raises(ValueError, match="shape"):
        _ = _candidate()(*inputs)


@_CUDA
@pytest.mark.parametrize("batch", [1, 3, 8, 17, 32, 64])
@pytest.mark.parametrize("strided", [False, True])
def test_irregular_rows_match_independent_dense_reference(
    batch: int, *, strided: bool
) -> None:
    # Given sorted, coalesced CSR with loops, empty rows and degree/tile tails.
    nodes = 1031
    degrees = [0, 1, 3, 7, 8, 17, 31, 32, 33, 127, 128, 129, 513, nodes]
    dense = torch.zeros((nodes, nodes), dtype=torch.float64)
    for row, degree in enumerate(degrees):
        columns = torch.arange(degree)
        dense[row, columns] = torch.sin(columns.double() * 0.37 + row) + 0.05
    dense[-2, -2] = -0.75
    dense = dense.float()
    occupied = dense != 0
    crow = torch.cat((torch.zeros(1, dtype=torch.int64), occupied.sum(1).cumsum(0)))
    columns = occupied.nonzero()[:, 1]
    weights = dense[occupied]
    state = torch.cos(torch.arange(nodes * batch).float() * 0.13).reshape(nodes, batch)
    expected = (dense.double() @ state.double()).float()
    inputs = [tensor.cuda() for tensor in (crow, columns, weights, state)]
    if strided:
        for index in range(3):
            storage = inputs[index].new_empty((inputs[index].numel() * 2 + 1,))
            view = storage[1::2]
            _ = view.copy_(inputs[index])
            inputs[index] = view
        storage = state.new_empty((batch * 2 + 1, nodes * 3 + 1), device="cuda")
        inputs[3] = storage[1::2, 1::3].T
        _ = inputs[3].copy_(state)
    # When executing the actual public CUDA operator.
    actual = _candidate()(*inputs)
    # Then every neuron retains its row, without permutation or missing zeros.
    assert actual.shape == (nodes, batch)
    assert actual.dtype == torch.float32
    assert actual.device == inputs[3].device
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-5, atol=1e-4)


@_CUDA
@pytest.mark.parametrize(("nodes", "batch", "edges"), [(0, 8, 0), (7, 8, 0), (3, 0, 2)])
def test_empty_dimensions_preserve_output_shape(
    nodes: int, batch: int, edges: int
) -> None:
    # Given an empty graph or batch, with a valid row pointer even for zero nodes.
    crow = torch.zeros(nodes + 1, dtype=torch.int64, device="cuda")
    crow[-1] = edges
    columns = torch.arange(edges, dtype=torch.int64, device="cuda")
    values = torch.ones(edges, device="cuda")
    state = torch.empty((nodes, batch), device="cuda")
    # When reducing an empty input.
    actual = _candidate()(crow, columns, values, state)
    # Then empty sums are zero and the output is always [neuron, batch].
    torch.testing.assert_close(actual, torch.zeros((nodes, batch), device="cuda"))


@_CUDA
def test_broadcast_views_follow_current_stream() -> None:
    # Given inputs produced on a non-default stream and zero strides.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        crow = torch.tensor([0, 1, 1, 3], device="cuda")
        columns = torch.tensor([0, 0, 2], device="cuda")
        values = torch.tensor([2.0], device="cuda").expand(3)
        state = torch.arange(3, dtype=torch.float32, device="cuda")[:, None].expand(
            3, 8
        )
        # When consuming these tensors immediately on their producing stream.
        actual = _candidate()(crow, columns, values, state)
        complete = stream.record_event()
    complete.synchronize()
    # Then the same-stream dependency and broadcast strides are respected.
    expected = torch.tensor([0.0, 0.0, 4.0])[:, None].expand(3, 8)
    torch.testing.assert_close(actual.cpu(), expected)


@_CUDA
def test_backend_import_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a fresh candidate whose optional dependency import fails.
    _ = _candidate()
    module = reload(import_module("flyrl.cuda_sparse_mm"))

    def unavailable(name: str) -> ModuleType:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(module, "import_module", unavailable)
    inputs = _inputs()
    # When calling the backend, then the original failure escapes, never CPU fallback.
    with pytest.raises(ModuleNotFoundError):
        _ = _candidate()(*inputs)


@_CUDA
def test_workspace_excludes_dense_or_edge_batch_intermediates() -> None:
    # Given enough edges that E*B and N*N allocations exceed the bounded budget.
    nodes, degree, batch = 4097, 33, 64
    crow = torch.arange(nodes + 1, device="cuda") * degree
    columns = torch.arange(degree, device="cuda").repeat(nodes)
    values = torch.full((nodes * degree,), 1.0 / degree, device="cuda")
    state = torch.ones((nodes, batch), device="cuda")
    candidate = _candidate()
    warm = candidate(crow, columns, values, state)
    torch.cuda.synchronize()
    del warm
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    # When invoking the real operator after compilation, counting PyTorch allocations.
    actual = candidate(crow, columns, values, state)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    # Then only the output plus allocator rounding fits (not E*B or N*N workspace).
    assert peak <= actual.numel() * actual.element_size() + 1024 * 1024
    torch.testing.assert_close(actual, torch.ones_like(actual))


@_CUDA
def test_prepared_recurrence_dispatches_both_directions_to_csr_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: one prepared CUDA recurrence and an observed CSR operator boundary.
    edges = torch.tensor([[0, 2, 1, 2], [1, 0, 2, 2]], device="cuda")
    topology = SparseTopology(edges, nodes=3)
    weights = torch.tensor([0.3, -0.2, 0.7, 0.1], device="cuda", requires_grad=True)
    state = torch.arange(6, dtype=torch.float32, device="cuda").reshape(3, 2)
    _ = state.requires_grad_()
    prepared = prepare_recurrence(weights, topology, need_reverse=True)
    candidate = _candidate()
    calls = 0

    def observed(
        crow: torch.Tensor,
        columns: torch.Tensor,
        values: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return candidate(crow, columns, values, current)

    monkeypatch.setattr(ar_prepared, "sparse_matrix_state", observed)
    # When: the actual prepared recurrence executes forward and state backward.
    output = ar_sparse.sparse_recur_prepared(
        weights, state, prepared, topology, chunk=2
    )
    _ = torch.autograd.grad(output.sum(), (weights, state))
    # Then: forward and transpose each use the cached CSR kernel exactly once.
    assert calls == 2
