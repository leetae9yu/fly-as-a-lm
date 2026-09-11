"""Opt-in float32 CUDA edge derivatives with O(E) output and bounded workspace.

Triton is imported only on the first nonempty CUDA call. Each program owns 128
original edges and reduces batches in register tiles of at most 32 columns.
Parallel edges are deliberately neither sorted nor coalesced. This first-order
backward primitive assumes graph indices have already been validated by Graph.
"""

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from typing import Final, Protocol, runtime_checkable

import torch
from pydantic import ConfigDict, TypeAdapter

_MATRIX_DIMENSIONS: Final = 2
_EDGE_ENDPOINTS: Final = 2


class _Integer(Protocol):
    """Triton's compile-time integer supports Python's index protocol."""

    def __index__(self) -> int: ...


class _Value(Protocol):
    """Tensor expressions interpreted by Triton, not executed by Python."""

    def __add__(self, other: "_Value | _Integer | int") -> "_Value": ...

    def __radd__(self, other: int) -> "_Value": ...

    def __mul__(self, other: "_Value | _Integer | int") -> "_Value": ...

    def __lt__(self, other: _Integer | int) -> "_Value": ...

    def __and__(self, other: "_Value") -> "_Value": ...

    def __getitem__(self, key: tuple[slice | None, ...]) -> "_Value": ...


class _DType(Protocol):
    name: str


@runtime_checkable
class _Language(Protocol):
    float32: _DType

    def constexpr(self, value: int) -> _Integer: ...

    def arange(self, start: int, end: _Integer | int) -> _Value: ...

    def program_id(self, axis: int) -> _Value: ...

    def full(self, shape: tuple[int, ...], value: float, dtype: _DType) -> _Value: ...

    def load(self, pointer: _Value, mask: _Value, other: int) -> _Value: ...

    def store(self, pointer: _Value, value: _Value, mask: _Value) -> None: ...

    def sum(self, value: _Value, axis: int) -> _Value: ...


class _Compiled(Protocol):
    n_regs: int


@runtime_checkable
class _Kernel(Protocol):
    def __getitem__(
        self, grid: tuple[int]
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], _Compiled
    ]: ...


@runtime_checkable
class _Jit(Protocol):
    def __call__(
        self, function: Callable[[_Value, _Value, _Value, _Value, int], None]
    ) -> _Kernel: ...


@lru_cache(maxsize=32)
def _kernel(batch: int, strides: tuple[int, int, int, int, int, int]) -> _Kernel:
    """Specialize immutable layout constants without a CPU Triton dependency."""
    config = ConfigDict(arbitrary_types_allowed=True)
    language = TypeAdapter(_Language, config=config).validate_python(
        import_module("triton.language")
    )
    jit = TypeAdapter(_Jit, config=config).validate_python(
        vars(import_module("triton"))["jit"]
    )
    batch_size = language.constexpr(batch)
    batch_tile = language.constexpr(min(32, 1 << (batch - 1).bit_length()))
    grad_row, grad_col, state_row, state_col, edge_row, edge_col = (
        language.constexpr(stride) for stride in strides
    )

    def fused(
        grad: "_Value", state: "_Value", edges: "_Value", output: "_Value", count: int
    ) -> None:
        index = language.program_id(0) * 128 + language.arange(0, 128)
        live = index < count
        target = language.load(edges + index * edge_col, live, 0)
        source = language.load(edges + edge_row + index * edge_col, live, 0)
        total = language.full((128,), 0.0, language.float32)
        for start in range(0, batch_size, batch_tile):
            column = start + language.arange(0, batch_tile)
            mask = live[:, None] & (column < batch_size)[None, :]
            g = language.load(
                grad + target[:, None] * grad_row + column[None, :] * grad_col,
                mask,
                0,
            )
            x = language.load(
                state + source[:, None] * state_row + column[None, :] * state_col,
                mask,
                0,
            )
            total = total + language.sum(g * x, 1)
        language.store(output + index, total, live)

    return jit(fused)


def edge_weight_gradient(
    grad: torch.Tensor, state: torch.Tensor, edges: torch.Tensor
) -> torch.Tensor:
    """Return sum_b grad[target,b]*state[source,b] in original edge order.

    Inputs are strided CUDA float32 [neuron, batch] tensors and CUDA int64
    [2, edge] indices on the same device. Index bounds are the caller's validated
    topology contract. No autograd graph or higher-order derivative is built.
    Import, compilation, and launch failures propagate without fallback.
    """
    if not grad.is_cuda or state.device != grad.device or edges.device != grad.device:
        message = "Edge gradients require inputs on the same CUDA device"
        raise ValueError(message)
    if grad.dtype != torch.float32 or state.dtype != torch.float32:
        message = "Edge gradients require float32 grad and state"
        raise ValueError(message)
    if edges.dtype != torch.int64:
        message = "Edge gradients require int64 indices"
        raise ValueError(message)
    if (
        grad.ndim != _MATRIX_DIMENSIONS
        or state.shape != grad.shape
        or edges.ndim != _MATRIX_DIMENSIONS
    ):
        message = "Expected matching [neuron, batch] tensors and [2, edge] indices"
        raise ValueError(message)
    if edges.shape[0] != _EDGE_ENDPOINTS:
        message = "Expected [2, edge] indices"
        raise ValueError(message)
    count, batch = edges.shape[1], state.shape[1]
    output = grad.new_empty((count,))
    if count == 0:
        return output
    if batch == 0:
        return output.zero_()
    strides = (
        grad.stride(0),
        grad.stride(1),
        state.stride(0),
        state.stride(1),
        edges.stride(0),
        edges.stride(1),
    )
    with torch.cuda.device(grad.device):
        _ = _kernel(batch, strides)[((count + 127) // 128,)](
            grad, state, edges, output, count
        )
    return output
