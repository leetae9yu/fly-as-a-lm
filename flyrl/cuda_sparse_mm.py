"""Lazy float32 CUDA CSR multiplication with one output row per neuron.

Each Triton program owns one row and at most 32 batch columns. Independent
programs dynamically traverse that row in fixed edge tiles, so a high-degree
row does not serialize adjacent rows. Only the [neuron, batch] output is
allocated; edge-by-batch products live in bounded register tiles, not HBM.
"""

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from typing import Final, Protocol, runtime_checkable

import torch
from pydantic import ConfigDict, TypeAdapter

_MATRIX_DIMENSIONS: Final = 2


class _Integer(Protocol):
    """Compile-time integers supplied by Triton's constexpr wrapper."""

    def __index__(self) -> int: ...


class _Value(Protocol):
    """Symbolic tensor expressions consumed by Triton rather than Python."""

    def __add__(self, other: "_Value | _Integer | int") -> "_Value": ...

    def __mul__(self, other: "_Value | _Integer | int") -> "_Value": ...

    def __lt__(self, other: "_Value | _Integer | int") -> "_Value": ...

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

    def full(
        self, shape: tuple[_Integer | int, ...], value: float, dtype: _DType
    ) -> _Value: ...

    def load(
        self, pointer: _Value, mask: _Value | None = None, other: int = 0
    ) -> _Value: ...

    def store(self, pointer: _Value, value: _Value, mask: _Value) -> None: ...

    def sum(self, value: _Value, axis: int) -> _Value: ...


class _Compiled(Protocol):
    n_regs: int


@runtime_checkable
class _Kernel(Protocol):
    def __getitem__(
        self, grid: tuple[int, int]
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _Compiled,
    ]: ...


@runtime_checkable
class _Jit(Protocol):
    def __call__(
        self, function: Callable[[_Value, _Value, _Value, _Value, _Value], None]
    ) -> _Kernel: ...


@lru_cache(maxsize=32)
def _kernel(batch: int, strides: tuple[int, int, int, int, int]) -> _Kernel:
    """Cache immutable layout specialization; Triton chooses the current stream."""
    config = ConfigDict(arbitrary_types_allowed=True)
    language = TypeAdapter(_Language, config=config).validate_python(
        import_module("triton.language")
    )
    jit = TypeAdapter(_Jit, config=config).validate_python(
        vars(import_module("triton"))["jit"]
    )
    tile = min(32, 1 << (batch - 1).bit_length())
    batch_size = language.constexpr(batch)
    batch_tile = language.constexpr(tile)
    edge_tile = language.constexpr(max(32, 128 // tile))
    crow_stride, col_stride, value_stride, state_row, state_col = (
        language.constexpr(stride) for stride in strides
    )

    def multiply(
        crow: "_Value",
        columns: "_Value",
        weights: "_Value",
        state: "_Value",
        output: "_Value",
    ) -> None:
        row = language.program_id(0)
        batch_index = language.program_id(1) * batch_tile + language.arange(
            0, batch_tile
        )
        start = language.load(crow + row * crow_stride)
        stop = language.load(crow + (row + 1) * crow_stride)
        edge_offset = language.arange(0, edge_tile)
        total = language.full((edge_tile, batch_tile), 0.0, language.float32)
        while start < stop:
            edge = start + edge_offset
            live = edge < stop
            column = language.load(columns + edge * col_stride, live, 0)
            value = language.load(weights + edge * value_stride, live, 0)
            mask = live[:, None] & (batch_index < batch_size)[None, :]
            source = language.load(
                state + column[:, None] * state_row + batch_index[None, :] * state_col,
                mask,
                0,
            )
            total = total + value[:, None] * source
            start = start + edge_tile
        language.store(
            output + row * batch_size + batch_index,
            language.sum(total, 0),
            batch_index < batch_size,
        )

    return jit(multiply)


def sparse_matrix_state(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """Return canonical CSR @ float32 [neuron, batch] state on CUDA.

    CSR inputs are int64 [neuron+1] row pointers, int64 [edge] sorted columns,
    and float32 [edge] values, on the state's CUDA device. The caller validates
    pointer monotonicity/endpoints and column bounds once when building topology;
    parallel edges must already be coalesced. No host read or synchronization is
    used to revalidate index contents on this hot path. Strided views, including
    zero strides, are supported. Each neuron keeps its original output row.

    This is a forward primitive for use inside a custom autograd Function, not
    an autograd implementation. Import, compilation and launch errors propagate
    without fallback. Float32 reductions may differ in rounding from cuSPARSE.
    """
    inputs = (crow_indices, col_indices, values, state)
    if not state.is_cuda or any(tensor.device != state.device for tensor in inputs):
        message = "CSR multiplication requires inputs on the same CUDA device"
        raise ValueError(message)
    if crow_indices.dtype != torch.int64 or col_indices.dtype != torch.int64:
        message = "CSR multiplication requires int64 indices"
        raise ValueError(message)
    if values.dtype != torch.float32 or state.dtype != torch.float32:
        message = "CSR multiplication requires float32 values and state"
        raise ValueError(message)
    if state.ndim != _MATRIX_DIMENSIONS or any(
        tensor.ndim != 1 for tensor in inputs[:3]
    ):
        message = "Expected CSR vector shapes and a [neuron, batch] state shape"
        raise ValueError(message)
    nodes, batch = state.shape
    if crow_indices.numel() != nodes + 1 or col_indices.shape != values.shape:
        message = "Incompatible CSR and state shapes"
        raise ValueError(message)
    output = state.new_empty((nodes, batch))
    if nodes == 0 or batch == 0:
        return output
    if values.numel() == 0:
        return output.zero_()
    strides = (
        crow_indices.stride(0),
        col_indices.stride(0),
        values.stride(0),
        state.stride(0),
        state.stride(1),
    )
    with torch.cuda.device(state.device):
        _ = _kernel(batch, strides)[(nodes, (batch + 31) // 32)](
            crow_indices, col_indices, values, state, output
        )
    return output
