"""O(E + NB) sparse recurrence with explicitly bounded weight-gradient workspace.

Native COO sparse-mm executes only inside a custom Function, never through its
weight backward (which can materialize an NxN gradient). Input gradients use a
second sparse-mm. CUDA float32 edge gradients fuse gathers and batch reduction
in Triton without materializing edge-by-batch tensors. Other supported device
and dtype combinations retain the bounded native chunk path. There is no dense
or CPU fallback on CUDA failures.
CUDA sparse reductions need not be bitwise deterministic; we do not claim they
are. No second-order derivatives are implemented.
"""

from typing import Protocol, assert_never, runtime_checkable

import torch
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import override

from flyrl.ar_prepared import (
    PreparedRecurrence,
    multiply_prepared,
)
from flyrl.ar_sparse_ops import edge_gradient, multiply_parts
from flyrl.ar_topology import SparseTopology


class _Context(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    topology: SparseTopology | None
    prepared: "PreparedRecurrence | None"

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


@runtime_checkable
class _Apply(Protocol):
    def apply(
        self,
        weights: torch.Tensor,
        state: torch.Tensor,
        edges: torch.Tensor | SparseTopology,
        chunk: int,
    ) -> torch.Tensor: ...


@runtime_checkable
class _PreparedApply(Protocol):
    def apply(
        self,
        weights: torch.Tensor,
        state: torch.Tensor,
        prepared: "PreparedRecurrence",
        topology: SparseTopology,
        chunk: int,
    ) -> torch.Tensor: ...


class _SparseRecur(torch.autograd.Function):
    @staticmethod
    @override
    def forward(
        ctx: _Context,
        weights: torch.Tensor,
        state: torch.Tensor,
        edges: torch.Tensor | SparseTopology,
        chunk: int,
    ) -> torch.Tensor:
        match edges:
            case SparseTopology():
                ctx.topology = edges
                ctx.prepared = None
                indices = edges.edges
                parts = edges.matrix_parts(weights)
                coalesced = True
            case torch.Tensor():
                ctx.topology = None
                ctx.prepared = None
                indices = edges
                parts = (indices, weights)
                coalesced = False
            case _:
                assert_never(edges)
        ctx.save_for_backward(weights, state, indices, torch.tensor(chunk))
        return multiply_parts(state, parts, coalesced)

    @staticmethod
    @override
    def backward(
        ctx: _Context, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        if torch.is_grad_enabled():
            message = "Higher-order sparse derivatives are unsupported"
            raise RuntimeError(message)
        grad = grad_outputs[0]
        weights, state, edges, chunk_tensor = ctx.saved_tensors
        weight_grad = edge_gradient(
            weights,
            grad,
            state,
            edges,
            int(chunk_tensor.item()),
        )
        parts = (
            ctx.topology.matrix_parts(weights, transpose=True)
            if ctx.topology is not None
            else (edges.flip(0), weights)
        )
        state_grad = multiply_parts(grad, parts, ctx.topology is not None)
        return weight_grad, state_grad, None, None


class _SparseRecurPrepared(torch.autograd.Function):
    @staticmethod
    @override
    def forward(
        ctx: _Context,
        weights: torch.Tensor,
        state: torch.Tensor,
        prepared: PreparedRecurrence,
        topology: SparseTopology,
        chunk: int,
    ) -> torch.Tensor:
        ctx.topology = topology
        ctx.prepared = prepared
        reverse_values = prepared.reverse_values
        saved_reverse = (
            weights.new_empty((0,)) if reverse_values is None else reverse_values
        )
        ctx.save_for_backward(
            weights,
            state,
            topology.edges,
            prepared.values,
            saved_reverse,
            torch.tensor(chunk),
        )
        return multiply_prepared(
            weights,
            state,
            prepared,
            topology,
            transpose=False,
        )

    @staticmethod
    @override
    def backward(
        ctx: _Context, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        if torch.is_grad_enabled():
            message = "Higher-order sparse derivatives are unsupported"
            raise RuntimeError(message)
        grad = grad_outputs[0]
        weights, state, edges, _, reverse_values, chunk_tensor = ctx.saved_tensors
        prepared = ctx.prepared
        topology = ctx.topology
        if prepared is None or topology is None or reverse_values.numel() == 0:
            message = "Prepared recurrence omitted the reverse operator"
            raise RuntimeError(message)
        weight_grad = edge_gradient(
            weights,
            grad,
            state,
            edges,
            int(chunk_tensor.item()),
        )
        return (
            weight_grad,
            multiply_prepared(
                weights,
                grad,
                prepared,
                topology,
                transpose=True,
            ),
            None,
            None,
            None,
        )


_APPLY = TypeAdapter(
    _Apply, config=ConfigDict(arbitrary_types_allowed=True)
).validate_python(_SparseRecur)
_PREPARED_APPLY = TypeAdapter(
    _PreparedApply, config=ConfigDict(arbitrary_types_allowed=True)
).validate_python(_SparseRecurPrepared)


def sparse_recur_prepared(
    weights: torch.Tensor,
    state: torch.Tensor,
    prepared: PreparedRecurrence,
    topology: SparseTopology,
    chunk: int,
) -> torch.Tensor:
    """Multiply with sequence values while retaining original-edge gradients."""
    if not torch.is_grad_enabled():
        return multiply_prepared(
            weights,
            state,
            prepared,
            topology,
            transpose=False,
        )
    if prepared.reverse_values is None:
        message = "Gradient-enabled recurrence requires a prepared transpose"
        raise RuntimeError(message)
    return _PREPARED_APPLY.apply(
        weights,
        state,
        prepared,
        topology,
        chunk,
    )


def sparse_recur(
    weights: torch.Tensor,
    state: torch.Tensor,
    edges: torch.Tensor | SparseTopology,
    chunk: int,
) -> torch.Tensor:
    """Multiply target/source COO edges by [neuron, batch] state."""
    return _APPLY.apply(weights, state, edges, chunk)
