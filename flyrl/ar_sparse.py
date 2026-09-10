"""O(E + NB) sparse recurrence with explicitly bounded weight-gradient workspace.

Native COO sparse-mm executes only inside a custom Function, never through its
weight backward (which can materialize an NxN gradient). Input gradients use a
second sparse-mm. Edge gradients gather at most edge_chunk * batch elements.
This implementation runs on CPU and CUDA without a dense or CPU fallback.
CUDA sparse reductions need not be bitwise deterministic; we do not claim they
are. No second-order derivatives are implemented.
"""

from typing import Protocol, assert_never, runtime_checkable

import torch
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import override

from flyrl.ar_topology import SparseTopology


@runtime_checkable
class _Factory(Protocol):
    def __call__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        size: tuple[int, int],
        *,
        check_invariants: bool,
        is_coalesced: bool,
    ) -> torch.Tensor: ...


@runtime_checkable
class _Operations(Protocol):
    def __call__(self, matrix: torch.Tensor, state: torch.Tensor) -> torch.Tensor: ...


class _Context(Protocol):
    saved_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    topology: SparseTopology | None

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


_FACTORY = TypeAdapter(
    _Factory, config=ConfigDict(arbitrary_types_allowed=True)
).validate_python(vars(torch)["sparse_coo_tensor"])
_OPERATIONS = TypeAdapter(
    _Operations, config=ConfigDict(arbitrary_types_allowed=True)
).validate_python(vars(torch.sparse)["mm"])


def _multiply(
    state: torch.Tensor,
    parts: tuple[torch.Tensor, torch.Tensor],
    coalesced: bool,
) -> torch.Tensor:
    matrix = _FACTORY(
        parts[0],
        parts[1],
        (state.shape[0], state.shape[0]),
        check_invariants=False,  # Indices originate in the validated Graph boundary.
        is_coalesced=coalesced,
    )
    return _OPERATIONS(matrix, state)


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
                indices = edges.edges
                parts = edges.matrix_parts(weights)
                coalesced = True
            case torch.Tensor():
                ctx.topology = None
                indices = edges
                parts = (indices, weights)
                coalesced = False
            case _:
                assert_never(edges)
        ctx.save_for_backward(weights, state, indices, torch.tensor(chunk))
        return _multiply(state, parts, coalesced)

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
        chunk = int(chunk_tensor.item())
        weight_grad = torch.empty_like(weights)
        for start in range(0, weights.numel(), chunk):
            stop = start + chunk
            weight_grad[start:stop] = (
                grad[edges[0, start:stop]] * state[edges[1, start:stop]]
            ).sum(dim=1)
        parts = (
            ctx.topology.matrix_parts(weights, transpose=True)
            if ctx.topology is not None
            else (edges.flip(0), weights)
        )
        state_grad = _multiply(grad, parts, ctx.topology is not None)
        return weight_grad, state_grad, None, None


_APPLY = TypeAdapter(
    _Apply, config=ConfigDict(arbitrary_types_allowed=True)
).validate_python(_SparseRecur)


def sparse_recur(
    weights: torch.Tensor,
    state: torch.Tensor,
    edges: torch.Tensor | SparseTopology,
    chunk: int,
) -> torch.Tensor:
    """Multiply target/source COO edges by [neuron, batch] state."""
    return _APPLY.apply(weights, state, edges, chunk)
