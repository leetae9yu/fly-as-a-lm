"""Native sparse wrappers and original-edge derivative dispatch."""

from typing import Protocol, runtime_checkable

import torch
from pydantic import ConfigDict, TypeAdapter

from flyrl.cuda_edge_grad import edge_weight_gradient


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


_FACTORY = TypeAdapter(
    _Factory, config=ConfigDict(arbitrary_types_allowed=True)
).validate_python(vars(torch)["sparse_coo_tensor"])
_OPERATIONS = TypeAdapter(
    _Operations, config=ConfigDict(arbitrary_types_allowed=True)
).validate_python(vars(torch.sparse)["mm"])


def sparse_matrix(
    indices: torch.Tensor,
    values: torch.Tensor,
    nodes: int,
    *,
    coalesced: bool = True,
) -> torch.Tensor:
    """Wrap validated COO storage without checking it again."""
    return _FACTORY(
        indices,
        values,
        (nodes, nodes),
        check_invariants=False,
        is_coalesced=coalesced,
    )


def multiply_parts(
    state: torch.Tensor,
    parts: tuple[torch.Tensor, torch.Tensor],
    coalesced: bool,
) -> torch.Tensor:
    """Build one sparse wrapper and multiply it by neuron-batch state."""
    return multiply_sparse(
        sparse_matrix(
            parts[0],
            parts[1],
            state.shape[0],
            coalesced=coalesced,
        ),
        state,
    )


def multiply_sparse(matrix: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Multiply one already-prepared native sparse matrix."""
    return _OPERATIONS(matrix, state)


def edge_gradient(
    weights: torch.Tensor,
    grad: torch.Tensor,
    state: torch.Tensor,
    edges: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """Use fused CUDA work or the bounded native device/dtype path."""
    if weights.is_cuda and weights.dtype == torch.float32:
        return edge_weight_gradient(grad, state, edges)
    weight_grad = torch.empty_like(weights)
    for start in range(0, weights.numel(), chunk):
        stop = start + chunk
        weight_grad[start:stop] = (
            grad[edges[0, start:stop]] * state[edges[1, start:stop]]
        ).sum(dim=1)
    return weight_grad
