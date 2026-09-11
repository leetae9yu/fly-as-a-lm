"""Sequence-scoped sparse operators derived from one immutable weight snapshot."""

from dataclasses import dataclass

import torch

from flyrl.ar_sparse_ops import multiply_sparse, sparse_matrix
from flyrl.ar_topology import SparseTopology
from flyrl.cuda_sparse_mm import sparse_matrix_state


@dataclass(frozen=True, slots=True)
class PreparedRecurrence:
    """Detached values and native wrappers valid for one sequence."""

    values: torch.Tensor
    reverse_values: torch.Tensor | None
    forward: torch.Tensor | None
    reverse: torch.Tensor | None


@torch.no_grad()
def prepare_recurrence(
    weights: torch.Tensor,
    topology: SparseTopology,
    *,
    need_reverse: bool,
) -> PreparedRecurrence:
    """Prepare current forward and optional transpose values once."""
    values = topology.ordered_values(weights)
    reverse_values = values[topology.reverse_order] if need_reverse else None
    use_cuda_kernel = weights.is_cuda and weights.dtype == torch.float32
    return PreparedRecurrence(
        values=values,
        reverse_values=reverse_values,
        forward=(
            None
            if use_cuda_kernel
            else sparse_matrix(
                topology.forward_edges,
                values,
                topology.node_count,
            )
        ),
        reverse=(
            None
            if use_cuda_kernel or reverse_values is None
            else sparse_matrix(
                topology.reverse_edges,
                reverse_values,
                topology.node_count,
            )
        ),
    )


def multiply_prepared(
    weights: torch.Tensor,
    state: torch.Tensor,
    prepared: PreparedRecurrence,
    topology: SparseTopology,
    *,
    transpose: bool,
) -> torch.Tensor:
    """Use cached CSR on CUDA float32 and detached COO wrappers elsewhere."""
    if weights.is_cuda and weights.dtype == torch.float32:
        values = prepared.reverse_values if transpose else prepared.values
        if values is None:
            message = "Prepared recurrence omitted transpose values"
            raise RuntimeError(message)
        crow, columns = topology.csr_layout(transpose=transpose)
        return sparse_matrix_state(crow, columns, values, state)
    matrix = prepared.reverse if transpose else prepared.forward
    if matrix is None:
        message = "Prepared recurrence omitted a native sparse operator"
        raise RuntimeError(message)
    return multiply_sparse(matrix, state)
