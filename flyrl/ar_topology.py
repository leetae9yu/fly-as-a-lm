"""Reusable canonical COO layouts for an unchanged directed edge pattern."""

import torch


class SparseTopology(torch.nn.Module):
    """Store only O(E) index buffers; weights remain live trainable inputs."""

    edges: torch.Tensor
    forward_edges: torch.Tensor
    reverse_edges: torch.Tensor
    order: torch.Tensor
    reverse_order: torch.Tensor
    groups: torch.Tensor
    has_duplicates: bool

    def __init__(self, edges: torch.Tensor) -> None:
        """Canonicalize once without overflowing packed row/column keys."""
        super().__init__()
        columns = edges[1].argsort(stable=True)
        order = columns[edges[0, columns].argsort(stable=True)]
        sorted_edges = edges[:, order]
        different = torch.ones(edges.shape[1], dtype=torch.bool, device=edges.device)
        different[1:] = (sorted_edges[:, 1:] != sorted_edges[:, :-1]).any(dim=0)
        self.has_duplicates = not bool(different.all().item())
        unique = sorted_edges[:, different] if self.has_duplicates else sorted_edges
        reverse_order = unique[1].argsort(stable=True)
        groups = (
            different.cumsum(dim=0, dtype=torch.long) - 1
            if self.has_duplicates
            else edges.new_empty((0,))
        )
        for name, tensor in (
            ("edges", edges),
            ("forward_edges", unique),
            ("reverse_edges", unique.flip(0)[:, reverse_order]),
            ("order", order),
            ("reverse_order", reverse_order),
            ("groups", groups),
        ):
            self.register_buffer(name, tensor)
        self.edges = self.get_buffer("edges")
        self.forward_edges = self.get_buffer("forward_edges")
        self.reverse_edges = self.get_buffer("reverse_edges")
        self.order = self.get_buffer("order")
        self.reverse_order = self.get_buffer("reverse_order")
        self.groups = self.get_buffer("groups")

    def matrix_parts(
        self, weights: torch.Tensor, *, transpose: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sorted unique indices and current coalesced values."""
        ordered = weights[self.order]
        values = (
            weights.new_zeros((self.forward_edges.shape[1],)).index_add(
                0, self.groups, ordered
            )
            if self.has_duplicates
            else ordered
        )
        if transpose:
            return self.reverse_edges, values[self.reverse_order]
        return self.forward_edges, values
