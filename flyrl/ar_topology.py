"""Reusable canonical COO layouts for an unchanged directed edge pattern."""

import torch


class SparseTopology(torch.nn.Module):
    """Store only O(E) index buffers; weights remain live trainable inputs."""

    node_count: int
    edges: torch.Tensor
    forward_edges: torch.Tensor
    reverse_edges: torch.Tensor
    order: torch.Tensor
    reverse_order: torch.Tensor
    groups: torch.Tensor
    forward_crow: torch.Tensor
    forward_col: torch.Tensor
    reverse_crow: torch.Tensor
    reverse_col: torch.Tensor
    has_duplicates: bool

    def __init__(self, edges: torch.Tensor, *, nodes: int | None = None) -> None:
        """Canonicalize once without overflowing packed row/column keys."""
        super().__init__()
        inferred = int(edges.max().item()) + 1
        self.node_count = inferred if nodes is None else nodes
        if self.node_count < inferred:
            message = "Sparse topology node count excludes an edge endpoint"
            raise ValueError(message)
        columns = edges[1].argsort(stable=True)
        order = columns[edges[0, columns].argsort(stable=True)]
        sorted_edges = edges[:, order]
        different = torch.ones(edges.shape[1], dtype=torch.bool, device=edges.device)
        different[1:] = (sorted_edges[:, 1:] != sorted_edges[:, :-1]).any(dim=0)
        self.has_duplicates = not bool(different.all().item())
        unique = sorted_edges[:, different] if self.has_duplicates else sorted_edges
        reverse_order = unique[1].argsort(stable=True)
        reverse_edges = unique.flip(0)[:, reverse_order]
        groups = (
            different.cumsum(dim=0, dtype=torch.long) - 1
            if self.has_duplicates
            else edges.new_empty((0,))
        )
        for name, tensor in (
            ("edges", edges),
            ("forward_edges", unique),
            ("reverse_edges", reverse_edges),
            ("order", order),
            ("reverse_order", reverse_order),
            ("groups", groups),
        ):
            self.register_buffer(name, tensor)
        for name, tensor in (
            ("forward_crow", self._crow(unique[0])),
            ("forward_col", unique[1].contiguous()),
            ("reverse_crow", self._crow(reverse_edges[0])),
            ("reverse_col", reverse_edges[1].contiguous()),
        ):
            self.register_buffer(name, tensor, persistent=False)
        self.edges = self.get_buffer("edges")
        self.forward_edges = self.get_buffer("forward_edges")
        self.reverse_edges = self.get_buffer("reverse_edges")
        self.order = self.get_buffer("order")
        self.reverse_order = self.get_buffer("reverse_order")
        self.groups = self.get_buffer("groups")
        self.forward_crow = self.get_buffer("forward_crow")
        self.forward_col = self.get_buffer("forward_col")
        self.reverse_crow = self.get_buffer("reverse_crow")
        self.reverse_col = self.get_buffer("reverse_col")

    def _crow(self, rows: torch.Tensor) -> torch.Tensor:
        """Convert sorted row IDs to a complete CSR pointer including empty rows."""
        counts = torch.bincount(rows, minlength=self.node_count)
        return torch.cat((rows.new_zeros((1,)), counts.cumsum(dim=0)))

    def ordered_values(self, weights: torch.Tensor) -> torch.Tensor:
        """Order and coalesce one live parameter snapshot."""
        ordered = weights[self.order]
        return (
            weights.new_zeros((self.forward_edges.shape[1],)).index_add(
                0, self.groups, ordered
            )
            if self.has_duplicates
            else ordered
        )

    def csr_layout(
        self, *, transpose: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached row pointers and columns for one operator direction."""
        return (
            (self.reverse_crow, self.reverse_col)
            if transpose
            else (self.forward_crow, self.forward_col)
        )

    def matrix_parts(
        self, weights: torch.Tensor, *, transpose: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sorted unique indices and current coalesced values."""
        values = self.ordered_values(weights)
        if transpose:
            return self.reverse_edges, values[self.reverse_order]
        return self.forward_edges, values
