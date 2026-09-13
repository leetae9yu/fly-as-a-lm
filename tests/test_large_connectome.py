"""Synthetic regression fixtures only; never used to produce real artifacts."""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.lib.npyio import NpzFile

from flyrl.connectome import GraphError, load_graph
from scripts import connectome_arrays as arrays
from scripts import connectome_source as source

if TYPE_CHECKING:
    from numpy import generic


def test_direction_stable_ids_and_duplicate_aggregation() -> None:
    # Given synthetic IDs above float64's exact-integer range and duplicate rows.
    ids = np.array([2**53 + 3, 7, 2**53 + 1], dtype=np.uint64)
    pre = np.array([7, 7, 2**53 + 1, 99, 2**53 + 3], dtype=np.int64)
    post = np.array([2**53 + 1, 2**53 + 1, 7, 7, 2**53 + 3], dtype=np.int64)
    counts = np.array([2, 5, 3, 100, 4], dtype=np.int64)
    # When rows are indexed then canonicalized.
    sorted_ids = ids.copy()
    sorted_ids.sort()
    indexed = arrays.index_edges(sorted_ids, arrays.Edges(pre, post, counts))
    graph = arrays.aggregate_edges(sorted_ids, indexed)
    # Then direction and exact stable identities survive and counts sum by pair.
    np.testing.assert_array_equal(graph.ids, [7, 2**53 + 1, 2**53 + 3])
    np.testing.assert_array_equal(graph.edges.source, [0, 1, 2])
    np.testing.assert_array_equal(graph.edges.target, [1, 0, 2])
    np.testing.assert_array_equal(graph.edges.count, [7, 3, 4])


def test_induced_subgraph_keeps_only_internal_edges_and_original_ids() -> None:
    # Given a synthetic directed graph with a boundary edge and an autapse.
    graph = arrays.aggregate_edges(
        np.array([10, 20, 30], dtype=np.uint64),
        arrays.Edges(
            np.array([0, 1, 2, 2]), np.array([2, 0, 0, 2]), np.array([2, 99, 4, 5])
        ),
    )
    # When selection order differs from stable-ID order.
    subset = arrays.induced_subgraph(graph, np.array([2, 0]))
    # Then IDs remain canonical and directed counts are unchanged.
    np.testing.assert_array_equal(subset.ids, [10, 30])
    np.testing.assert_array_equal(subset.edges.source, [0, 1, 1])
    np.testing.assert_array_equal(subset.edges.target, [1, 0, 1])
    np.testing.assert_array_equal(subset.edges.count, [2, 4, 5])


def test_connected_order_is_deterministic_and_excludes_isolates() -> None:
    # Given a synthetic chain and an isolate; node 2 has greatest contact degree.
    graph = arrays.aggregate_edges(
        np.array([10, 20, 30, 40, 50], dtype=np.uint64),
        arrays.Edges(np.array([0, 1, 2]), np.array([1, 2, 3]), np.array([1, 3, 2])),
    )
    # When a nested weakly connected ordering is requested.
    order = arrays.connected_order(graph, 4)
    # Then the root expands with strength / stable-ID tie breaks.
    np.testing.assert_array_equal(order, [2, 1, 3, 0])


def test_connected_order_rejects_disconnected_padding() -> None:
    graph = arrays.aggregate_edges(
        np.array([10, 20, 30], dtype=np.uint64),
        arrays.Edges(np.array([0]), np.array([1]), np.array([1])),
    )
    with pytest.raises(GraphError):
        _ = arrays.connected_order(graph, 3)


@pytest.mark.parametrize("counts", [[0], [-1], [1.5], [float("nan")]])
def test_source_counts_reject_invalid_values(counts: list[float]) -> None:
    with pytest.raises(GraphError):
        _ = arrays.index_edges(
            np.array([10, 20], dtype=np.uint64),
            arrays.Edges(np.array([10]), np.array([20]), np.array(counts)),
        )


def test_source_ids_never_round_through_float() -> None:
    with pytest.raises(GraphError):
        _ = arrays.exact_ids(np.array([float(2**53)]))


def test_source_integrity_checks_size_and_digest(tmp_path: Path) -> None:
    # Given explicit synthetic bytes with a trusted checksum.
    path = tmp_path / "synthetic.bin"
    _ = path.write_bytes(b"synthetic-only")
    pin = source.SourcePin(
        url="https://example.org/synthetic",
        bytes=14,
        sha256=hashlib.sha256(b"synthetic-only").hexdigest(),
    )
    source.verify_source(path, pin)
    # When corruption preserves length, digest validation still rejects it.
    _ = path.write_bytes(b"synthetic-onlX")
    with pytest.raises(GraphError):
        source.verify_source(path, pin)


def test_source_integrity_rejects_wrong_size(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.bin"
    _ = path.write_bytes(b"synthetic-only")
    pin = source.SourcePin(
        url="https://example.org/synthetic",
        bytes=15,
        sha256=hashlib.sha256(b"synthetic-only").hexdigest(),
    )
    with pytest.raises(GraphError):
        source.verify_source(path, pin)


def test_obtain_source_verifies_only_requested_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: one locally present pinned annotation and no edge artifact.
    annotation = tmp_path / "annotations.feather"
    _ = annotation.write_bytes(b"synthetic-only")
    monkeypatch.setitem(
        source.PINS,
        "annotations.feather",
        source.SourcePin(
            url="https://example.org/synthetic",
            bytes=14,
            sha256=hashlib.sha256(b"synthetic-only").hexdigest(),
        ),
    )
    # When: the caller requests only the annotation source.
    source.obtain_source(tmp_path, "annotations.feather", download=False)
    # Then: the requested file is verified without requiring the 1 GiB edge table.
    assert not (tmp_path / "edges.feather").exists()


def test_npz_uses_raw_counts_and_separately_labeled_model_weights(
    tmp_path: Path,
) -> None:
    # Given an explicitly synthetic one-way connection.
    graph = arrays.aggregate_edges(
        np.array([10, 20], dtype=np.uint64),
        arrays.Edges(np.array([0]), np.array([1]), np.array([7])),
    )
    path = tmp_path / "synthetic.npz"
    # When exported to the existing Graph boundary.
    arrays.export_graph(graph, path, "synthetic-test-only")
    loaded = load_graph(path)
    # Then the model weight is not confused with the anatomical contact count.
    assert loaded.node_ids == ("malecns:v1.0:10", "malecns:v1.0:20")
    np.testing.assert_array_equal(loaded.source, [0])
    np.testing.assert_array_equal(loaded.target, [1])
    np.testing.assert_allclose(loaded.weight, np.log1p([7]))
    with path.open("rb") as stream:
        archive: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with archive:
            np.testing.assert_array_equal(archive["synapse_count"], [7])
