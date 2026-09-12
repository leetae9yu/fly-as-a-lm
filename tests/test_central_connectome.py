"""Synthetic central-brain selection regressions; no scientific training."""

from pathlib import Path

import numpy as np
import pytest

from flyrl.connectome import GraphError, load_graph
from scripts import connectome_arrays as arrays
from scripts import prepare_central_connectome as prepare


def test_central_intrinsic_subgraphs_preserve_induced_anatomy() -> None:
    # Given: central neurons plus a stronger visual boundary.
    graph = arrays.aggregate_edges(
        np.array([10, 20, 30, 40, 50], dtype=np.uint64),
        arrays.Edges(
            np.array([0, 1, 0, 2, 3, 4], dtype=np.int64),
            np.array([1, 0, 2, 3, 4, 0], dtype=np.int64),
            np.array([100, 100, 5, 4, 1, 1], dtype=np.int64),
        ),
    )
    labels = (
        "cb_intrinsic",
        "ol_intrinsic",
        "cb_intrinsic",
        "cb_intrinsic",
        "cb_intrinsic",
    )
    # When: nested central-only connected subsets are selected.
    selection = prepare.central_intrinsic_subgraphs(graph, labels, (2, 4))
    # Then: outside strength is ignored and every internal original edge is preserved.
    np.testing.assert_array_equal(selection.pool.ids[selection.order], [30, 10, 40, 50])
    np.testing.assert_array_equal(selection.graphs[0].ids, [10, 30])
    np.testing.assert_array_equal(selection.graphs[0].edges.count, [5])
    np.testing.assert_array_equal(selection.graphs[1].ids, [10, 30, 40, 50])
    np.testing.assert_array_equal(selection.graphs[1].edges.count, [5, 4, 1, 1])


def test_central_intrinsic_subgraphs_reject_disconnected_padding() -> None:
    # Given: three central neurons but only two in the connected component.
    graph = arrays.aggregate_edges(
        np.array([10, 20, 30], dtype=np.uint64),
        arrays.Edges(
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ),
    )
    # When/Then: node quotas cannot be met with disconnected padding.
    with pytest.raises(GraphError):
        _ = prepare.central_intrinsic_subgraphs(
            graph,
            ("cb_intrinsic", "cb_intrinsic", "cb_intrinsic"),
            (3,),
        )


def test_prepare_central_graphs_writes_machine_readable_provenance(
    tmp_path: Path,
) -> None:
    # Given: one parsed source graph and an explicit small comparison plan.
    graph = arrays.aggregate_edges(
        np.array([10, 20, 30, 40, 50], dtype=np.uint64),
        arrays.Edges(
            np.array([0, 1, 0, 2, 3, 4], dtype=np.int64),
            np.array([1, 0, 2, 3, 4, 0], dtype=np.int64),
            np.array([100, 100, 5, 4, 1, 1], dtype=np.int64),
        ),
    )
    plan = prepare.CentralPlan(
        output=tmp_path,
        sizes=(2, 4),
        comparison_edges=1,
        source_graph_sha256="a" * 64,
        annotations_sha256="b" * 64,
    )
    # When: the central preparation boundary writes both nested graph artifacts.
    manifest = prepare.prepare_central_graphs(
        graph,
        (
            "cb_intrinsic",
            "ol_intrinsic",
            "cb_intrinsic",
            "cb_intrinsic",
            "cb_intrinsic",
        ),
        plan,
    )
    # Then: stable IDs, exact counts, comparison deltas and discovery order persist.
    assert [artifact.neurons for artifact in manifest.artifacts] == [2, 4]
    assert [artifact.directed_edges for artifact in manifest.artifacts] == [1, 4]
    assert [artifact.edge_delta for artifact in manifest.artifacts] == [0, 3]
    assert (
        prepare.CentralManifest.model_validate_json(
            (tmp_path / "manifest.json").read_text()
        )
        == manifest
    )
    assert (tmp_path / "selection_order.npy").is_file()
    loaded = load_graph(tmp_path / "malecns_v1_cb_intrinsic_n2.npz")
    assert loaded.node_ids == ("malecns:v1.0:10", "malecns:v1.0:30")
    np.testing.assert_array_equal(loaded.weight, np.log1p([5]))
