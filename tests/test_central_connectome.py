"""Synthetic central-brain selection regressions; no scientific training."""

from pathlib import Path

import numpy as np
import pytest

import scripts.anatomy_port_artifacts as port_artifacts
import scripts.anatomy_ports as ports
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


def test_anatomy_port_audit_counts_paths_and_contacts() -> None:
    # Given: one ALPN, one Kenyon cell and two MBONs in a directed anatomy.
    graph = arrays.aggregate_edges(
        np.array([10, 20, 30, 40, 50], dtype=np.uint64),
        arrays.Edges(
            np.array([0, 1, 0, 3], dtype=np.int64),
            np.array([1, 2, 2, 4], dtype=np.int64),
            np.array([5, 4, 2, 1], dtype=np.int64),
        ),
    )
    # When: exact publisher classes define candidate input and output ports.
    audit = ports.anatomy_port_audit(
        graph,
        ("ALPN", "Kenyon_Cell", "MBON", "other", "MBON"),
    )
    # Then: direct pairs, contacts and directed reachability remain distinct.
    np.testing.assert_array_equal(audit.alpn_indices, [0])
    np.testing.assert_array_equal(audit.mbon_indices, [2, 4])
    np.testing.assert_array_equal(audit.kenyon_indices, [1])
    assert audit.alpn_to_kenyon.directed_pairs == 1
    assert audit.alpn_to_kenyon.contacts == 5
    assert audit.kenyon_to_mbon.directed_pairs == 1
    assert audit.kenyon_to_mbon.contacts == 4
    assert audit.alpn_to_mbon.directed_pairs == 1
    assert audit.alpn_to_mbon.contacts == 2
    assert audit.reachable_mbons == 1


def test_anatomy_port_artifacts_bind_indices_and_source_hashes(
    tmp_path: Path,
) -> None:
    # Given: an audited synthetic ALPN-to-Kenyon-to-MBON circuit.
    graph = arrays.aggregate_edges(
        np.array([10, 20, 30], dtype=np.uint64),
        arrays.Edges(
            np.array([0, 1], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([5, 4], dtype=np.int64),
        ),
    )
    audit = ports.anatomy_port_audit(graph, ("ALPN", "Kenyon_Cell", "MBON"))
    plan = port_artifacts.PortArtifactPlan(
        output=tmp_path,
        graph_sha256="a" * 64,
        annotations_sha256="b" * 64,
    )
    # When: the reproducible port boundary freezes indices and metadata.
    manifest = port_artifacts.write_port_artifacts(graph, audit, plan)
    # Then: both files validate against the exact graph and source identities.
    assert manifest.alpn_count == 1
    assert manifest.mbon_count == 1
    assert manifest.reachable_mbons == 1
    assert manifest.graph_sha256 == "a" * 64
    assert manifest.annotations_sha256 == "b" * 64
    assert port_artifacts.load_port_indices(tmp_path / manifest.arrays, manifest) == (
        (0,),
        (2,),
    )
    anatomy = port_artifacts.load_anatomy_indices(
        tmp_path / manifest.arrays,
        manifest,
    )
    assert anatomy.alpn == (0,)
    assert anatomy.kenyon == (1,)
    assert anatomy.mbon == (2,)


def test_align_port_classes_preserves_graph_index_order() -> None:
    # Given: graph IDs are a strict subset of an independently ordered annotation table.
    graph_ids = np.array([20, 40], dtype=np.uint64)
    annotation_ids = np.array([40, 10, 20, 30], dtype=np.uint64)
    annotation_classes = ("MBON", "other", "ALPN", "Kenyon_Cell")
    # When: classes are joined by exact body ID.
    aligned = port_artifacts.align_classes(
        graph_ids, annotation_ids, annotation_classes
    )
    # Then: output order follows graph indices rather than source table order.
    assert aligned == ("ALPN", "MBON")


def test_condition_ports_are_reproducible_disjoint_and_factorial(
    tmp_path: Path,
) -> None:
    # Given: exact anatomy ports with enough unlabeled random candidates.
    graph = arrays.aggregate_edges(
        np.arange(10, 20, dtype=np.uint64),
        arrays.Edges(
            np.arange(9, dtype=np.int64),
            np.arange(1, 10, dtype=np.int64),
            np.ones(9, dtype=np.int64),
        ),
    )
    audit = ports.anatomy_port_audit(
        graph,
        ("ALPN", "ALPN", "Kenyon_Cell", "MBON", *("other",) * 6),
    )
    manifest = port_artifacts.write_port_artifacts(
        graph,
        audit,
        port_artifacts.PortArtifactPlan(
            output=tmp_path,
            graph_sha256="a" * 64,
            annotations_sha256="b" * 64,
        ),
    )
    # When: one seed resolves all four explicit factorial policies.
    anatomy = port_artifacts.condition_port_indices(manifest, tmp_path, "alpn_mbon", 7)
    mixed_input = port_artifacts.condition_port_indices(
        manifest, tmp_path, "alpn_random", 7
    )
    mixed_output = port_artifacts.condition_port_indices(
        manifest, tmp_path, "random_mbon", 7
    )
    random = port_artifacts.condition_port_indices(
        manifest, tmp_path, "random_random", 7
    )
    repeated = port_artifacts.condition_port_indices(
        manifest, tmp_path, "random_random", 7
    )
    # Then: anatomy and random halves are reused exactly without overlap.
    assert anatomy.sensory_indices == mixed_input.sensory_indices == (0, 1)
    assert anatomy.readout_indices == mixed_output.readout_indices == (3,)
    assert mixed_output.sensory_indices == random.sensory_indices
    assert mixed_input.readout_indices == random.readout_indices
    assert random == repeated
    random_union = set(random.sensory_indices) | set(random.readout_indices)
    assert not set(random.sensory_indices) & set(random.readout_indices)
    assert not random_union & {0, 1, 3}
    configured = port_artifacts.load_condition_ports(
        tmp_path / "anatomy_ports.json",
        "random_random",
        7,
    )
    assert configured.sensory_indices == random.sensory_indices
    assert configured.readout_indices == random.readout_indices
    assert len(configured.manifest_sha256) == 64
