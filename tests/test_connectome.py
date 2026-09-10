from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from numpy.lib.format import write_array
from numpy.typing import NDArray

from flyrl.connectome import (
    Graph,
    GraphError,
    from_adjacency,
    load_graph,
    save_graph,
    shuffled_graph,
)


def test_counts_preserve_direction_and_log_strength() -> None:
    # Given: distinct directed counts, including an observed self-connection.
    counts = np.array([[0.0, 3.0], [7.0, 1.0]])
    # When: importing the anatomical adjacency.
    graph = from_adjacency(counts, ("left:0:P", "left:1:O"), "synthetic")
    # Then: rows are sources, columns are targets, and no edge is invented.
    np.testing.assert_array_equal(graph.source, [0, 1, 1])
    np.testing.assert_array_equal(graph.target, [1, 0, 1])
    np.testing.assert_allclose(graph.weight, np.log1p([3, 7, 1]))


@pytest.mark.parametrize(
    "counts",
    [
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[0.0, np.nan], [1.0, 0.0]]),
        np.zeros((2, 3)),
        np.zeros((2, 2)),
    ],
)
def test_invalid_counts_are_rejected(counts: NDArray[np.float64]) -> None:
    # Given: invalid counts or an empty graph.
    # When / Then: the data boundary rejects them.
    with pytest.raises(GraphError):
        _ = from_adjacency(counts, ("a", "b"), "synthetic")


def test_duplicate_node_ids_are_rejected() -> None:
    # Given: two distinct rows assigned the same identity.
    counts = np.array([[0.0, 1.0], [1.0, 0.0]])
    # When / Then: identity ambiguity cannot enter the model.
    with pytest.raises(GraphError):
        _ = from_adjacency(counts, ("a", "a"), "synthetic")


def test_graph_roundtrip_preserves_identity_and_weights(tmp_path: Path) -> None:
    # Given: an imported graph saved without pickle.
    graph = from_adjacency(np.array([[0.0, 2.0], [1.0, 0.0]]), ("a", "b"), "fixture")
    path = tmp_path / "graph.npz"
    save_graph(graph, path)
    # When: reopening the artifact.
    restored = load_graph(path)
    # Then: exact scientific inputs survive.
    assert restored.node_ids == graph.node_ids
    assert restored.provenance == graph.provenance
    np.testing.assert_array_equal(restored.source, graph.source)
    np.testing.assert_array_equal(restored.target, graph.target)
    np.testing.assert_array_equal(restored.weight, graph.weight)


def test_shuffle_preserves_directed_degrees_and_source_sign() -> None:
    # Given: a sparse directed ring with mixed source signs.
    source = np.asarray([node for node in range(12) for _ in (1, 3)], dtype=np.int64)
    target = np.asarray(
        [(node + offset) % 12 for node in range(12) for offset in (1, 3)],
        dtype=np.int64,
    )
    weights = np.asarray(
        [1.0 if node % 2 == 0 else -2.0 for node in range(12) for _ in (1, 3)]
    )
    graph = Graph(tuple(map(str, range(12))), source, target, weights, "synthetic")
    # When: creating a topological control.
    shuffled = shuffled_graph(graph, 42)
    # Then: both degrees and source-associated weights remain matched.
    np.testing.assert_array_equal(shuffled.source, graph.source)
    np.testing.assert_array_equal(shuffled.weight, graph.weight)
    np.testing.assert_array_equal(
        np.bincount(shuffled.target, minlength=12),
        np.bincount(graph.target, minlength=12),
    )
    assert (
        len(set(zip(shuffled.source.tolist(), shuffled.target.tolist(), strict=True)))
        == 24
    )
    assert np.not_equal(shuffled.source, shuffled.target).all()
    assert np.not_equal(shuffled.target, graph.target).any()
    np.testing.assert_array_equal(shuffled_graph(graph, 42).target, shuffled.target)


def test_load_rejects_out_of_bounds_indices(tmp_path: Path) -> None:
    # Given: a saved file whose edge references a nonexistent neuron.
    path = tmp_path / "bad.npz"
    arrays = (
        ("node_ids", np.array(["a", "b"])),
        ("source", np.array([0])),
        ("target", np.array([2])),
        ("weight", np.array([1.0])),
        ("provenance", np.array("synthetic")),
    )
    with ZipFile(path, "w") as archive:
        for name, array in arrays:
            with archive.open(f"{name}.npy", "w") as stream:
                write_array(stream, array, allow_pickle=False)
    # When / Then: a corrupt graph fails before simulation.
    with pytest.raises(GraphError):
        _ = load_graph(path)
