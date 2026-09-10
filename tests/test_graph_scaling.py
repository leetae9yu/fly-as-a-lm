import tracemalloc

import numpy as np
import pytest

from flyrl.connectome import Graph, GraphError


def test_graph_validation_has_bounded_auxiliary_memory() -> None:
    # Given: a million unique synthetic edges, allocated before measurement.
    count = 1000
    source, target = np.indices((count, count), dtype=np.int64).reshape(2, -1)
    weight = np.ones(count * count, dtype=np.float64)
    identities = tuple(f"synthetic:{index}" for index in range(count))
    # When: validating an input graph with a fixed auxiliary-memory budget.
    tracemalloc.start()
    try:
        graph = Graph(identities, source, target, weight, "synthetic memory regression")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Then: no per-edge Python-object expansion is required.
    assert graph.weight.size == 1_000_000
    assert peak < 64 * graph.weight.size


def test_unsorted_duplicate_edges_are_rejected() -> None:
    # Given: duplicate pairs separated by another source in the input order.
    source = np.array([2, 0, 1, 2], dtype=np.int64)
    target = np.array([1, 2, 0, 1], dtype=np.int64)
    # When / Then: large-graph validation still detects real duplicate pairs.
    with pytest.raises(GraphError):
        _ = Graph(("a", "b", "c"), source, target, np.ones(4), "synthetic")
