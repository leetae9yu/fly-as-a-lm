"""Deterministic, capacity-matched regional probe groups."""

import numpy as np

from flyrl.connectome import Graph
from scripts.regional_probe_groups import RegionalSelection, regional_probe_groups


def graph() -> Graph:
    nodes = 30
    source = np.arange(nodes, dtype=np.int64)
    target = (source + 1) % nodes
    return Graph(
        tuple(f"node:{index}" for index in range(nodes)),
        np.asarray([*source.tolist(), *source.tolist()], dtype=np.int64),
        np.asarray(
            [*target.tolist(), *((source + 5) % nodes).tolist()],
            dtype=np.int64,
        ),
        np.linspace(0.1, 2.0, 2 * nodes, dtype=np.float64),
        "synthetic regional probe fixture",
    )


def test_regional_groups_are_deterministic_and_capacity_matched() -> None:
    # Given: disjoint anatomical classes and source-model functional ports.
    selection = RegionalSelection(
        alpn=tuple(range(5)),
        mbon=tuple(range(5, 8)),
        kenyon=tuple(range(8, 18)),
        sensory=(18, 19),
        trained_readout=(8, 20, 21),
        seed=7,
    )
    # When: the prospective groups are constructed twice.
    first = regional_probe_groups(graph(), selection, group_size=3, draws=2)
    second = regional_probe_groups(graph(), selection, group_size=3, draws=2)
    # Then: every head has equal width and all deterministic identities match.
    assert first == second
    assert len(first) == 8
    assert all(len(group.indices) == 3 for group in first)
    assert [group.name for group in first] == [
        "alpn",
        "alpn",
        "kenyon",
        "kenyon",
        "mbon",
        "degree_matched",
        "centrality",
        "trained_readout",
    ]
    by_name = {group.name: group for group in first}
    assert by_name["mbon"].indices == (5, 6, 7)
    assert by_name["trained_readout"].indices == (8, 20, 21)
    assert set(by_name["degree_matched"].indices) <= set(range(22, 30))
    assert set(by_name["centrality"].indices) <= set(range(22, 30))


def test_anatomical_draws_exclude_existing_functional_ports() -> None:
    # Given: source sensory/readout ports overlapping the Kenyon class.
    groups = regional_probe_groups(
        graph(),
        RegionalSelection(
            alpn=tuple(range(5)),
            mbon=tuple(range(5, 8)),
            kenyon=tuple(range(8, 18)),
            sensory=(8, 9),
            trained_readout=(10, 20, 21),
            seed=8,
        ),
        group_size=3,
        draws=2,
    )
    # When/Then: sampled anatomy never inherits direct input/readout membership.
    excluded = {8, 9, 10, 20, 21}
    for group in groups:
        if group.name in {"alpn", "kenyon", "degree_matched", "centrality"}:
            assert not set(group.indices) & excluded
