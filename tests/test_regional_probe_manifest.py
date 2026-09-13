"""Portable exact manifests for regional probe membership."""

import numpy as np

from flyrl.connectome import Graph
from scripts.anatomy_port_artifacts import AnatomyIndices
from scripts.regional_probe_groups import RegionalSelection
from scripts.regional_probe_manifest import (
    GroupManifestIdentities,
    RegionalGroupManifest,
    build_group_manifest,
)


def test_group_manifest_binds_exact_indices_and_node_ids() -> None:
    # Given: one synthetic graph, anatomy and paired source ports.
    nodes = 30
    source = np.arange(nodes, dtype=np.int64)
    graph = Graph(
        tuple(f"node:{index}" for index in range(nodes)),
        np.asarray([*source.tolist(), *source.tolist()], dtype=np.int64),
        np.asarray(
            [*((source + 1) % nodes).tolist(), *((source + 5) % nodes).tolist()],
            dtype=np.int64,
        ),
        np.linspace(0.1, 2.0, 2 * nodes, dtype=np.float64),
        "synthetic regional manifest fixture",
    )
    selection = RegionalSelection(
        alpn=tuple(range(5)),
        mbon=tuple(range(5, 8)),
        kenyon=tuple(range(8, 18)),
        sensory=(18, 19),
        trained_readout=(8, 20, 21),
        seed=7,
    )
    # When: the exact group manifest is built and JSON-round-tripped.
    manifest = build_group_manifest(
        graph,
        AnatomyIndices(selection.alpn, selection.mbon, selection.kenyon),
        (selection,),
        GroupManifestIdentities(
            "a" * 64,
            "b" * 64,
            "c" * 64,
            group_size=3,
            draws=2,
        ),
    )
    restored = RegionalGroupManifest.model_validate_json(manifest.model_dump_json())
    # Then: membership, source ports and stable IDs remain bound.
    assert restored == manifest
    assert restored.seeds[0].sensory_indices == (18, 19)
    assert len(restored.seeds[0].groups) == 8
    for group in restored.seeds[0].groups:
        assert group.node_ids == tuple(graph.node_ids[index] for index in group.indices)
