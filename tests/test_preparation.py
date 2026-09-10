import shutil
from pathlib import Path

import pytest

from flyrl.connectome import GraphError, load_graph
from scripts.prepare_data import prepare


def test_real_data_preparation_preserves_neuron_count(tmp_path: Path) -> None:
    # Given: the unmodified pinned public source data.
    source = Path(__file__).resolve().parents[1] / "data" / "raw"
    _ = shutil.copytree(source, tmp_path / "data" / "raw")
    # When: running preparation.
    graph = prepare(tmp_path)
    # Then: the saved graph keeps every source row as a distinct neuron.
    restored = load_graph(tmp_path / "data" / "larva_left_mb.npz")
    assert len(graph.node_ids) == 209
    assert restored.node_ids == graph.node_ids
    assert restored.weight.size > 1000
    assert (tmp_path / "data" / "provenance.json").exists()


def test_changed_raw_data_is_rejected(tmp_path: Path) -> None:
    # Given: a raw adjacency changed after download.
    source = Path(__file__).resolve().parents[1] / "data" / "raw"
    _ = shutil.copytree(source, tmp_path / "data" / "raw")
    _ = (tmp_path / "data" / "raw" / "left_adjacency.csv").write_text("0 1\n1 0\n")
    # When / Then: a stale or changed source cannot be silently relabeled.
    with pytest.raises(GraphError):
        _ = prepare(tmp_path)
