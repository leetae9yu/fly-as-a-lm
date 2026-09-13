# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "pydantic>=2.10,<3"]
# ///
# Run: uv run --no-sync python scripts/prepare_regional_probe_groups.py
"""Freeze exact regional probe groups for source checkpoint seeds 7-12."""

import hashlib
import sys
from pathlib import Path

from flyrl.connectome import load_graph
from flyrl.story_pilot import PilotReport
from scripts.anatomy_port_artifacts import (
    AnatomyPortManifest,
    load_anatomy_indices,
)
from scripts.regional_probe_groups import RegionalSelection
from scripts.regional_probe_manifest import (
    GroupManifestIdentities,
    build_group_manifest,
)


def digest(path: Path) -> str:
    """Hash one immutable protocol input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path.cwd()
graph_path = root / "data/central_connectome/malecns_v1_cb_intrinsic_n5600.npz"
port_manifest_path = root / "data/central_connectome/anatomy_ports.json"
port_manifest = AnatomyPortManifest.model_validate_json(port_manifest_path.read_text())
anatomy = load_anatomy_indices(
    port_manifest_path.parent / port_manifest.arrays,
    port_manifest,
)
graph = load_graph(graph_path)
selections: list[RegionalSelection] = []
for seed in range(7, 13):
    reports = tuple(
        PilotReport.model_validate_json(
            (
                root
                / "results/anatomy-factorial-t4"
                / f"seed-{seed}"
                / f"{wiring}-random_random"
                / "report.json"
            ).read_text()
        )
        for wiring in ("real", "shuffled")
    )
    real, shuffled = reports
    if (
        real.config.port_policy != "random_random"
        or shuffled.config.port_policy != "random_random"
        or real.config.seed != seed
        or shuffled.config.seed != seed
        or real.config.sensory_indices is None
        or real.config.readout_indices is None
        or real.config.sensory_indices != shuffled.config.sensory_indices
        or real.config.readout_indices != shuffled.config.readout_indices
    ):
        message = f"Source report ports differ for paired seed {seed}"
        raise ValueError(message)
    selections.append(
        RegionalSelection(
            alpn=anatomy.alpn,
            mbon=anatomy.mbon,
            kenyon=anatomy.kenyon,
            sensory=real.config.sensory_indices,
            trained_readout=real.config.readout_indices,
            seed=seed,
        )
    )
manifest = build_group_manifest(
    graph,
    anatomy,
    tuple(selections),
    GroupManifestIdentities(
        graph_sha256=digest(graph_path),
        port_manifest_sha256=digest(port_manifest_path),
        port_arrays_sha256=port_manifest.arrays_sha256,
    ),
)
output = port_manifest_path.parent / "regional_probe_groups.json"
_ = output.write_text(manifest.model_dump_json(indent=2) + "\n")
_ = sys.stdout.write(f"REGIONAL_PROBE_GROUPS_READY {digest(output)}\n")
