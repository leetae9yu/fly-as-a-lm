"""Sealed worker boundary tests without loading or running real checkpoints."""

import runpy
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import TypeAdapter

from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.connectome import Graph
from flyrl.regional_probe_types import ExtractionConfig, ProbeConfig
from scripts.connectome_source import file_digest
from scripts.regional_probe_groups import SELECTION_SEED
from scripts.regional_probe_manifest import (
    ProbeGroupRecord,
    RegionalGroupManifest,
    SeedGroupPlan,
)
from scripts.regional_probe_protocol import ProbeSourceCheckpoint, RegionalProbeProtocol
from scripts.regional_probe_schema import EXPECTED_DRAWS, SEEDS, WIRINGS
from scripts.regional_probe_worker_support import (
    WorkerInputs,
    cuda_device_index,
    parameter_fingerprint,
    project_file,
    source_files,
    union_indices,
    validate_inputs,
)

WORKER = Path(__file__).resolve().parents[1] / "scripts/regional_probe_worker.py"


@pytest.fixture
def inputs() -> WorkerInputs:
    edges = np.arange(120, dtype=np.int64)
    graph = Graph(
        tuple(str(i) for i in range(120)),
        edges,
        (edges + 1) % 120,
        np.ones(120, dtype=np.float64),
        "synthetic",
    )
    indices = tuple(range(1, 98))
    groups = tuple(
        ProbeGroupRecord(
            name=name,
            draw=draw,
            indices=indices,
            node_ids=tuple(graph.node_ids[i] for i in indices),
        )
        for name, count in EXPECTED_DRAWS.items()
        for draw in range(count)
    )
    manifest = RegionalGroupManifest(
        graph_sha256="a" * 64,
        port_manifest_sha256="b" * 64,
        port_arrays_sha256="c" * 64,
        selection_seed=SELECTION_SEED,
        group_size=97,
        draws=5,
        seeds=tuple(
            SeedGroupPlan(
                seed=seed,
                sensory_indices=(0,),
                trained_readout_indices=indices,
                groups=groups,
            )
            for seed in SEEDS
        ),
    )
    sources = tuple(
        ProbeSourceCheckpoint(
            seed=seed,
            wiring=wiring,
            filename=f"seed-{seed}-{wiring}-random_random.zip",
            archive_sha256="d" * 64,
            archive_bytes=123,
            checkpoint_sha256="e" * 64,
            report_sha256="f" * 64,
            runtime_sha256="0" * 64,
            parameter_fingerprint="1" * 64,
        )
        for seed in SEEDS
        for wiring in WIRINGS
    )
    protocol = RegionalProbeProtocol(
        graph="graph.npz",
        graph_sha256=manifest.graph_sha256,
        graph_nodes=120,
        graph_edges=120,
        corpus="corpus.npz",
        corpus_sha256="2" * 64,
        corpus_fingerprint="3" * 64,
        port_manifest="ports.json",
        port_manifest_sha256=manifest.port_manifest_sha256,
        port_arrays="ports.npz",
        port_arrays_sha256=manifest.port_arrays_sha256,
        group_manifest="regional_probe_groups.json",
        group_manifest_sha256="4" * 64,
        source_sha256="5" * 64,
        worker_sha256=file_digest(WORKER),
        checkpoints=sources,
        extraction=ExtractionConfig(),
        probe=ProbeConfig(vocab_size=4096, updates=0),
        group_size=97,
        draws=5,
        heads_per_checkpoint=14,
        practical_threshold=0.10,
        heldout_policy="sealed",
    )
    return WorkerInputs(graph, protocol, manifest)


def test_complete_ordered_matrix_and_union(inputs: WorkerInputs) -> None:
    validate_inputs(inputs)
    assert union_indices(inputs.groups.seeds[0]) == tuple(range(1, 98))
    assert len(inputs.protocol.checkpoints) * len(inputs.groups.seeds[0].groups) == 168


def test_unspecified_cuda_device_uses_current_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert cuda_device_index(torch.device("cuda")) == 2
    assert cuda_device_index(torch.device("cuda:1")) == 1


@pytest.mark.parametrize(
    "fault", ["sources", "groups", "capacity", "nodes", "ports", "threshold"]
)
def test_worker_rejects_wrong_identities_and_incomplete_order(
    inputs: WorkerInputs, fault: str
) -> None:
    protocol, groups = inputs.protocol, inputs.groups
    plan = groups.seeds[0]
    if fault == "sources":
        protocol = protocol.model_copy(
            update={"checkpoints": tuple(reversed(protocol.checkpoints))}
        )
    elif fault == "groups":
        plan = plan.model_copy(update={"groups": tuple(reversed(plan.groups))})
    elif fault == "capacity":
        group = plan.groups[0].model_copy(
            update={"indices": plan.groups[0].indices[:-1]}
        )
        plan = plan.model_copy(update={"groups": (group, *plan.groups[1:])})
    elif fault == "nodes":
        group = plan.groups[0].model_copy(
            update={"node_ids": tuple(reversed(plan.groups[0].node_ids))}
        )
        plan = plan.model_copy(update={"groups": (group, *plan.groups[1:])})
    elif fault == "ports":
        plan = plan.model_copy(update={"sensory_indices": (1,)})
    else:
        protocol = protocol.model_copy(update={"practical_threshold": 0.01})
    groups = groups.model_copy(update={"seeds": (plan, *groups.seeds[1:])})
    with pytest.raises(ValueError, match="differ"):
        validate_inputs(WorkerInputs(inputs.graph, protocol, groups))


def test_source_bytes_and_model_mutation_are_detected(
    tmp_path: Path, inputs: WorkerInputs
) -> None:
    fields = {
        "checkpoint.npz": "checkpoint_sha256",
        "report.json": "report_sha256",
        "runtime.json": "runtime_sha256",
    }
    hashes: dict[str, str] = {}
    for name, field in fields.items():
        path = tmp_path / name
        _ = path.write_bytes(name.encode())
        hashes[field] = file_digest(path)
    source = ProbeSourceCheckpoint.model_validate(
        {**inputs.protocol.checkpoints[0].model_dump(), **hashes}
    )
    source_files(tmp_path, source)
    _ = (tmp_path / "runtime.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        source_files(tmp_path, source)
    learner = ARLearner(inputs.graph, ARConfig(alphabet_size=2, readout_neurons=2))
    before = parameter_fingerprint(learner)
    with torch.no_grad():
        _ = learner.model.bias.add_(1)
    assert parameter_fingerprint(learner) != before
    with torch.no_grad():
        learner.model.bias[0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        _ = parameter_fingerprint(learner)


def test_project_paths_reject_escape_ambiguity_and_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside"):
        _ = project_file(tmp_path, "../outside")
    with pytest.raises(ValueError, match="missing"):
        _ = project_file(tmp_path, "graph.npz")
    nested = tmp_path / "data/central_connectome/graph.npz"
    nested.parent.mkdir(parents=True)
    _ = nested.write_bytes(b"graph")
    assert project_file(tmp_path, "graph.npz") == nested
    _ = (tmp_path / "graph.npz").write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="ambiguous"):
        _ = project_file(tmp_path, "graph.npz")


def test_real_worker_surface_rejects_cpu_before_loading_sources(
    tmp_path: Path,
    inputs: WorkerInputs,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "flyrl-0.1.0"
    project.mkdir()
    protocol = inputs.protocol
    hashes: dict[str, str] = {}
    for name, field in (
        (protocol.graph, "graph_sha256"),
        (protocol.corpus, "corpus_sha256"),
        (protocol.port_manifest, "port_manifest_sha256"),
        (protocol.port_arrays, "port_arrays_sha256"),
        (protocol.group_manifest, "group_manifest_sha256"),
    ):
        path = project / name
        _ = path.write_bytes(name.encode())
        hashes[field] = file_digest(path)
    protocol = RegionalProbeProtocol.model_validate({**protocol.model_dump(), **hashes})
    _ = (project / "protocol.json").write_text(protocol.model_dump_json())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    adapter = TypeAdapter[Callable[[Path], None]](Callable[[Path], None])
    run = adapter.validate_python(runpy.run_path(str(WORKER))["run"])
    with pytest.raises(ValueError, match="Tesla T4"):
        run(tmp_path)
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "results").exists()
