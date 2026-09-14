"""Synthetic paired sources exercise real old-training extraction and matching."""

import sys
from collections.abc import Callable
from hashlib import sha256
from itertools import accumulate
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from flyrl.ar_checkpoint import save_checkpoint
from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.connectome import Graph
from flyrl.regional_probe_types import ExtractionConfig, ProbeConfig
from flyrl.story_data import StorySplit
from scripts import alpn_causal_calibration_worker as worker
from scripts.alpn_causal_calibration_support import Context, snapshot
from scripts.alpn_causal_calibration_types import CalibrationSeal, FileIdentity, Runtime
from scripts.alpn_causal_source_archive import SOURCE_ARCHIVE_MEMBERS
from scripts.anatomy_port_artifacts import AnatomyIndices
from scripts.connectome_source import file_digest
from scripts.regional_probe_manifest import (
    ProbeGroupRecord,
    RegionalGroupManifest,
    SeedGroupPlan,
)
from scripts.regional_probe_protocol import ProbeSourceCheckpoint, RegionalProbeProtocol
from scripts.regional_probe_worker_support import WorkerInputs, parameter_fingerprint


def synthetic_context(
    root: Path, *, balanced: bool = False
) -> tuple[Context, dict[str, ARLearner]]:
    edges = np.arange(7, dtype=np.int64)
    graph = Graph(
        tuple(str(i) for i in range(7)),
        edges,
        (edges + 1) % 7,
        np.ones(7, dtype=np.float64),
        "synthetic",
    )
    if balanced:
        graph = Graph(
            graph.node_ids, edges[:1], edges[-1:], np.ones(1), "isolated pool"
        )
    models: dict[str, ARLearner] = {}
    sources: list[ProbeSourceCheckpoint] = []
    files: list[FileIdentity] = []
    source_dir = root / "sources"
    source_dir.mkdir()
    for seed in range(7, 13):
        for wiring in ("real", "shuffled"):
            config = ARConfig(
                alphabet_size=3,
                control=wiring,
                seed=seed,
                sensory_indices=(0,),
                readout_indices=(6,),
                readout_neurons=1,
                port_policy="random_random",
                port_manifest_sha256="a" * 64,
            )
            learner = ARLearner(graph, config)
            _ = learner.model.eval()
            checkpoint = root / "checkpoint.npz"
            save_checkpoint(learner, checkpoint, "a" * 64)
            _ = learner.model.requires_grad_(requires_grad=False)
            filename = f"seed-{seed}-{wiring}-random_random.zip"
            archive = source_dir / filename
            runtime = b'{"torch":"synthetic","cuda":"synthetic"}'
            with ZipFile(archive, "w") as output:
                output.write(checkpoint, "checkpoint.npz")
                output.writestr("report.json", b"{}")
                output.writestr("runtime.json", runtime)
                for member in SOURCE_ARCHIVE_MEMBERS:
                    if member not in {"checkpoint.npz", "report.json", "runtime.json"}:
                        output.writestr(member, b"synthetic artifact")
            sources.append(
                ProbeSourceCheckpoint(
                    seed=seed,
                    wiring=wiring,
                    filename=filename,
                    archive_sha256=file_digest(archive),
                    archive_bytes=archive.stat().st_size,
                    checkpoint_sha256=file_digest(checkpoint),
                    report_sha256=sha256(b"{}").hexdigest(),
                    runtime_sha256=sha256(runtime).hexdigest(),
                    parameter_fingerprint=parameter_fingerprint(learner),
                )
            )
            files.append(
                FileIdentity(
                    path=f"sources/{filename}",
                    sha256=file_digest(archive),
                    size=archive.stat().st_size,
                )
            )
            models[filename] = learner
    groups = RegionalGroupManifest(
        graph_sha256="a" * 64,
        port_manifest_sha256="a" * 64,
        port_arrays_sha256="a" * 64,
        selection_seed=20260913,
        group_size=1,
        draws=5,
        seeds=tuple(
            SeedGroupPlan(
                seed=seed,
                sensory_indices=(0,),
                trained_readout_indices=(6,),
                groups=tuple(
                    ProbeGroupRecord(
                        name="alpn", draw=draw, indices=(1,), node_ids=("1",)
                    )
                    for draw in range(5)
                ),
            )
            for seed in range(7, 13)
        ),
    )
    protocol = RegionalProbeProtocol(
        graph="graph.npz",
        graph_sha256="a" * 64,
        graph_nodes=7,
        graph_edges=graph.source.size,
        corpus="corpus.npz",
        corpus_sha256="a" * 64,
        corpus_fingerprint="a" * 64,
        port_manifest="ports.json",
        port_manifest_sha256="a" * 64,
        port_arrays="ports.npz",
        port_arrays_sha256="a" * 64,
        group_manifest="groups.json",
        group_manifest_sha256="a" * 64,
        source_sha256="a" * 64,
        worker_sha256="a" * 64,
        checkpoints=tuple(sources),
        extraction=ExtractionConfig(chunk_size=64, story_batch_size=1000),
        probe=ProbeConfig(vocab_size=3),
        group_size=1,
        draws=5,
        heads_per_checkpoint=5,
        practical_threshold=0.10,
        heldout_policy="sealed",
    )
    offsets = tuple(accumulate([231] * 745 + [230] * 255, initial=0))
    tokens = np.zeros(offsets[-1], dtype=np.int64)
    split = StorySplit(offsets=offsets, sha256=tuple(str(i) for i in range(1000)))
    seal = CalibrationSeal(
        protocol_sha256="a" * 64,
        files=tuple(files),
        training_tokens_sha256=sha256(tokens.tobytes()).hexdigest(),
        training_offsets_sha256=sha256(
            np.asarray(offsets, dtype=np.int64).tobytes()
        ).hexdigest(),
    )
    return Context(
        root,
        seal,
        WorkerInputs(graph, protocol, groups),
        AnatomyIndices((1, 2), (4,), (5,)),
        tokens,
        split,
    ), models


def restoration(
    models: dict[str, ARLearner],
) -> Callable[[Context, ProbeSourceCheckpoint], tuple[ARLearner, Runtime]]:
    def restore(
        _context: Context, source: ProbeSourceCheckpoint
    ) -> tuple[ARLearner, Runtime]:
        _ = sys.stdout.write("private boundary detail\n")
        return models[source.filename], Runtime(
            gpu="Tesla T4",
            device="cuda:0",
            dtype="torch.float32",
            python="synthetic",
            torch="synthetic",
            cuda="synthetic",
            threads=1,
            deterministic=False,
            tf32=False,
        )

    return restore


def test_real_extraction_all_seeds_metric_free_and_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, models = synthetic_context(tmp_path)
    before = {name: snapshot(model) for name, model in models.items()}
    monkeypatch.setattr(worker, "restore_checkpoint", restoration(models))
    result = worker.run_calibration(context, tmp_path / "result.zip")
    assert tuple(seed.seed for seed in result.seeds) == tuple(range(7, 13))
    assert result.status == "insufficient_common_support"
    assert all(
        seed.indices == (1, 2, 3) and seed.positions == 229745 for seed in result.seeds
    )
    assert all(len(seed.matching.controls) == 10 for seed in result.seeds)
    assert all(seed.matching.eligible == (3,) for seed in result.seeds)
    assert before == {name: snapshot(model) for name, model in models.items()}
    assert capsys.readouterr().out.splitlines() == [
        *(f"ALPN_CALIBRATION_SEED_READY seed={seed}" for seed in range(7, 13)),
        "ALPN_CALIBRATION_STATUS status=insufficient_common_support",
    ]
    with ZipFile(tmp_path / "result.zip") as archive:
        assert archive.namelist() == ["calibration.json"]
        assert archive.testzip() is None


@pytest.mark.parametrize(
    "kind", ["parameters", "buffers", "rng", "progress", "optimizer"]
)
def test_snapshot_detects_each_mutable_source_domain(tmp_path: Path, kind: str) -> None:
    _, models = synthetic_context(tmp_path)
    learner = next(iter(models.values()))
    before = snapshot(learner)
    with torch.no_grad():
        if kind == "parameters":
            _ = learner.model.bias.add_(1)
        elif kind == "buffers":
            learner.model.topology.forward_col[0] += 1
        elif kind == "rng":
            _ = learner.window_rng.manual_seed(99)
        elif kind == "progress":
            learner.updates += 1
        else:
            learner.optimizer.state[learner.model.bias] = {"step": torch.tensor(1.0)}
    assert before != snapshot(learner)


def test_partial_matrix_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    context, models = synthetic_context(tmp_path)
    restore = restoration(models)

    def fail(
        context: Context, source: ProbeSourceCheckpoint
    ) -> tuple[ARLearner, Runtime]:
        if source.seed == 8:
            message = "private failure 123.456"
            raise RuntimeError(message)
        return restore(context, source)

    monkeypatch.setattr(worker, "restore_checkpoint", fail)
    with pytest.raises(RuntimeError, match="private failure"):
        _ = worker.run_calibration(context, tmp_path / "result.zip")
    assert not (tmp_path / "result.zip").exists()
    assert capsys.readouterr().out == "ALPN_CALIBRATION_SEED_READY seed=7\n"
