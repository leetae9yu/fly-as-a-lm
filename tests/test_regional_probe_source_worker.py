"""Exercise real extraction, fitting and packaging behind a synthetic CPU restore."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.connectome import Graph
from flyrl.language_runtime import graph_fingerprint
from flyrl.regional_probe_baseline import story_baselines
from flyrl.regional_probe_types import ExtractionConfig, ProbeConfig
from flyrl.story_data import prepare_stories
from flyrl.story_metrics import evaluate_heldout
from flyrl.story_pilot import PilotReport
from flyrl.story_source import Preparation
from scripts import regional_probe_source_worker
from scripts.connectome_source import file_digest
from scripts.regional_probe_artifact_types import SourceArtifact
from scripts.regional_probe_artifacts import load_head
from scripts.regional_probe_groups import SELECTION_SEED
from scripts.regional_probe_manifest import (
    ProbeGroupRecord,
    RegionalGroupManifest,
    SeedGroupPlan,
)
from scripts.regional_probe_protocol import ProbeSourceCheckpoint, RegionalProbeProtocol
from scripts.regional_probe_schema import EXPECTED_DRAWS, ProbeScore
from scripts.regional_probe_worker_support import WorkerInputs, parameter_fingerprint


def test_fourteen_real_fits_package_one_unchanged_synthetic_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    torch.set_num_threads(1)
    raw = tmp_path / "stories.txt"
    _ = raw.write_text("aa\n<|endoftext|>\nbb\n<|endoftext|>\ncc\n<|endoftext|>\n")
    stories = prepare_stories(
        Preparation(
            raw_files=(raw,),
            source="synthetic",
            revision="test",
            license_text="test",
            train_stories=1,
            valid_stories=1,
            test_stories=1,
            vocab_size=257,
        )
    )
    edges = np.arange(100, dtype=np.int64)
    graph = Graph(
        tuple(str(i) for i in range(100)),
        edges,
        (edges + 1) % 100,
        np.ones(100, dtype=np.float64),
        "synthetic",
    )
    config = ARConfig(
        alphabet_size=257, tokenization="bpe", trainable_codes=True, readout_neurons=2
    )
    learner = ARLearner(graph, config)
    metrics = evaluate_heldout(learner, stories)
    report = PilotReport(
        config=config,
        updates=0,
        graph_fingerprint=graph_fingerprint(graph),
        corpus_fingerprint=stories.corpus.fingerprint,
        graph_nodes=100,
        graph_edges=100,
        training_windows=1,
        trainable_parameters=sum(p.numel() for p in learner.model.parameters()),
        initial=metrics,
        final=metrics,
        prompt="",
        generated_token_ids={},
        generated_with_prompt={},
        activations=None,
        trace=(),
    )
    fingerprint = parameter_fingerprint(learner)
    _ = learner.model.eval()
    _ = learner.model.requires_grad_(requires_grad=False)
    directory = tmp_path / "sources/seed-7-real-random_random"
    directory.mkdir(parents=True)
    _ = (directory / "checkpoint.npz").write_bytes(b"synthetic restore boundary")
    _ = (directory / "report.json").write_text(report.model_dump_json())
    _ = (directory / "runtime.json").write_bytes(b"synthetic CPU runtime")
    source = ProbeSourceCheckpoint(
        seed=7,
        wiring="real",
        filename="seed-7-real-random_random.zip",
        archive_sha256="a" * 64,
        archive_bytes=1,
        checkpoint_sha256=file_digest(directory / "checkpoint.npz"),
        report_sha256=file_digest(directory / "report.json"),
        runtime_sha256=file_digest(directory / "runtime.json"),
        parameter_fingerprint=fingerprint,
    )
    indices = tuple(range(97))
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
        graph_sha256="b" * 64,
        port_manifest_sha256="c" * 64,
        port_arrays_sha256="d" * 64,
        selection_seed=SELECTION_SEED,
        group_size=97,
        draws=5,
        seeds=(
            SeedGroupPlan(
                seed=7,
                sensory_indices=(99,),
                trained_readout_indices=indices,
                groups=groups,
            ),
        ),
    )
    protocol = RegionalProbeProtocol(
        graph="graph.npz",
        graph_sha256=manifest.graph_sha256,
        graph_nodes=100,
        graph_edges=100,
        corpus="corpus.npz",
        corpus_sha256="e" * 64,
        corpus_fingerprint=stories.corpus.fingerprint,
        port_manifest="ports.json",
        port_manifest_sha256=manifest.port_manifest_sha256,
        port_arrays="ports.npz",
        port_arrays_sha256=manifest.port_arrays_sha256,
        group_manifest="regional_probe_groups.json",
        group_manifest_sha256="f" * 64,
        source_sha256="0" * 64,
        worker_sha256="1" * 64,
        checkpoints=(source,),
        extraction=ExtractionConfig(chunk_size=1),
        probe=ProbeConfig(vocab_size=257, updates=1, batch_size=2),
        group_size=97,
        draws=5,
        heads_per_checkpoint=14,
        practical_threshold=0.10,
        heldout_policy="sealed",
    )
    project = tmp_path / "flyrl-0.1.0"
    project.mkdir()
    _ = (project / "protocol.json").write_text(protocol.model_dump_json())

    def restore(
        directory: Path, source: ProbeSourceCheckpoint, inputs: WorkerInputs
    ) -> tuple[ARLearner, PilotReport]:
        assert directory.name == Path(source.filename).stem
        assert inputs.protocol == protocol
        return learner, report

    monkeypatch.setattr(regional_probe_source_worker, "restore_source", restore)
    unigram, bigram = story_baselines(
        stories.corpus.train,
        stories.metadata.train,
        stories.corpus.test,
        stories.metadata.test,
        257,
    )
    results, baselines = regional_probe_source_worker.run_source(
        tmp_path,
        WorkerInputs(graph, protocol, manifest),
        stories,
        source,
        (
            ProbeScore.model_validate(unigram.model_dump()),
            ProbeScore.model_validate(bigram.model_dump()),
        ),
    )
    assert len(results) == 14
    assert baselines.original_head_nll == report.final.test.nll
    assert parameter_fingerprint(learner) == fingerprint
    assert all(not parameter.requires_grad for parameter in learner.model.parameters())
    assert capsys.readouterr().out == ""
    output = tmp_path / "results/seed-7-real"
    artifact = SourceArtifact.model_validate_json((output / "source.json").read_bytes())
    assert artifact.source_parameter_count == report.trainable_parameters
    assert artifact.runtime.gpu == "CPU"
    assert artifact.heads == tuple(
        f"{group.name}-{group.draw}.json" for group in groups
    )
    assert artifact.baselines == baselines
    for name in artifact.heads:
        head, result = load_head(output / name)
        assert head.source == source
        assert (
            len(result.trace),
            result.parameter_count,
            result.config.seed,
        ) == (1, 98 * 257, protocol.probe.seed + source.seed)
    with ZipFile(tmp_path / "seed-7-real-probes.zip") as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 30
        assert set(archive.namelist()) == {path.name for path in output.iterdir()}
