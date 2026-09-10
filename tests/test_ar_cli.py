"""Execute the real Typer surface, reports, resumption and capacity measurement."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from flyrl.ar_benchmark import Benchmark
from flyrl.ar_reporting import RunReport
from flyrl.autoregressive import app
from flyrl.connectome import Graph, save_graph
from flyrl.language_data import CorpusLimits, TextSplits, make_corpus, save_corpus


@pytest.mark.parametrize("progress", [False, True])
def test_cli_controls_resume_and_capacity_surface(
    tmp_path: Path, progress: bool
) -> None:
    nodes = 12
    source, target = np.nonzero(np.ones((nodes, nodes)) - np.eye(nodes))
    graph_path, corpus_path = tmp_path / "graph.npz", tmp_path / "corpus.npz"
    save_graph(
        Graph(
            tuple(f"synthetic:{i}" for i in range(nodes)),
            source.astype(np.int64),
            target.astype(np.int64),
            np.ones(source.size, dtype=np.float64),
            "CLI synthetic fixture",
        ),
        graph_path,
    )
    text = "ab" * 50
    save_corpus(
        make_corpus(TextSplits(text, text, text, "synthetic"), CorpusLimits()),
        corpus_path,
    )
    command = [
        "--graph",
        str(graph_path),
        "--corpus",
        str(corpus_path),
        "--output",
        str(tmp_path / "runs"),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--context",
        "3",
        "--seeds",
        "0",
        "--controls",
        "real,shuffled,frozen",
        "--eval-windows",
        "4",
        "--sample-length",
        "5",
        "--checkpoint-steps",
        "1",
    ]
    if progress:
        command.append("--progress")
    first = CliRunner().invoke(app, [*command, "--updates", "1"])
    assert first.exit_code == 0, first.output
    reports = [
        RunReport.model_validate_json(line) for line in first.stdout.splitlines()
    ]
    assert {report.config.control for report in reports} == {
        "real",
        "shuffled",
        "frozen",
    }
    assert all(report.device["actual"] == "cpu" for report in reports)
    if progress:
        adapter = TypeAdapter(dict[str, str | int | float])
        events = [
            adapter.validate_json(line.removeprefix("AR_PROGRESS "))
            for line in first.stderr.splitlines()
            if line.startswith("AR_PROGRESS ")
        ]
        assert [event["update"] for event in events] == [1, 1, 1]
        assert {event["control"] for event in events} == {"real", "shuffled", "frozen"}
    resumed = CliRunner().invoke(app, [*command, "--updates", "2", "--resume"])
    assert resumed.exit_code == 0, resumed.output
    for line, initial in zip(resumed.stdout.splitlines(), reports, strict=True):
        report = RunReport.model_validate_json(line)
        assert report.updates == 2
        assert report.initial == initial.initial
        assert len(report.trace) == 2
        assert len(report.generated["sampled"]) == 5
        assert (
            report.final["test"].windows == report.baselines["test"]["unigram"].windows
        )
    probe = CliRunner().invoke(app, [*command, "--benchmark-only"])
    assert probe.exit_code == 0, probe.output
    measurements = [
        Benchmark.model_validate_json(line) for line in probe.stdout.splitlines()
    ]
    assert all(value.edges == source.size for value in measurements)
    assert all(value.backward_seconds > 0 for value in measurements)
    stored = tmp_path / "runs" / "seed-0" / "real" / "report.json"
    assert RunReport.model_validate_json(stored.read_text()).updates == 2
    trace = TypeAdapter(list[dict[str, float]]).validate_python(
        json.loads((stored.parent / "trace.json").read_text())
    )
    assert len(trace) == 2
