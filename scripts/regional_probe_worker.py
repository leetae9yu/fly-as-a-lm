# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.6,<3", "numpy>=2,<3", "pydantic>=2.10,<3",
#   "typer>=0.15,<1", "tokenizers==0.22.2", "matplotlib>=3.10,<4",
# ]
# ///
"""Run the sealed T4 regional probe matrix without exposing heldout metrics."""

import logging
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Final
from zipfile import BadZipFile

import torch

from flyrl.connectome import load_graph
from flyrl.regional_probe_baseline import story_baselines
from flyrl.story_data import load_story_corpus
from scripts.connectome_source import file_digest
from scripts.regional_probe_artifacts import seal_zip
from scripts.regional_probe_manifest import RegionalGroupManifest
from scripts.regional_probe_protocol import RegionalProbeProtocol
from scripts.regional_probe_schema import (
    ProbeBaselines,
    ProbeResult,
    ProbeScore,
    evaluate_regional_probes,
)
from scripts.regional_probe_source_worker import run_source
from scripts.regional_probe_worker_support import (
    WorkerInputs,
    project_file,
    source_files,
    validate_inputs,
)

PROGRESS: Final = sys.stdout
LOGGER: Final = logging.getLogger(__name__)


def check_files(identities: dict[Path, str]) -> None:
    """Reject any mutation of the sealed protocol, worker, corpus, graph or ports."""
    if any(file_digest(path) != expected for path, expected in identities.items()):
        message = "Regional probe sealed input hash differs"
        raise ValueError(message)


def run(root: Path) -> None:
    """Complete all 168 heads before aggregating the frozen prospective gate."""
    project = root / "flyrl-0.1.0"
    protocol_path = project / "protocol.json"
    protocol = RegionalProbeProtocol.model_validate_json(protocol_path.read_bytes())
    identities = {
        protocol_path: file_digest(protocol_path),
        Path(__file__): protocol.worker_sha256,
    }
    for name, digest in (
        (protocol.graph, protocol.graph_sha256),
        (protocol.corpus, protocol.corpus_sha256),
        (protocol.port_manifest, protocol.port_manifest_sha256),
        (protocol.port_arrays, protocol.port_arrays_sha256),
        (protocol.group_manifest, protocol.group_manifest_sha256),
    ):
        identities[project_file(project, name)] = digest
    check_files(identities)
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != "Tesla T4":
        message = "Regional probe requires an actual Tesla T4"
        raise ValueError(message)
    if (
        protocol.probe.device.startswith("cuda")
        and torch.cuda.get_device_name(torch.device(protocol.probe.device))
        != "Tesla T4"
    ):
        message = "Regional probe head fitting requires the sealed Tesla T4"
        raise ValueError(message)
    inputs = WorkerInputs(
        load_graph(project_file(project, protocol.graph)),
        protocol,
        RegionalGroupManifest.model_validate_json(
            project_file(project, protocol.group_manifest).read_bytes()
        ),
    )
    validate_inputs(inputs)
    stories = load_story_corpus(project_file(project, protocol.corpus))
    if (
        stories.corpus.fingerprint != protocol.corpus_fingerprint
        or len(stories.corpus.vocabulary) != protocol.probe.vocab_size
    ):
        message = "Regional probe corpus identity differs"
        raise ValueError(message)
    unigram, bigram = story_baselines(
        stories.corpus.train,
        stories.metadata.train,
        stories.corpus.test,
        stories.metadata.test,
        protocol.probe.vocab_size,
    )
    scores = (
        ProbeScore.model_validate(unigram.model_dump()),
        ProbeScore.model_validate(bigram.model_dump()),
    )
    results: list[ProbeResult] = []
    baselines: list[ProbeBaselines] = []
    archives: list[Path] = []
    archive_hashes: dict[Path, str] = {}
    for source in protocol.checkpoints:
        source_results, baseline = run_source(root, inputs, stories, source, scores)
        results.extend(source_results)
        baselines.append(baseline)
        archive = root / f"seed-{source.seed}-{source.wiring}-probes.zip"
        archives.append(archive)
        archive_hashes[archive] = file_digest(archive)
        check_files(identities)
        _ = PROGRESS.write(
            f"REGIONAL_SOURCE_READY seed={source.seed} wiring={source.wiring}\n"
        )
        _ = PROGRESS.flush()
    for source in protocol.checkpoints:
        source_files(root / "sources" / Path(source.filename).stem, source)
    summary = evaluate_regional_probes(
        tuple(results), tuple(baselines), protocol.practical_threshold
    )
    summary_path = root / "summary.json"
    _ = summary_path.write_text(summary.model_dump_json() + "\n")
    check_files(identities | archive_hashes)
    seal_zip(
        root / "regional-probe-results.zip", (protocol_path, summary_path, *archives)
    )
    _ = PROGRESS.write("REGIONAL_PROBE_RESULTS_PACKAGED\n")
    _ = PROGRESS.flush()


def main() -> None:
    """Keep library output and failure details off the metric-free control channel."""
    root = Path("/content/regional-probe")
    with (
        (root / "worker.log").open("w") as stream,
        redirect_stdout(stream),
        redirect_stderr(stream),
    ):
        try:
            run(root)
        except (ValueError, RuntimeError, OSError, KeyError, BadZipFile):
            LOGGER.exception("Regional probe execution failed")
            message = "Regional probe failed; inspect sealed worker.log"
            raise SystemExit(message) from None


if __name__ == "__main__":
    main()
