"""Strict sealed-input identity and group validation for the remote probe worker."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, cast

import torch
from numpy.lib.npyio import NpzFile
from pydantic import TypeAdapter

from flyrl.ar_checkpoint import Metadata, load_checkpoint
from flyrl.ar_config import parameter_identity_json
from flyrl.ar_learning import ARLearner
from flyrl.connectome import Graph
from flyrl.language_runtime import graph_fingerprint
from flyrl.regional_probe import PROBE_WIDTH
from flyrl.story_pilot import PilotReport
from scripts.anatomy_factorial_recovery_schema import CheckpointRuntime, WorkerRuntime
from scripts.connectome_source import file_digest
from scripts.regional_probe_groups import SELECTION_SEED
from scripts.regional_probe_manifest import RegionalGroupManifest, SeedGroupPlan
from scripts.regional_probe_protocol import ProbeSourceCheckpoint, RegionalProbeProtocol
from scripts.regional_probe_schema import EXPECTED_DRAWS, SEEDS, WIRINGS

VOCAB_SIZE: Final = 4096
PRACTICAL_THRESHOLD: Final = 0.10


@dataclass(frozen=True, slots=True)
class WorkerInputs:
    """Validated immutable source model construction inputs."""

    graph: Graph
    protocol: RegionalProbeProtocol
    groups: RegionalGroupManifest


def project_file(project: Path, name: str) -> Path:
    """Resolve a sealed relative name, supporting the existing data directory layout."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        message = "Protocol paths must stay inside the project"
        raise ValueError(message)
    candidates = {project / relative, project / "data/central_connectome" / relative}
    existing = tuple(path for path in candidates if path.is_file())
    if len(existing) != 1:
        message = "Protocol artifact is missing or ambiguous"
        raise ValueError(message)
    return existing[0]


def cuda_device_index(device: torch.device) -> int:
    """Resolve bare ``cuda`` to the runtime's current concrete device index."""
    index = cast("int | None", device.index)
    return torch.cuda.current_device() if index is None else index


def validate_inputs(inputs: WorkerInputs) -> None:
    """Require the exact ordered twelve sources and six fourteen-head seed plans."""
    protocol, manifest, graph = inputs.protocol, inputs.groups, inputs.graph
    expected_sources = tuple((seed, wiring) for seed in SEEDS for wiring in WIRINGS)
    expected_groups = tuple(
        (name, draw) for name, count in EXPECTED_DRAWS.items() for draw in range(count)
    )
    if (
        tuple((source.seed, source.wiring) for source in protocol.checkpoints)
        != expected_sources
        or tuple(plan.seed for plan in manifest.seeds) != SEEDS
        or protocol.group_size != PROBE_WIDTH
        or manifest.group_size != PROBE_WIDTH
        or protocol.draws != EXPECTED_DRAWS["alpn"]
        or manifest.draws != EXPECTED_DRAWS["alpn"]
        or protocol.heads_per_checkpoint != len(expected_groups)
        or protocol.probe.vocab_size != VOCAB_SIZE
        or protocol.practical_threshold != PRACTICAL_THRESHOLD
        or manifest.selection_seed != SELECTION_SEED
        or manifest.graph_sha256 != protocol.graph_sha256
        or manifest.port_manifest_sha256 != protocol.port_manifest_sha256
        or manifest.port_arrays_sha256 != protocol.port_arrays_sha256
        or len(graph.node_ids) != protocol.graph_nodes
        or graph.source.size != protocol.graph_edges
    ):
        message = "Regional probe protocol or manifest identities differ"
        raise ValueError(message)
    for source in protocol.checkpoints:
        if source.filename != f"seed-{source.seed}-{source.wiring}-random_random.zip":
            message = "Regional probe source filename differs"
            raise ValueError(message)
    for plan in manifest.seeds:
        if (
            tuple((group.name, group.draw) for group in plan.groups) != expected_groups
            or len(set(plan.sensory_indices)) != len(plan.sensory_indices)
            or not plan.sensory_indices
            or len(set(plan.trained_readout_indices)) != protocol.group_size
            or bool(set(plan.sensory_indices) & set(plan.trained_readout_indices))
            or any(
                index < 0 or index >= protocol.graph_nodes
                for index in (*plan.sensory_indices, *plan.trained_readout_indices)
            )
        ):
            message = "Regional probe group order or ports differ"
            raise ValueError(message)
        for group in plan.groups:
            if (
                len(group.indices) != protocol.group_size
                or tuple(sorted(set(group.indices))) != group.indices
                or any(
                    index < 0 or index >= protocol.graph_nodes
                    for index in group.indices
                )
                or group.node_ids
                != tuple(graph.node_ids[index] for index in group.indices)
                or (
                    group.name == "trained_readout"
                    and group.indices != tuple(sorted(plan.trained_readout_indices))
                )
            ):
                message = "Regional group capacity, order or node identity differs"
                raise ValueError(message)


def source_files(directory: Path, source: ProbeSourceCheckpoint) -> None:
    """Check original extracted source bytes before and after all source work."""
    for filename, expected in (
        ("checkpoint.npz", source.checkpoint_sha256),
        ("report.json", source.report_sha256),
        ("runtime.json", source.runtime_sha256),
    ):
        if file_digest(directory / filename) != expected:
            message = "Regional probe source bytes changed"
            raise ValueError(message)


def parameter_fingerprint(learner: ARLearner) -> str:
    """Bind exact model parameters to the original export's configuration identity."""
    digest = sha256(parameter_identity_json(learner.config).encode())
    for name, parameter in learner.model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            message = "Regional probe source parameters are nonfinite"
            raise ValueError(message)
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def restore_source(
    directory: Path, source: ProbeSourceCheckpoint, inputs: WorkerInputs
) -> tuple[ARLearner, PilotReport]:
    """Restore the exact source learner, runtime flags and progress, then freeze it."""
    source_files(directory, source)
    report = PilotReport.model_validate_json((directory / "report.json").read_bytes())
    runtime = WorkerRuntime.model_validate_json(
        (directory / "runtime.json").read_bytes()
    )
    plan = next(plan for plan in inputs.groups.seeds if plan.seed == source.seed)
    protocol, config = inputs.protocol, report.config
    if (
        config.seed != source.seed
        or config.control != source.wiring
        or config.port_policy != "random_random"
        or config.architecture != "connectome"
        or config.alphabet_size != protocol.probe.vocab_size
        or config.sensory_indices != plan.sensory_indices
        or config.readout_indices != plan.trained_readout_indices
        or config.port_manifest_sha256 != protocol.port_manifest_sha256
        or report.graph_fingerprint != graph_fingerprint(inputs.graph)
        or report.graph_nodes != protocol.graph_nodes
        or report.graph_edges != protocol.graph_edges
        or report.corpus_fingerprint != protocol.corpus_fingerprint
        or runtime.seed != source.seed
        or runtime.wiring != source.wiring
        or runtime.port_policy != "random_random"
        or runtime.condition != f"{source.wiring}-random_random"
        or runtime.torch != str(torch.__version__)
        or runtime.cuda != torch.version.cuda
    ):
        message = "Regional probe source report or runtime identity differs"
        raise ValueError(message)
    with (
        (directory / "checkpoint.npz").open("rb") as stream,
        NpzFile(stream, allow_pickle=False) as data,
    ):
        metadata = Metadata.model_validate_json(
            TypeAdapter(str).validate_python(data["metadata"].item(), strict=True)
        )
    checkpoint_runtime = CheckpointRuntime.model_validate_json(metadata.runtime)
    device = torch.device(config.device)
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.get_device_name(device) != "Tesla T4"
        or metadata.config != config
        or metadata.updates != report.updates
        or metadata.trace != report.trace
        or checkpoint_runtime.torch != runtime.torch
    ):
        message = "Regional probe requires the exact Tesla T4 source checkpoint"
        raise ValueError(message)
    torch.cuda.set_device(cuda_device_index(device))
    torch.set_num_threads(checkpoint_runtime.threads)
    torch.use_deterministic_algorithms(checkpoint_runtime.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = checkpoint_runtime.tf32
    learner = ARLearner(inputs.graph, config)
    load_checkpoint(learner, directory / "checkpoint.npz", protocol.corpus_fingerprint)
    if (
        parameter_fingerprint(learner) != source.parameter_fingerprint
        or sum(p.numel() for p in learner.model.parameters() if p.requires_grad)
        != report.trainable_parameters
    ):
        message = "Regional probe restored parameter identity differs"
        raise ValueError(message)
    _ = learner.model.eval()
    _ = learner.model.requires_grad_(requires_grad=False)
    return learner, report


def union_indices(plan: SeedGroupPlan) -> tuple[int, ...]:
    """Extract each seed's union once in stable increasing graph-index order."""
    return tuple(sorted({index for group in plan.groups for index in group.indices}))
