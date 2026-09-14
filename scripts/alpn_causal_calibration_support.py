"""Old-training-only authentication and exact paired-source calibration context."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.npyio import NpzFile
from pydantic import TypeAdapter

from flyrl.ar_framework import optimizer_tensors
from flyrl.ar_learning import ARLearner
from flyrl.connectome import load_graph
from flyrl.language_data import IntVector
from flyrl.story_data import MIN_STORY_TOKENS, StoryMetadata, StorySplit
from scripts.alpn_causal_calibration import OLD_TRAINING_TARGETS
from scripts.alpn_causal_calibration_types import (
    CalibrationSeal,
    FileIdentity,
    Snapshot,
)
from scripts.anatomy_port_artifacts import (
    AnatomyIndices,
    AnatomyPortManifest,
    load_anatomy_indices,
)
from scripts.connectome_source import file_digest
from scripts.regional_probe_groups import RegionalSelection
from scripts.regional_probe_manifest import RegionalGroupManifest
from scripts.regional_probe_protocol import RegionalProbeProtocol
from scripts.regional_probe_worker_support import (
    WorkerInputs,
    parameter_fingerprint,
    project_file,
    validate_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch

PROTOCOL_SHA256 = "6b134d0e928caacf13900ccf793a65954e02dadef6e40eee014fa7feb9bb0dba"
TOKENIZER_SHA256 = "9dbe72484ae01b374801f3f7f8ffdcba3f126cb71774dc5f36ef75d06d064e27"


@dataclass(frozen=True, slots=True)
class Context:
    """Authenticated old training, graph, anatomy and frozen regional groups."""

    root: Path
    seal: CalibrationSeal
    inputs: WorkerInputs
    anatomy: AnatomyIndices
    tokens: IntVector
    split: StorySplit


def selection(context: Context, seed: int) -> RegionalSelection:
    """Use all anatomical ALPN, not only the five selected draws."""
    plan = next(plan for plan in context.inputs.groups.seeds if plan.seed == seed)
    anatomy = context.anatomy
    return RegionalSelection(
        anatomy.alpn,
        anatomy.mbon,
        anatomy.kenyon,
        plan.sensory_indices,
        plan.trained_readout_indices,
        seed,
    )


def selected_indices(context: Context, seed: int) -> tuple[int, ...]:
    """Only ALPN plus eligible candidates enter the old-training feature cache."""
    groups = selection(context, seed)
    excluded = set(
        groups.alpn
        + groups.mbon
        + groups.kenyon
        + groups.sensory
        + groups.trained_readout
    )
    return tuple(
        sorted(
            set(groups.alpn)
            | (set(range(len(context.inputs.graph.node_ids))) - excluded)
        )
    )


def check_files(root: Path, seal: CalibrationSeal) -> None:
    """Detect input mutation, traversal, duplicate paths and missing artifacts."""
    paths = tuple(item.path for item in seal.files)
    if len(set(paths)) != len(paths):
        message = "Duplicate calibration input path"
        raise ValueError(message)
    for item in seal.files:
        path = root / item.path
        if (
            Path(item.path).is_absolute()
            or ".." in Path(item.path).parts
            or not path.resolve().is_relative_to(root.resolve())
            or path.stat().st_size != item.size
            or file_digest(path) != item.sha256
        ):
            message = "Calibration sealed input hash or path differs"
            raise ValueError(message)


def old_training(
    root: Path,
    protocol: RegionalProbeProtocol,
    *,
    tokenizer_sha256: str = TOKENIZER_SHA256,
) -> tuple[IntVector, StorySplit]:
    """Read train and embedded identities only; never materialize valid/test arrays."""
    path = project_file(root, protocol.corpus)
    if file_digest(path) != protocol.corpus_sha256:
        message = "Old-training corpus hash differs"
        raise ValueError(message)
    with path.open("rb") as stream, NpzFile(stream, allow_pickle=False) as data:
        if len(data.files) != len(set(data.files)):
            message = "Duplicate old-training corpus member"
            raise ValueError(message)
        text = TypeAdapter(str)
        tokenizer = text.validate_python(data["tokenizer_json"].item(), strict=True)
        fingerprint = text.validate_python(data["fingerprint"].item(), strict=True)
        split = StoryMetadata.model_validate_json(
            text.validate_python(data["provenance"].item(), strict=True)
        ).train
        tokens = np.asarray(data["train"])
    if (
        fingerprint != protocol.corpus_fingerprint
        or sha256(tokenizer.encode()).hexdigest() != tokenizer_sha256
        or tokens.dtype != np.int64
        or tokens.ndim != 1
        or len(split.offsets) != len(split.sha256) + 1
        or split.offsets[0] != 0
        or split.offsets[-1] != tokens.size
        or any(
            b - a < MIN_STORY_TOKENS
            for a, b in zip(split.offsets[:-1], split.offsets[1:], strict=True)
        )
        or tokens.size - len(split.sha256) != OLD_TRAINING_TARGETS
        or bool(((tokens < 0) | (tokens >= protocol.probe.vocab_size)).any())
    ):
        message = "Old-training identity, boundaries or target count differs"
        raise ValueError(message)
    tokens.setflags(write=False)
    return tokens, split


def create_seal(root: Path) -> CalibrationSeal:
    """Build an offline input seal, without reading source reports or fresh text."""
    protocol_path = root / "protocol.json"
    if file_digest(protocol_path) != PROTOCOL_SHA256:
        message = "Frozen regional protocol hash differs"
        raise ValueError(message)
    protocol = RegionalProbeProtocol.model_validate_json(protocol_path.read_bytes())
    paths = [protocol_path, root / "pyproject.toml", root / "uv.lock"]
    for name, expected in (
        (protocol.graph, protocol.graph_sha256),
        (protocol.corpus, protocol.corpus_sha256),
        (protocol.port_manifest, protocol.port_manifest_sha256),
        (protocol.port_arrays, protocol.port_arrays_sha256),
        (protocol.group_manifest, protocol.group_manifest_sha256),
    ):
        path = project_file(root, name)
        if file_digest(path) != expected:
            message = "Frozen calibration data hash differs"
            raise ValueError(message)
        paths.append(path)
    for source in protocol.checkpoints:
        path = root / "sources" / source.filename
        if (
            file_digest(path) != source.archive_sha256
            or path.stat().st_size != source.archive_bytes
        ):
            message = "Frozen calibration source archive differs"
            raise ValueError(message)
        paths.append(path)
    for package in ("flyrl", "scripts"):
        paths.extend((root / package).rglob("*.py"))
    tokens, split = old_training(root, protocol)
    return CalibrationSeal(
        protocol_sha256=PROTOCOL_SHA256,
        files=tuple(
            FileIdentity(
                path=str(path.relative_to(root)),
                sha256=file_digest(path),
                size=path.stat().st_size,
            )
            for path in sorted(set(paths))
        ),
        training_tokens_sha256=sha256(tokens.tobytes()).hexdigest(),
        training_offsets_sha256=sha256(
            np.asarray(split.offsets, dtype=np.int64).tobytes()
        ).hexdigest(),
    )


def load_context(root: Path, seal: CalibrationSeal) -> Context:
    """Revalidate the separately trusted seal and exact regional groups before work."""
    check_files(root, seal)
    code = tuple(item for item in seal.files if item.path.endswith(".py"))
    check_files(
        Path(__file__).resolve().parents[1], seal.model_copy(update={"files": code})
    )
    if create_seal(root) != seal:
        message = "Calibration protocol/code/data/source seal differs"
        raise ValueError(message)
    protocol = RegionalProbeProtocol.model_validate_json(
        (root / "protocol.json").read_bytes()
    )
    inputs = WorkerInputs(
        load_graph(project_file(root, protocol.graph)),
        protocol,
        RegionalGroupManifest.model_validate_json(
            project_file(root, protocol.group_manifest).read_bytes()
        ),
    )
    validate_inputs(inputs)
    anatomy_manifest = AnatomyPortManifest.model_validate_json(
        project_file(root, protocol.port_manifest).read_bytes()
    )
    anatomy = load_anatomy_indices(
        project_file(root, protocol.port_arrays), anatomy_manifest
    )
    if (
        anatomy_manifest.graph_sha256 != protocol.graph_sha256
        or anatomy_manifest.arrays_sha256 != protocol.port_arrays_sha256
    ):
        message = "Calibration anatomy identity differs"
        raise ValueError(message)
    tokens, split = old_training(root, protocol)
    return Context(root, seal, inputs, anatomy, tokens, split)


def tensor_digest(tensors: "Iterable[tuple[str, torch.Tensor]]") -> str:
    """Hash typed shaped named tensors, including nonpersistent topology buffers."""
    digest = sha256()
    for name, tensor in tensors:
        digest.update(
            json.dumps((name, str(tensor.dtype), tuple(tensor.shape))).encode()
        )
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def snapshot(learner: ARLearner) -> Snapshot:
    """Capture all mutable checkpoint domains without changing any of them."""
    optimizer = optimizer_tensors(learner.optimizer)
    progress = (
        learner.updates,
        tuple(step.model_dump_json() for step in learner.trace),
        tuple(module.training for module in learner.model.modules()),
        tuple(p.requires_grad for p in learner.model.parameters()),
    )
    return Snapshot(
        parameters=parameter_fingerprint(learner),
        buffers=tensor_digest(learner.model.named_buffers()),
        optimizer=tensor_digest(
            (f"{name}.{key}", value)
            for name, parameter in learner.model.named_parameters()
            for key, value in sorted(optimizer.get(parameter, {}).items())
        ),
        progress=sha256(json.dumps(progress).encode()).hexdigest(),
        rng=tensor_digest((("window_rng", learner.window_rng.get_state()),)),
    )
