"""Atomic pickle-free AdamW checkpoints with strict config/data/runtime identity."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import torch
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from pydantic import TypeAdapter

from flyrl.ar_config import ARConfig, TraceEntry
from flyrl.ar_framework import optimizer_tensors
from flyrl.ar_learning import ExperimentLearner
from flyrl.language_models import Settings
from flyrl.language_runtime import graph_fingerprint

if TYPE_CHECKING:
    from numpy import generic


class Metadata(Settings):
    """All mutable state belongs to model parameters, AdamW, progress and one RNG."""

    schema_version: Literal[1] = 1
    config: ARConfig
    graph: str
    corpus: str
    runtime: str
    updates: int
    trace: tuple[TraceEntry, ...]


def runtime_identity(learner: ExperimentLearner) -> str:
    """Record device, library and execution flags relevant to exact continuation."""
    device = learner.model.weight.device
    return json.dumps(
        {
            "torch": str(torch.__version__),
            "device": str(device),
            "name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
            "threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        sort_keys=True,
    )


def save_checkpoint(learner: ExperimentLearner, path: Path, corpus: str) -> None:
    """Flush a complete uncompressed NPZ then atomically replace the destination."""
    metadata = Metadata(
        config=learner.config,
        graph=graph_fingerprint(learner.graph),
        corpus=corpus,
        runtime=runtime_identity(learner),
        updates=learner.updates,
        trace=tuple(learner.trace),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with ZipFile(stream, "w", compression=ZIP_STORED) as archive:
                with archive.open("metadata.npy", "w") as member:
                    write_array(
                        member,
                        np.asarray(metadata.model_dump_json()),
                        allow_pickle=False,
                    )
                tensors = {"rng": learner.window_rng.get_state()}
                optimizer = optimizer_tensors(learner.optimizer)
                for name, parameter in learner.model.named_parameters():
                    tensors[f"model.{name}"] = parameter.detach()
                    if parameter in optimizer:
                        state = optimizer[parameter]
                        for key, value in state.items():
                            tensors[f"adam.{name}.{key}"] = value
                for name, tensor in tensors.items():
                    with archive.open(f"{name}.npy", "w") as member:
                        write_array(member, tensor.cpu().numpy(), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
            _ = temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _tensor(
    data: "NpzFile[generic]", key: str, reference: torch.Tensor
) -> torch.Tensor:
    array = data[key]
    expected_dtype = np.asarray(reference.detach().cpu().numpy()).dtype
    if (
        array.shape != tuple(reference.shape)
        or array.dtype != expected_dtype
        or not np.isfinite(array).all()
    ):
        message = f"Checkpoint tensor invariant violation: {key}"
        raise ValueError(message)
    return torch.tensor(array, device=reference.device)


def load_checkpoint(learner: ExperimentLearner, path: Path, corpus: str) -> None:
    """Validate all data before applying model, Adam moments and consumed RNG state."""
    with path.open("rb") as stream:
        data: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with data:
            metadata = Metadata.model_validate_json(
                TypeAdapter(str).validate_python(data["metadata"].item(), strict=True)
            )
            if (
                metadata.config != learner.config
                or metadata.graph != graph_fingerprint(learner.graph)
                or metadata.corpus != corpus
                or metadata.runtime != runtime_identity(learner)
            ):
                message = (
                    "Checkpoint compatibility mismatch: config/graph/corpus/runtime"
                )
                raise ValueError(message)
            if metadata.updates < 0 or tuple(x.update for x in metadata.trace) != tuple(
                range(1, metadata.updates + 1)
            ):
                message = "Checkpoint progress invariant violation"
                raise ValueError(message)
            rng = torch.Generator()
            _ = rng.set_state(_tensor(data, "rng", learner.window_rng.get_state()))
            parameters: dict[str, torch.Tensor] = {}
            moments: dict[str, dict[str, torch.Tensor]] = {}
            for name, parameter in learner.model.named_parameters():
                restored = _tensor(data, f"model.{name}", parameter)
                if not parameter.requires_grad and not torch.equal(restored, parameter):
                    message = "Checkpoint changed a frozen core"
                    raise ValueError(message)
                parameters[name] = restored
                if parameter.requires_grad and metadata.updates:
                    moments[name] = {
                        key: _tensor(data, f"adam.{name}.{key}", parameter)
                        for key in ("exp_avg", "exp_avg_sq")
                    }
                    step = _tensor(data, f"adam.{name}.step", torch.tensor(0.0))
                    if step.item() != metadata.updates or bool(
                        (moments[name]["exp_avg_sq"] < 0).any()
                    ):
                        message = "Checkpoint AdamW invariant violation"
                        raise ValueError(message)
                    moments[name]["step"] = step
    with torch.no_grad():
        learner.optimizer.state.clear()
        for name, parameter in learner.model.named_parameters():
            _ = parameter.copy_(parameters[name])
            if name in moments:
                learner.optimizer.state[parameter] = moments[name]
    learner.window_rng = rng
    learner.updates, learner.trace = metadata.updates, list(metadata.trace)
