"""Atomic pickle-free checkpoints with strict compatibility and tensor validation."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Annotated, Literal
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import numpy as np
import torch
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from pydantic import Field, TypeAdapter, ValidationError

from flyrl.language_learning import CharacterLearner
from flyrl.language_models import LanguageConfig, LanguageError, Settings
from flyrl.language_runtime import (
    device_evidence,
    graph_fingerprint,
    tensor_fingerprint,
)

if TYPE_CHECKING:
    from numpy import generic


class Metadata(Settings):
    """Checkpoint protocol; all consumed randomness belongs to two generators."""

    schema_version: Literal[1] = 1
    config: LanguageConfig
    graph: str
    corpus: str
    weights: str
    torch_version: str
    device_name: str
    updates: Annotated[int, Field(ge=0)]
    baseline: Annotated[float, Field(ge=0, le=1)]
    rewards: tuple[Annotated[float, Field(ge=0, le=1)], ...]


def atomic_text(path: Path, content: str) -> None:
    """Replace a report atomically after flushing its full contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False, mode="w") as stream:
        temporary = Path(stream.name)
        try:
            _ = stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            _ = temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def save_checkpoint(learner: CharacterLearner, path: Path, corpus: str) -> None:
    """Save exact float32 weights, EMA/progress, and both generator byte states."""
    evidence = device_evidence(learner.weights, learner.config.device)
    metadata = Metadata(
        config=learner.config,
        graph=graph_fingerprint(learner.graph),
        corpus=corpus,
        weights=tensor_fingerprint(learner.weights),
        torch_version=evidence.torch_version,
        device_name=evidence.name,
        updates=learner.updates,
        baseline=learner.baseline,
        rewards=tuple(learner.rewards),
    )
    arrays = (
        ("metadata", np.asarray(metadata.model_dump_json())),
        ("weights", np.asarray(learner.weights.cpu().numpy(), dtype=np.float32)),
        (
            "policy_rng",
            np.asarray(learner.policy_rng.get_state().numpy(), dtype=np.uint8),
        ),
        (
            "window_rng",
            np.asarray(learner.window_rng.get_state().numpy(), dtype=np.uint8),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
                for name, array in arrays:
                    with archive.open(f"{name}.npy", "w") as member:
                        write_array(member, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
            _ = temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_checkpoint(learner: CharacterLearner, path: Path, corpus: str) -> None:
    """Parse and validate every field before committing restored mutable state."""
    try:
        with path.open("rb") as stream:
            archive: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
            with archive as data:
                metadata = Metadata.model_validate_json(
                    TypeAdapter(str).validate_python(
                        data["metadata"].item(), strict=True
                    )
                )
                if data["weights"].dtype != np.float32:
                    raise LanguageError(reason="checkpoint weights must be float32")
                weights = torch.tensor(
                    np.asarray(data["weights"], dtype=np.float32), device=learner.device
                )
                states: list[torch.Tensor] = []
                for key in ("policy_rng", "window_rng"):
                    if data[key].dtype != np.uint8 or data[key].ndim != 1:
                        raise LanguageError(
                            reason="checkpoint RNG state must be uint8 vector"
                        )
                    states.append(torch.tensor(np.asarray(data[key], dtype=np.uint8)))
    except (BadZipFile, KeyError, ValueError, EOFError, ValidationError) as error:
        raise LanguageError(reason=f"malformed checkpoint: {error}") from error
    evidence = device_evidence(learner.weights, learner.config.device)
    if (
        metadata.config != learner.config
        or metadata.graph != graph_fingerprint(learner.graph)
        or metadata.corpus != corpus
        or metadata.torch_version != evidence.torch_version
        or metadata.device_name != evidence.name
    ):
        raise LanguageError(
            reason="checkpoint compatibility mismatch: graph/corpus/config/runtime"
        )
    if (
        weights.shape != learner.weights.shape
        or not bool(torch.isfinite(weights).all())
        or metadata.weights != tensor_fingerprint(weights)
        or not torch.equal(weights.sign(), learner.signs)
        or bool((weights.abs() > learner.config.max_weight).any())
        or not torch.equal(
            weights[learner.plastic == 0], learner.weights[learner.plastic == 0]
        )
        or (learner.config.frozen and not torch.equal(weights, learner.weights))
        or len(metadata.rewards) != metadata.updates
    ):
        raise LanguageError(reason="checkpoint tensor or progress invariant violation")
    policy = torch.Generator(device=learner.device)
    window = torch.Generator(device=learner.device)
    try:
        _ = policy.set_state(states[0])
        _ = window.set_state(states[1])
    except RuntimeError as error:
        raise LanguageError(reason="invalid checkpoint RNG state") from error
    learner.weights = weights
    learner.policy_rng, learner.window_rng = policy, window
    learner.baseline, learner.updates = metadata.baseline, metadata.updates
    learner.rewards = list(metadata.rewards)
