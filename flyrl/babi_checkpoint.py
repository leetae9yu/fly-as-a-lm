"""Atomic immutable no-pickle transactions with validation before learner mutation."""

import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import torch
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile

from flyrl.ar_framework import optimizer_tensors
from flyrl.babi_data import BabiError, Record
from flyrl.babi_learning import BabiLearner
from flyrl.babi_pilot_schema import Manifest, Progress, Validation
from flyrl.babi_protocol import BabiProtocol, CodeFile, runtime_identity
from flyrl.babi_recovery_validation import checkpoint as verify_checkpoint
from flyrl.language_checkpoint import atomic_text
from flyrl.language_runtime import graph_fingerprint
from scripts.connectome_source import file_digest


def publish(path: Path, record: Record) -> None:
    """Publish fsynced immutable JSON, accepting only identical existing bytes."""
    text = record.model_dump_json(exclude_computed_fields=True)
    if path.exists():
        if path.read_text() != text:
            raise BabiError(reason=f"Immutable record collision: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        _ = stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def timed_copy(checkpoint: Path, destination: Path) -> float:
    """Measure a real verified recovery write on the external destination."""
    with TemporaryDirectory(dir=destination) as temporary:
        started = monotonic()
        copied = Path(temporary) / "disposable.npz"
        _ = shutil.copyfile(checkpoint, copied)
        if file_digest(copied) != file_digest(checkpoint):
            raise BabiError(reason="Recovery timing checksum mismatch")
        return monotonic() - started


def tensor(
    data: NpzFile[np.generic], key: str, reference: torch.Tensor
) -> torch.Tensor:
    """Parse one exact-shape, exact-dtype finite tensor at the checkpoint boundary."""
    array = data[key]
    if (
        array.shape != tuple(reference.shape)
        or array.dtype != reference.detach().cpu().numpy().dtype
        or not np.isfinite(array).all()
    ):
        raise BabiError(reason=f"Checkpoint tensor mismatch: {key}")
    return torch.tensor(array, device=reference.device)


@dataclass(frozen=True, slots=True)
class Checkpoints:
    """One run's transaction namespace and its immutable compatibility contract."""

    output: Path
    protocol: BabiProtocol
    training_ids: tuple[str, ...]

    def path(self, update: int) -> Path:
        """Address an immutable transaction, including its manifest and NPZ."""
        return self.output / f"update-{update:06d}"

    def save(self, learner: BabiLearner, history: tuple[Validation, ...]) -> Manifest:
        """Commit a whole directory before advancing the atomic latest pointer."""
        progress = Progress(
            protocol_sha256=self.protocol.sha256,
            updates=learner.updates,
            next_lr=self.protocol.next_lr(learner.updates),
            trace=tuple(learner.trace),
            exposures=tuple(
                (identity, learner.exposure_counts[identity])
                for identity in self.training_ids
            ),
            validation=history,
            validation_sha256=tuple(
                file_digest(self.output / f"validation-{v.update:06d}.json")
                for v in history
                if (self.output / f"validation-{v.update:06d}.json").exists()
            ),
        )
        progress.verify(self.protocol, self.training_ids)
        self.output.mkdir(parents=True, exist_ok=True)
        destination = self.path(learner.updates)
        with TemporaryDirectory(dir=self.output) as temporary:
            folder = Path(temporary) / "transaction"
            folder.mkdir()
            checkpoint = folder / "checkpoint.npz"
            self.write_payload(learner, checkpoint, progress)
            manifest = Manifest(
                checkpoint_sha256=file_digest(checkpoint), progress=progress
            )
            publish(folder / "manifest.json", manifest)
            if destination.exists():
                if (destination / "manifest.json").read_bytes() != (
                    folder / "manifest.json"
                ).read_bytes() or file_digest(
                    destination / "checkpoint.npz"
                ) != manifest.checkpoint_sha256:
                    raise BabiError(reason="Immutable checkpoint transaction collision")
            else:
                _ = folder.rename(destination)
        atomic_text(self.output / "latest.json", manifest.model_dump_json())
        self._retain()
        return manifest

    def write_payload(
        self, learner: BabiLearner, path: Path, progress: Progress
    ) -> None:
        """Write optimizer-bearing bytes, also used for disposable timing fixtures."""
        tensors = {"rng": learner.window_rng.get_state()}
        states = optimizer_tensors(learner.optimizer)
        for name, parameter in learner.model.named_parameters():
            tensors[f"model.{name}"] = parameter.detach()
            if parameter in states:
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    tensors[f"adam.{name}.{key}"] = states[parameter][key]
        with path.open("wb") as stream:
            with ZipFile(stream, "w", compression=ZIP_STORED) as archive:
                with archive.open("progress.npy", "w") as member:
                    write_array(
                        member,
                        np.asarray(progress.model_dump_json()),
                        allow_pickle=False,
                    )
                for name, value in tensors.items():
                    if not bool(torch.isfinite(value).all()):
                        raise BabiError(reason="Nonfinite checkpoint tensor")
                    with archive.open(f"{name}.npy", "w") as member:
                        write_array(member, value.cpu().numpy(), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())

    def manifest(self, update: int | None = None) -> Manifest:
        """Verify immutable bytes and latest pointer without constructing a learner."""
        pointer = (
            self.output / "latest.json"
            if update is None
            else self.path(update) / "manifest.json"
        )
        manifest = Manifest.model_validate_json(pointer.read_bytes())
        progress = manifest.progress
        progress.verify(self.protocol, self.training_ids)
        folder = self.path(progress.updates)
        if (
            Manifest.model_validate_json((folder / "manifest.json").read_bytes())
            != manifest
            or file_digest(folder / "checkpoint.npz") != manifest.checkpoint_sha256
        ):
            raise BabiError(reason="Checkpoint manifest identity mismatch")
        if progress.validation_sha256 and progress.validation_sha256 != tuple(
            file_digest(self.output / f"validation-{v.update:06d}.json")
            for v in progress.validation
        ):
            raise BabiError(reason="Checkpoint validation evidence identity mismatch")
        return manifest

    def load(self, learner: BabiLearner, update: int | None = None) -> Manifest:
        """Stage every tensor, history and replayed RNG before applying any mutation."""
        manifest = self.manifest(update)
        progress = manifest.progress
        _ = verify_checkpoint(
            self.path(progress.updates), self.protocol, self.training_ids
        )
        if (
            learner.config != self.protocol.config
            or graph_fingerprint(learner.graph) != self.protocol.graph.fingerprint
            or runtime_identity(learner.config.device) != self.protocol.runtime
        ):
            raise BabiError(reason="Checkpoint model/graph/runtime identity mismatch")
        with (
            (self.path(progress.updates) / "checkpoint.npz").open("rb") as stream,
            NpzFile[np.generic](stream, allow_pickle=False) as data,
        ):
            rng = torch.Generator()
            _ = rng.set_state(tensor(data, "rng", rng.get_state()))
            parameters: dict[str, torch.Tensor] = {}
            moments: dict[str, dict[str, torch.Tensor]] = {}
            for name, parameter in learner.model.named_parameters():
                key = f"model.{name}"
                parameters[name] = tensor(data, key, parameter)
                if parameter.requires_grad and progress.updates:
                    state = {
                        "step": tensor(data, f"adam.{name}.step", torch.tensor(0.0))
                    }
                    state.update(
                        {
                            k: tensor(data, f"adam.{name}.{k}", parameter)
                            for k in ("exp_avg", "exp_avg_sq")
                        }
                    )
                    moments[name] = state
        with torch.no_grad():
            learner.optimizer.state.clear()
            for name, parameter in learner.model.named_parameters():
                _ = parameter.copy_(parameters[name])
                if name in moments:
                    learner.optimizer.state[parameter] = moments[name]
        learner.window_rng = rng
        learner.updates, learner.trace = progress.updates, list(progress.trace)
        learner.exposure_counts = Counter(dict(progress.exposures))
        for group in learner.optimizer.param_groups:
            group["lr"] = progress.next_lr or self.protocol.schedule.rate(
                progress.updates
            )
        return manifest

    def _retain(self) -> None:
        paths = sorted(self.output.glob("update-*"))
        keep = {self.path(u) for u in self.protocol.milestones} | set(paths[-2:])
        for path in paths:
            if path not in keep:
                shutil.rmtree(path)

    def recover(self, destination: Path) -> tuple[CodeFile, ...]:
        """Copy a recovery bundle and independently verify every copied hash."""
        if destination.resolve().is_relative_to(self.output.resolve()):
            raise BabiError(reason="Recovery must be outside run output")
        manifest = self.manifest()
        target = destination / f"recovery-{manifest.progress.updates:06d}"
        files = tuple(
            p
            for p in self.output.rglob("*")
            if p.is_file()
            and not any(
                part.startswith("tmp") for part in p.relative_to(self.output).parts
            )
        )
        identities = tuple(
            CodeFile(path=p.relative_to(self.output), sha256=file_digest(p))
            for p in sorted(files)
        )
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=target.parent) as temporary:
                folder = Path(temporary) / "bundle"
                for identity in identities:
                    copied = folder / identity.path
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    _ = shutil.copyfile(self.output / identity.path, copied)
                _ = folder.rename(target)
        if any(file_digest(target / item.path) != item.sha256 for item in identities):
            raise BabiError(reason="Recovery bundle hash mismatch")
        return identities
