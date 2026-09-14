"""Immutable Task 1 identities and the non-negotiable quality-profile schedule."""

import hashlib
import platform
from importlib.metadata import version
from math import cos, pi
from pathlib import Path
from time import monotonic
from typing import Annotated, Final, Literal, Self, assert_never

import torch
from pydantic import Field, TypeAdapter, model_validator

from flyrl.ar_config import ARConfig
from flyrl.babi_data import TOKENIZER_HASH, VOCABULARY, BabiError, Record
from flyrl.babi_learning import BabiExample, BabiLearner
from flyrl.babi_metrics import evaluate_nll
from flyrl.babi_storage import SHA, Provenance
from flyrl.quality_pilot_schema import ArtifactIdentity
from scripts.connectome_source import file_digest

MILESTONES: Final = (0, 500, 2000, 4000, 8000, 12000)
TARGET: Final = 12000
PANEL_SIZE: Final = 12
PANEL_FACTS: Final = 2
NOVEL_COUNT: Final = 984
WARMUPS: Final = 5
MEMORY_LIMIT: Final = 10 * 1024**3
PROJECTION_LIMIT: Final = 5400
OPTIMIZATION_LIMIT: Final = 6000
RELEASE_LIMIT: Final = 6900
QUALITY_CONFIG: Final = ARConfig(
    alphabet_size=4096,
    tokenization="bpe",
    trainable_codes=True,
    device="cuda",
    context=160,
    batch_size=8,
)


class BabiSchedule(Record):
    """The exact one-based 200-update warmup and 11800-update cosine decay."""

    target_updates: Literal[12000] = 12000
    warmup_updates: Literal[200] = 200
    peak: Annotated[float, Field(ge=0.003, le=0.003)] = 0.003
    minimum: Annotated[float, Field(ge=0.0003, le=0.0003)] = 0.0003

    def rate(self, update: int) -> float:
        """Return the rate of the update about to run, rejecting update zero."""
        if not 1 <= update <= self.target_updates:
            raise BabiError(reason="Learning-rate update outside schedule")
        if update <= self.warmup_updates:
            return self.peak * update / self.warmup_updates
        return self.minimum + 0.00135 * (1 + cos(pi * (update - 200) / 11800))


class CodeFile(Record):
    """One source file's exact bytes, not a mutable Git branch label."""

    path: Path
    sha256: SHA


def code_identity() -> tuple[CodeFile, ...]:
    """Bind the transitive Python implementation and frozen experiment document."""
    modules = (
        "ar_config",
        "ar_framework",
        "ar_model",
        "ar_prepared",
        "ar_sparse",
        "ar_sparse_ops",
        "ar_topology",
        "babi_checkpoint",
        "babi_data",
        "babi_generation",
        "babi_learning",
        "babi_metrics",
        "babi_pilot",
        "babi_pilot_schema",
        "babi_protocol",
        "babi_recovery",
        "babi_recovery_validation",
        "babi_storage",
        "bpe_data",
        "bpe_tokenizer",
        "connectome",
        "cuda_edge_grad",
        "cuda_sparse_mm",
        "language_checkpoint",
        "language_data",
        "language_models",
        "language_runtime",
        "quality_pilot_schema",
    )
    paths = (
        *(Path(f"flyrl/{name}.py") for name in modules),
        Path("scripts/connectome_source.py"),
        Path("scripts/run_babi_task1.py"),
    )
    return tuple(
        CodeFile(path=p, sha256=file_digest(p))
        for p in (*paths, Path("BABI_TASK1_PROTOCOL.md"))
    )


class Runtime(Record):
    """Library, hardware, kernel and numerical execution identity for continuation."""

    python: str
    platform: str
    torch: str
    numpy: str
    tokenizers: str
    cuda: str | None
    cudnn: int | None
    device: str
    device_name: str
    capability: tuple[int, int] | None
    threads: int
    deterministic: bool
    tf32: bool
    cudnn_tf32: bool
    matmul_precision: str


def runtime_identity(device_name: str) -> Runtime:
    """Inspect the requested device without ever substituting CPU for CUDA."""
    device = torch.device(device_name)
    cuda = device.type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise BabiError(reason="CUDA unavailable; CPU fallback forbidden")
    return Runtime(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=str(torch.__version__),
        numpy=version("numpy"),
        tokenizers=version("tokenizers"),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        device=str(device),
        device_name=torch.cuda.get_device_name(device) if cuda else "CPU",
        capability=torch.cuda.get_device_capability(device) if cuda else None,
        threads=torch.get_num_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        tf32=TypeAdapter(bool).validate_python(
            torch.backends.cuda.matmul.allow_tf32, strict=True
        ),
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),
    )


class CorpusIdentity(Record):
    """Separate training and sealed test identities; hashes need not open test."""

    path: Path
    train_valid_sha256: SHA
    test_sha256: SHA
    train_valid_fingerprint: SHA
    tokenizer_sha256: SHA = TOKENIZER_HASH
    counts: tuple[int, int, int] = (8983, 1000, 1000)
    provenance: Provenance = Provenance()


class BabiProtocol(Record):
    """Strict frozen choices; smoke is an explicitly non-benchmark CPU fixture."""

    schema_version: Literal[1] = 1
    profile: Literal["quality", "smoke"] = "quality"
    graph: ArtifactIdentity
    corpus: CorpusIdentity
    config: ARConfig = QUALITY_CONFIG
    runtime: Runtime
    code: Annotated[tuple[CodeFile, ...], Field(min_length=1)]
    schedule: BabiSchedule = BabiSchedule()
    updates: Annotated[int, Field(gt=0)] = 12000
    milestones: tuple[int, ...] = MILESTONES
    checkpoint_interval: Annotated[int, Field(gt=0)] = 250
    recovery_interval: Annotated[int, Field(gt=0)] = 1000

    @property
    def sha256(self) -> str:
        """Hash canonical JSON including every nested default and identity."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    @model_validator(mode="after")
    def fixed(self) -> Self:
        """Reject alternative experiments before any model or artifact mutation."""
        config = self.config
        if (
            config.architecture != "connectome"
            or config.control != "real"
            or config.seed != 0
            or config.port_policy != "legacy_random"
            or config.alphabet_size != VOCABULARY
            or config.tokenization != "bpe"
            or not config.trainable_codes
            or config.generation_context != "stateful"
            or self.corpus.provenance != Provenance()
            or self.corpus.tokenizer_sha256 != TOKENIZER_HASH
            or self.milestones != tuple(sorted(set(self.milestones)))
            or not self.milestones
            or self.milestones[0] != 0
            or self.milestones[-1] != self.updates
            or self.updates > TARGET
            or self.runtime.device != config.device
            or self.runtime.threads != 1
            or self.runtime.tf32
            or self.runtime.cudnn_tf32
        ):
            raise BabiError(
                reason="Protocol model, source, runtime or milestone mismatch"
            )
        match self.profile:
            case "quality":
                if (
                    config != QUALITY_CONFIG
                    or self.updates != TARGET
                    or self.milestones != MILESTONES
                    or (self.checkpoint_interval, self.recovery_interval) != (250, 1000)
                    or (self.graph.nodes, self.graph.edges) != (16384, 1187999)
                    or self.graph.path
                    != Path("data/large_connectome/malecns_v1_n16384.npz")
                    or self.corpus.counts != (8983, 1000, 1000)
                    or "T4" not in self.runtime.device_name
                ):
                    raise BabiError(reason="Quality profile must match frozen Task 1")
            case "smoke":
                if config.device != "cpu":
                    raise BabiError(reason="Smoke requires CPU")
            case _:
                assert_never(self.profile)
        return self

    def next_lr(self, update: int) -> float | None:
        """Explicit continuation rate, absent only at the terminal budget."""
        return self.schedule.rate(update + 1) if update < self.updates else None


@torch.no_grad()
def timing_fixture(learner: BabiLearner, data: tuple[BabiExample, ...]) -> None:
    """Time full answer NLL and eight decoding iterations without early stops."""
    _ = evaluate_nll(learner.model, data)
    model = learner.model
    for length in sorted({len(e.prompt_ids) for e in data}):
        group = tuple(e for e in data if len(e.prompt_ids) == length)
        for start in range(0, len(group), model.config.batch_size):
            batch = group[start : start + model.config.batch_size]
            inputs = torch.tensor(
                [e.prompt_ids for e in batch], device=model.weight.device
            )
            state = model.weight.new_zeros((model.nodes, len(batch)))
            logits = model.output_bias[None]
            for token in inputs.unbind(dim=1):
                logits, state = model.step(token, state)
            for _ in range(8):
                if not bool(torch.isfinite(logits).all()):
                    raise BabiError(reason="Nonfinite timing evaluation")
                logits, state = model.step(logits.argmax(dim=1), state)


def synchronized_time(cuda: bool) -> float:
    """Measure completed GPU work, not asynchronous kernel-launch latency."""
    if cuda:
        torch.cuda.synchronize()
    return monotonic()


class Allocation(Record):
    """Monotonic budget clock surviving restarts, never machine replacement."""

    boot_id: str
    started: float
    recovery: Path

    @property
    def elapsed(self) -> float:
        """Reject replacement allocations rather than resetting the budget clock."""
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != self.boot_id:
            raise BabiError(reason="INCOMPLETE_BUDGET_OR_INTERRUPTION: allocation lost")
        return monotonic() - self.started

    def require_time(self, reserve: float = 0) -> None:
        """Leave enough measured time for completed-evidence recovery and release."""
        if self.elapsed + reserve >= RELEASE_LIMIT:
            raise BabiError(reason="INCOMPLETE_BUDGET_OR_INTERRUPTION: release ceiling")
