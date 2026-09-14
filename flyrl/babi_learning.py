"""Storage-independent answer-only BPTT over the unchanged ConnectomeLM."""

from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Final, Self, TypeAlias

import torch
from pydantic import Field, model_validator

from flyrl.ar_config import ARConfig, TraceEntry
from flyrl.ar_framework import optimizer_step, optimizer_tensors
from flyrl.ar_model import ConnectomeLM
from flyrl.connectome import Graph
from flyrl.language_data import CorpusError
from flyrl.language_models import Settings

VOCABULARY: Final = 4096
TERMINATOR: Final = 199
IGNORE_TARGET: Final = -100
TokenId: TypeAlias = Annotated[int, Field(strict=True, ge=0, lt=VOCABULARY)]
TokenTuple: TypeAlias = Annotated[tuple[TokenId, ...], Field(min_length=1)]
ExampleId: TypeAlias = Annotated[str, Field(strict=True, min_length=1)]


class BabiExample(Settings):
    """One stable question identity with separately encoded prompt and answer."""

    example_id: ExampleId
    prompt_ids: TokenTuple
    answer_ids: Annotated[tuple[TokenId, ...], Field(min_length=2)]

    @model_validator(mode="after")
    def final_terminator(self) -> Self:
        """Require exactly one LF at the end of the nonempty answer segment."""
        if self.answer_ids[-1] != TERMINATOR or TERMINATOR in self.answer_ids[:-1]:
            raise CorpusError(reason="Answer must end with exactly one terminator")
        return self


@dataclass(frozen=True, slots=True)
class BabiBatch:
    """Immutable shifted rows; prompt and padding targets use the CE ignore index."""

    example_ids: tuple[str, ...]
    inputs: tuple[tuple[int, ...], ...]
    targets: tuple[tuple[int, ...], ...]
    answer_counts: tuple[int, ...]


def make_batch(examples: tuple[BabiExample, ...], context: int) -> BabiBatch:
    """Right-pad to context without truncating or dropping a final short batch."""
    if not examples or context < 1:
        raise CorpusError(reason="Batch requires examples and positive context")
    inputs: list[tuple[int, ...]] = []
    targets: list[tuple[int, ...]] = []
    for example in examples:
        sequence = example.prompt_ids + example.answer_ids
        padding = context - (len(sequence) - 1)
        if padding < 0:
            raise CorpusError(reason=f"Sequence overflow: {example.example_id}")
        inputs.append(sequence[:-1] + (TERMINATOR,) * padding)
        targets.append(
            (IGNORE_TARGET,) * (len(example.prompt_ids) - 1)
            + example.answer_ids
            + (IGNORE_TARGET,) * padding
        )
    return BabiBatch(
        tuple(example.example_id for example in examples),
        tuple(inputs),
        tuple(targets),
        tuple(len(example.answer_ids) for example in examples),
    )


def answer_token_losses(logits: torch.Tensor, batch: BabiBatch) -> torch.Tensor:
    """Return unreduced masked CE, with exactly zero loss at ignored positions."""
    targets = torch.tensor(batch.targets, device=logits.device)
    if logits.shape != (*targets.shape, VOCABULARY):
        raise CorpusError(reason="Answer logits must have batch/context/4096 shape")
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCABULARY), targets.flatten(), reduction="none"
    ).reshape(targets.shape)
    if not bool(torch.isfinite(losses).all()):
        raise CorpusError(reason="Nonfinite answer NLL")
    return losses


def answer_loss(logits: torch.Tensor, batch: BabiBatch) -> torch.Tensor:
    """Average tokens per question, then questions; never detach prompt recurrence."""
    losses = answer_token_losses(logits, batch)
    counts = losses.new_tensor(batch.answer_counts)
    return (losses.sum(dim=1) / counts).mean()


class BabiUpdate(TraceEntry):
    """One completed update's pre-step objective, unclipped norm and sampled IDs."""

    sampled_ids: tuple[str, ...]
    learning_rate: float


class BabiLearner:
    """Intentionally mutable owner of parameters, Adam, private RNG and progress.

    The runner owns the frozen budget/LR schedule and checkpoint transactions.
    ``update(examples, learning_rate=...)`` permits both scheduled main updates
    and constant-rate disposable diagnostics. A rejected nonfinite post-step state
    invalidates this learner: recover from the last complete checkpoint, not retry.
    """

    graph: Graph
    config: ARConfig
    model: ConnectomeLM
    optimizer: torch.optim.AdamW
    window_rng: torch.Generator
    updates: int
    trace: list[BabiUpdate]
    exposure_counts: Counter[str]

    def __init__(self, graph: Graph, config: ARConfig) -> None:
        """Initialize only the anatomical BPE core and existing AdamW configuration."""
        if (
            config.alphabet_size != VOCABULARY
            or not config.is_anatomical
            or config.control != "real"
            or not config.trainable_codes
        ):
            raise CorpusError(reason="bAbI requires a real 4096-token trainable core")
        self.graph, self.config = graph, config
        self.model = ConnectomeLM(graph, config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            foreach=False,
        )
        self.window_rng = torch.Generator(device="cpu").manual_seed(config.seed + 37)
        self.updates, self.trace = 0, []
        self.exposure_counts = Counter()

    def loss(self, batch: BabiBatch) -> torch.Tensor:
        """Run full-prompt recurrence before computing the answer-only objective."""
        inputs = torch.tensor(batch.inputs, device=self.model.weight.device)
        return answer_loss(self.model.forward(inputs), batch)

    def _require_finite_state(self) -> None:
        """Reject corrupt model or Adam tensors before accepting an update boundary."""
        tensors = [parameter.detach() for parameter in self.model.parameters()]
        tensors.extend(
            tensor
            for state in optimizer_tensors(self.optimizer).values()
            for tensor in state.values()
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise CorpusError(reason="Nonfinite model or optimizer update state")

    def update(
        self,
        examples: tuple[BabiExample, ...],
        *,
        learning_rate: float | None = None,
    ) -> BabiUpdate:
        """Sample uniformly with replacement; clip, step, then record exact exposure."""
        if not examples or len({e.example_id for e in examples}) != len(examples):
            raise CorpusError(reason="Training requires nonempty unique example IDs")
        if any(
            len(e.prompt_ids) + len(e.answer_ids) - 1 > self.config.context
            for e in examples
        ):
            raise CorpusError(reason="Training sequence overflow")
        rate = self.config.learning_rate if learning_rate is None else learning_rate
        if not 0 < rate <= 1:
            raise CorpusError(reason="Learning rate must be finite and in (0, 1]")
        self._require_finite_state()
        indices = torch.randint(
            len(examples), (self.config.batch_size,), generator=self.window_rng
        )
        selected = tuple(examples[int(index)] for index in indices)
        batch = make_batch(selected, self.config.context)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(batch)
        torch.autograd.backward(loss)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True
        )
        optimizer_step(self.optimizer)
        self._require_finite_state()
        entry = BabiUpdate(
            update=self.updates + 1,
            nll=float(loss.detach().item()),
            gradient_norm=float(norm.item()),
            sampled_ids=batch.example_ids,
            learning_rate=rate,
        )
        self.updates += 1
        self.trace.append(entry)
        self.exposure_counts.update(batch.example_ids)
        return entry
