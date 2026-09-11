"""Teacher-forced maximum likelihood, immutable evaluation and free generation."""

from math import exp, log
from sys import float_info
from typing import Generic, TypeVar, assert_never

import numpy as np
import torch

from flyrl.ar_config import ARConfig, ARMetrics, TraceEntry
from flyrl.ar_framework import optimizer_step
from flyrl.ar_model import ConnectomeLM
from flyrl.connectome import Graph
from flyrl.gru_model import GRULM
from flyrl.language_data import CorpusError, IntVector, evaluation_starts
from flyrl.transformer_model import TransformerLM

ModelT = TypeVar("ModelT", ConnectomeLM, GRULM, TransformerLM)


class TokenLearner(Generic[ModelT]):
    """AdamW and truncated-window BPTT; no held-out choice or persistent hidden state.

    The private CPU generator chooses training windows. Initialization uses its
    own NumPy generator; evaluation is deterministic and generation has a fresh
    private RNG. No global Python/NumPy/Torch RNG is consumed. Global-norm clipping
    at 1 by default limits exploding temporal gradients without clipping signs or
    silently discarding nonfinite updates. CUDA reductions may be nondeterministic.
    """

    graph: Graph
    config: ARConfig
    model: ModelT
    optimizer: torch.optim.AdamW
    window_rng: torch.Generator
    updates: int
    trace: list[TraceEntry]

    def __init__(self, graph: Graph, config: ARConfig, model: ModelT) -> None:
        """Own one model and a common optimizer/window sampling protocol."""
        self.graph, self.config = graph, config
        self.model = model
        self.optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            foreach=False,
        )
        self.window_rng = torch.Generator().manual_seed(config.seed + 37)
        self.updates, self.trace = 0, []

    def loss(self, windows: torch.Tensor) -> torch.Tensor:
        """Score every strict next-token pair, including the final position."""
        logits = self.model.forward(windows[:, :-1])
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.config.alphabet_size), windows[:, 1:].reshape(-1)
        )

    def train(
        self, tokens: IntVector, updates: int, *, starts: IntVector | None = None
    ) -> None:
        """Train reset windows, optionally using only caller-supplied legal starts."""
        if tokens.size <= self.config.context or updates < 0:
            message = "Training needs context plus target and nonnegative updates"
            raise ValueError(message)
        if starts is not None and (
            starts.ndim != 1
            or starts.dtype != np.int64
            or starts.size == 0
            or bool(
                ((starts < 0) | (starts >= tokens.size - self.config.context)).any()
            )
        ):
            message = "Training starts must be nonempty valid int64 window indices"
            raise ValueError(message)
        offsets = np.arange(self.config.context + 1, dtype=np.int64)
        for _ in range(updates):
            selected = torch.randint(
                tokens.size - self.config.context if starts is None else starts.size,
                (self.config.batch_size,),
                generator=self.window_rng,
            ).numpy()
            if starts is not None:
                selected = starts[selected]
            windows = torch.tensor(
                tokens[selected[:, None] + offsets], device=self.model.weight.device
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss(windows)
            torch.autograd.backward(loss)
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer_step(self.optimizer)
            self.updates += 1
            self.trace.append(
                TraceEntry(
                    update=self.updates,
                    nll=float(loss.detach().item()),
                    gradient_norm=float(norm.item()),
                )
            )

    @torch.no_grad()
    def evaluate(self, tokens: IntVector, *, zero_recurrent: bool = False) -> ARMetrics:
        """Use the baseline's final-target windows; never touch optimizer or RNG."""
        starts = evaluation_starts(
            tokens, self.config.context, self.config.eval_windows
        )
        total_nll, correct = 0.0, 0
        offsets = np.arange(self.config.context, dtype=np.int64)
        for offset in range(0, starts.size, self.config.batch_size):
            selected = starts[offset : offset + self.config.batch_size]
            context = torch.tensor(
                tokens[selected[:, None] + offsets], device=self.model.weight.device
            )
            target = torch.tensor(
                tokens[selected + self.config.context], device=self.model.weight.device
            )
            logits = self.model.forward(context, zero_recurrent=zero_recurrent)[:, -1]
            total_nll += float(
                torch.nn.functional.cross_entropy(
                    logits, target, reduction="sum"
                ).item()
            )
            correct += int((logits.argmax(dim=1) == target).sum().item())
        nll = total_nll / starts.size
        match self.config.tokenization:
            case "character":
                bpc, bpt, perplexity = nll / log(2), None, None
            case "bpe":
                bpc, bpt = None, nll / log(2)
                perplexity = exp(nll) if nll <= log(float_info.max) else None
            case _:
                assert_never(self.config.tokenization)
        return ARMetrics(
            windows=int(starts.size),
            greedy_accuracy=correct / starts.size,
            nll=nll,
            bits_per_character=bpc,
            bits_per_token=bpt,
            perplexity=perplexity,
        )

    @torch.no_grad()
    def generate(
        self, prompt: list[int], length: int, *, greedy: bool = False
    ) -> list[int]:
        """Continue a prompt without resetting state or feeding future targets."""
        if not prompt or length < 0:
            message = "Generation needs a nonempty prompt and nonnegative length"
            raise ValueError(message)
        device = self.model.weight.device
        rng = torch.Generator(device=device).manual_seed(self.config.seed + 91)
        match self.config.generation_context:
            case "windowed":
                result: list[int] = []
                for _ in range(length):
                    inputs = torch.tensor(
                        [(prompt + result)[-self.config.context :]], device=device
                    )
                    logits = self.model.forward(inputs)[:, -1]
                    result.append(int(select_token(logits, rng, greedy).item()))
                return result
            case "stateful":
                return self._stateful_generation(prompt, length, greedy=greedy)
            case _:
                assert_never(self.config.generation_context)

    def _stateful_generation(
        self, prompt: list[int], length: int, *, greedy: bool
    ) -> list[int]:
        """Preserve the original character pilot's unbounded recurrent continuation."""
        match self.model:
            case ConnectomeLM() as model:
                pass
            case GRULM() | TransformerLM():
                raise CorpusError(
                    reason="Stateful generation requires connectome model"
                )
            case _:
                assert_never(self.model)
        device = model.weight.device
        rng = torch.Generator(device=device).manual_seed(self.config.seed + 91)
        state = model.weight.new_zeros((model.nodes, 1))
        logits = model.output_bias[None]
        for character in prompt:
            logits, state = model.step(torch.tensor([character], device=device), state)
        result: list[int] = []
        for _ in range(length):
            token = select_token(logits, rng, greedy)
            result.append(int(token.item()))
            logits, state = model.step(token, state)
        return result


def select_token(
    logits: torch.Tensor, rng: torch.Generator, greedy: bool
) -> torch.Tensor:
    """Choose one token without consuming global framework RNG."""
    return (
        logits.argmax(dim=1)
        if greedy
        else torch.multinomial(logits.softmax(dim=1), 1, generator=rng).flatten()
    )
