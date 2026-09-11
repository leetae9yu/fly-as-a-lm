"""Parameter-matched causal Transformer with private CPU initialization randomness."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import torch
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import override

from flyrl.ar_config import ARConfig
from flyrl.ar_model import requested_device

_MODEL_WIDTH: Final = 160
_HEADS: Final = 4
_FEEDFORWARD_WIDTH: Final = 640
_BLOCKS: Final = 3


@runtime_checkable
class _RandomModule(Protocol):
    """Typed boundary for Torch's incompletely annotated RNG context manager."""

    def fork_rng(self, *, devices: list[int]) -> AbstractContextManager[None]: ...


_RANDOM: Final = TypeAdapter(
    _RandomModule, config=ConfigDict(arbitrary_types_allowed=True)
).validate_python(torch.random)


@dataclass(frozen=True, slots=True)
class ContextLengthError(ValueError):
    """A token window exceeds the learned positional embedding table."""

    length: int
    context: int

    @override
    def __str__(self) -> str:
        """Describe the requested and supported window lengths."""
        return f"Token length {self.length} exceeds Transformer context {self.context}"


@dataclass(frozen=True, slots=True)
class RecurrentAblationError(ValueError):
    """An anatomical recurrence control was requested for a Transformer."""

    @override
    def __str__(self) -> str:
        """Explain why this model cannot apply the requested control."""
        return "zero_recurrent is unsupported by TransformerLM"


class _CausalBlock(torch.nn.Module):
    """Pre-normalized causal self-attention and GELU feed-forward residuals."""

    attention_norm: torch.nn.LayerNorm
    qkv: torch.nn.Linear
    attention_output: torch.nn.Linear
    feedforward_norm: torch.nn.LayerNorm
    expand: torch.nn.Linear
    contract: torch.nn.Linear

    def __init__(self) -> None:
        """Create one deterministic, dropout-free block on CPU."""
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(_MODEL_WIDTH, device="cpu")
        self.qkv = torch.nn.Linear(_MODEL_WIDTH, 3 * _MODEL_WIDTH, device="cpu")
        self.attention_output = torch.nn.Linear(
            _MODEL_WIDTH, _MODEL_WIDTH, device="cpu"
        )
        self.feedforward_norm = torch.nn.LayerNorm(_MODEL_WIDTH, device="cpu")
        self.expand = torch.nn.Linear(_MODEL_WIDTH, _FEEDFORWARD_WIDTH, device="cpu")
        self.contract = torch.nn.Linear(_FEEDFORWARD_WIDTH, _MODEL_WIDTH, device="cpu")

    @override
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Attend only to the current position and its past, independently per head."""
        batch, length, _ = hidden.shape
        projected = self.qkv.forward(self.attention_norm.forward(hidden))
        query, key, value = (
            projected.reshape(batch, length, 3, _HEADS, _MODEL_WIDTH // _HEADS)
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        attended = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )
        merged = attended.transpose(1, 2).reshape(batch, length, _MODEL_WIDTH)
        residual = hidden + self.attention_output.forward(merged)
        expanded = self.expand.forward(self.feedforward_norm.forward(residual))
        return residual + self.contract.forward(torch.nn.functional.gelu(expanded))


class TransformerLM(torch.nn.Module):
    """Three fixed-width causal blocks; 2,248,096 parameters at V4096/context32.

    Initialization temporarily forks only the CPU RNG and seeds its default
    generator directly, never torch.manual_seed (which also seeds CUDA). All
    layers are created on CPU before transfer. Training uses no stochastic dropout.
    """

    config: ARConfig
    embedding: torch.nn.Embedding
    positions: torch.nn.Embedding
    blocks: tuple[_CausalBlock, ...]
    final_norm: torch.nn.LayerNorm
    output: torch.nn.Linear

    def __init__(self, config: ARConfig) -> None:
        """Build private-seeded CPU parameters, then transfer to the chosen device."""
        super().__init__()
        self.config = config
        device = requested_device(config.device)
        with _RANDOM.fork_rng(devices=[]):
            _ = torch.default_generator.manual_seed(config.seed)
            self.embedding = torch.nn.Embedding(
                config.alphabet_size, _MODEL_WIDTH, device="cpu"
            )
            self.positions = torch.nn.Embedding(
                config.context, _MODEL_WIDTH, device="cpu"
            )
            self.blocks = tuple(_CausalBlock() for _ in range(_BLOCKS))
            for index, block in enumerate(self.blocks):
                self.add_module(f"block_{index}", block)
            self.final_norm = torch.nn.LayerNorm(_MODEL_WIDTH, device="cpu")
            self.output = torch.nn.Linear(
                _MODEL_WIDTH, config.alphabet_size, device="cpu"
            )
        _ = self.to(device)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the embedding's device without registering a parameter alias."""
        return self.embedding.weight

    @override
    def forward(
        self, tokens: torch.Tensor, *, zero_recurrent: bool = False
    ) -> torch.Tensor:
        """Map [batch, time] tokens to causal [batch, time, vocabulary] logits."""
        if zero_recurrent:
            raise RecurrentAblationError
        length = tokens.shape[1]
        if length > self.config.context:
            raise ContextLengthError(length=length, context=self.config.context)
        positions = torch.arange(length, device=tokens.device)
        hidden = self.embedding.forward(tokens) + self.positions.forward(positions)
        for block in self.blocks:
            hidden = block.forward(hidden)
        return self.output.forward(self.final_norm.forward(hidden))
