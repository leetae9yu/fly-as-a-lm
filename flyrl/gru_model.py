"""Parameter-matched ordinary GRU language model with private CPU initialization."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import override

from flyrl.ar_config import ARConfig
from flyrl.ar_model import requested_device


@runtime_checkable
class _TorchRandom(Protocol):
    """Typed boundary for Torch's incompletely annotated CPU RNG utilities."""

    default_generator: torch.Generator

    def fork_rng(self, devices: list[int]) -> AbstractContextManager[None]: ...


_RANDOM = TypeAdapter(_TorchRandom, config=ConfigDict(arbitrary_types_allowed=True))


@dataclass(frozen=True, slots=True)
class UnsupportedGRUAblationError(ValueError):
    """The anatomical zero-recurrence ablation is undefined for an ordinary GRU."""

    @override
    def __str__(self) -> str:
        return "zero_recurrent is unsupported for an ordinary GRU"


class GRULM(torch.nn.Module):
    """Trainable causal GRU with zero initial hidden state per input window."""

    config: ARConfig
    embedding: torch.nn.Embedding
    recurrent: torch.nn.GRU
    readout: torch.nn.Linear

    def __init__(self, config: ARConfig) -> None:
        """Initialize on CPU without consuming caller CPU or CUDA random streams."""
        super().__init__()
        self.config = config
        device = requested_device(config.device)
        random = _RANDOM.validate_python(torch.random)
        with random.fork_rng(devices=[]):
            _ = random.default_generator.manual_seed(config.seed)
            self.embedding = torch.nn.Embedding(config.alphabet_size, 128, device="cpu")
            self.recurrent = torch.nn.GRU(128, 320, batch_first=True, device="cpu")
            self.readout = torch.nn.Linear(320, config.alphabet_size, device="cpu")
        _ = self.to(device)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the embedding's device without registering a parameter alias."""
        return self.embedding.weight

    @override
    def forward(
        self, tokens: torch.Tensor, *, zero_recurrent: bool = False
    ) -> torch.Tensor:
        """Map [batch, time] tokens to strictly causal [batch, time, vocab] logits."""
        if zero_recurrent:
            raise UnsupportedGRUAblationError
        hidden, _ = self.recurrent.forward(self.embedding.forward(tokens))
        return self.readout.forward(hidden)
