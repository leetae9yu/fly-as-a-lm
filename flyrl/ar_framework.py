"""Runtime-validated boundaries for incomplete upstream Torch optimizer annotations."""

from typing import Protocol, runtime_checkable

import torch
from pydantic import ConfigDict, TypeAdapter


@runtime_checkable
class _OptimizerStep(Protocol):
    def step(self) -> None: ...


_STEP = TypeAdapter(_OptimizerStep, config=ConfigDict(arbitrary_types_allowed=True))
_STATE = TypeAdapter(
    dict[torch.Tensor, dict[str, torch.Tensor]],
    config=ConfigDict(arbitrary_types_allowed=True),
)


def optimizer_step(optimizer: torch.optim.AdamW) -> None:
    """Validate the framework protocol rather than asserting an unchecked type."""
    _STEP.validate_python(optimizer).step()


def optimizer_tensors(
    optimizer: torch.optim.AdamW,
) -> dict[torch.Tensor, dict[str, torch.Tensor]]:
    """Validate the tensor-only state layout used by our non-AMSGrad AdamW."""
    return _STATE.validate_python(optimizer.state)
