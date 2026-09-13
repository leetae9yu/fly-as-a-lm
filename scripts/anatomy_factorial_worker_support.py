"""Metric-free progress and typed failures for the remote factorial worker."""

import sys
from dataclasses import dataclass

from typing_extensions import override


@dataclass(frozen=True, slots=True)
class FactorialEvidenceError(RuntimeError):
    """A remote condition violated the prospective factorial protocol."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


def emit_factorial_progress(message: str) -> None:
    """Write one metric-free progress record immediately."""
    _ = sys.stdout.write(message + "\n")
    _ = sys.stdout.flush()
