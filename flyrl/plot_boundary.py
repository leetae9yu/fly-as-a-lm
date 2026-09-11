"""Narrow typed boundary for matplotlib's dynamic keyword-driven plotting API."""

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from matplotlib.axis import Tick
from matplotlib.image import AxesImage
from matplotlib.text import Text
from numpy.typing import NDArray


class PlotAxes(Protocol):
    """The plotting operations used by an activation heatmap."""

    def imshow(self, data: NDArray[np.float32], /, **kwargs: str | float) -> AxesImage:
        """Display the signed state matrix."""
        ...

    def set_xticks(
        self, ticks: range, labels: list[str], **kwargs: str | float
    ) -> list[Tick]:
        """Label generated token positions."""
        ...

    def set_yticks(
        self, ticks: range, labels: tuple[str, ...], **kwargs: str | float
    ) -> list[Tick]:
        """Label neuron identities."""
        ...

    def set_xlabel(self, xlabel: str) -> Text:
        """Label the token axis."""
        ...

    def set_ylabel(self, ylabel: str) -> Text:
        """Label the neuron axis."""
        ...

    def set_title(self, label: str, **kwargs: str | float) -> Text:
        """Describe the page and subset."""
        ...


class PlotColorbar(Protocol):
    """State scale annotations."""

    def set_label(self, label: str) -> None:
        """Name the measured quantity."""
        ...

    def set_ticks(self, ticks: list[float]) -> None:
        """Keep a shared scale across pages."""
        ...


@runtime_checkable
class PlotFigure(Protocol):
    """Actual matplotlib Figure methods with the narrow types this caller uses."""

    def add_subplot(self) -> PlotAxes:
        """Create one plot."""
        ...

    def colorbar(
        self, mappable: AxesImage, *, ax: PlotAxes, fraction: float, pad: float
    ) -> PlotColorbar:
        """Attach its color scale."""
        ...

    def suptitle(self, t: str, **kwargs: str | float) -> Text:
        """Add the figure heading."""
        ...

    def text(self, x: float, y: float, s: str, **kwargs: str | float) -> Text:
        """Add the interpretation note."""
        ...

    def tight_layout(self, *, rect: tuple[float, float, float, float]) -> None:
        """Lay out labels within reserved margins."""
        ...

    def savefig(self, fname: Path, *, dpi: int, facecolor: str) -> None:
        """Render one actual vector or raster artifact."""
        ...

    def clear(self) -> None:
        """Release the figure's artists."""
        ...
