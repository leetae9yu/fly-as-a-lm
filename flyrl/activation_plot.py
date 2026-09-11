"""Headless scientific heatmaps of signed model states, without spatial claims."""

from pathlib import Path
from typing import Final

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter

from flyrl.plot_boundary import PlotFigure

TOKENS_PER_PAGE: Final = 32
LABEL_LIMIT: Final = 20
INTERPRETATION: Final = """Activation is not attention or causal importance.
Negative state does not identify an inhibitory neuron.
All neurons are in activations.npz; row order is not a spatial brain map."""


def render_heatmaps(
    states: NDArray[np.float32],
    token_labels: tuple[str, ...],
    row_labels: tuple[str, ...],
    output: Path,
) -> tuple[Path, ...]:
    """Save paginated PNG/SVG charts on a common [-1, 1] color scale.

    Rows retain stable ranking across pages. Labels escape non-ASCII byte-level
    vocabulary symbols explicitly: independently decoded BPE bytes can mislabel
    token identity or produce invalid Unicode. Full labels remain in JSON.
    """
    paths: list[Path] = []
    for start in range(0, len(token_labels), TOKENS_PER_PAGE):
        stop = min(start + TOKENS_PER_PAGE, len(token_labels))
        native = Figure(
            figsize=(
                max(10, (stop - start) * 0.38 + 4),
                max(5, len(row_labels) * 0.2 + 3),
            )
        )
        _ = FigureCanvasAgg(native)
        figure = TypeAdapter(
            PlotFigure, config=ConfigDict(arbitrary_types_allowed=True)
        ).validate_python(native)
        axes = figure.add_subplot()
        image = axes.imshow(
            states[start:stop].T,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        labels: list[str] = []
        for index in range(start, stop):
            label = token_labels[index].encode("unicode_escape").decode("ascii")
            if len(label) > LABEL_LIMIT:
                label = label[: LABEL_LIMIT - 3] + "..."
            labels.append(f"{index + 1}: {label}")
        _ = axes.set_xticks(
            range(stop - start), labels, rotation=60, ha="right", fontsize=8
        )
        _ = axes.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
        _ = axes.set_xlabel("Generated position: raw BPE token label (escaped)")
        _ = axes.set_ylabel("Original neuron ID / input (I), readout (O), other (H)")
        colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
        colorbar.set_label("Signed model state")
        colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        _ = figure.suptitle(
            "Circuit activity before each token is selected",
            x=0.03,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
        heading = f"Positions {start + 1}-{stop} | {len(row_labels)} neurons"
        _ = axes.set_title(
            f"{heading} ranked by mean absolute state",
            loc="left",
            fontsize=10,
            pad=12,
        )
        _ = figure.text(
            0.03,
            0.015,
            INTERPRETATION,
            fontsize=8,
            color="#404040",
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.93))
        for extension in ("png", "svg"):
            path = output / f"activations-{start // TOKENS_PER_PAGE + 1:03}.{extension}"
            figure.savefig(path, dpi=150, facecolor="white")
            paths.append(path)
        figure.clear()
    return tuple(paths)
