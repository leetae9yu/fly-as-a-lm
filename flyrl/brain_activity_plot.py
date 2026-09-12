"""Matplotlib renderer for token-synchronized anatomical activity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import matplotlib as mpl
import numpy as np
from matplotlib.animation import (
    AbstractMovieWriter,
    FFMpegWriter,
    FuncAnimation,
    PillowWriter,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pydantic import ConfigDict, TypeAdapter

from flyrl.brain_activity_data import (
    BrainPlayback,
    FloatArray,
    RenderOptions,
    RenderView,
    activity_changes,
    activity_emphasis,
)
from flyrl.brain_activity_scene import prepare_scene

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from matplotlib.text import Text

FONT_FAMILY: Final = ("DejaVu Sans",)
STORY_CHARACTER_LIMIT: Final = 112
INTERPRETATIONS: Final[dict[RenderView, str]] = {
    "activity_3d": (
        "Fixed camera · color/size = state change from prior frame "
        "(frame 1: zero baseline)"
    ),
    "projections": (
        "Fixed projections · color/size = state change from prior frame "
        "(frame 1: zero baseline)"
    ),
    "rotating_3d": (
        "Rotating anatomy view · color/size = state change from prior frame "
        "(frame 1: zero baseline)"
    ),
}
CAUTION: Final = "Model state, not biological firing, attention, or causal importance."


@runtime_checkable
class BrainFigure(Protocol):
    """The small dynamic Matplotlib figure surface used here."""

    def text(
        self,
        x: float,
        y: float,
        text: str,
        /,
        **kwargs: str | float,
    ) -> Text:
        """Place text in figure coordinates."""
        ...


def render_playback(playback: BrainPlayback, options: RenderOptions) -> Path:
    """Render fixed spatial projections with one frame per generated token."""
    mpl.rcParams["font.family"] = list(FONT_FAMILY)
    mpl.rcParams["axes.unicode_minus"] = False
    native = Figure(figsize=(12.8, 7.2), facecolor="#07090f")
    _ = FigureCanvasAgg(native)
    figure = TypeAdapter(
        BrainFigure, config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(native)
    scene = prepare_scene(
        native,
        playback.coordinates,
        len(playback.token_labels),
        options.view,
    )
    points = scene.points
    changes = activity_changes(playback.states)
    emphasis = activity_emphasis(playback.states)
    token_text = _figure_text(figure, 0.5, 0.84, 23, "white")
    story_text = _figure_text(figure, 0.5, 0.075, 12, "#d9deea")
    metric_text = _figure_text(figure, 0.5, 0.795, 10, "#8f9bb3")
    _ = figure.text(
        0.04,
        0.965,
        "Where a fruit-fly central-brain model changes for each token",
        color="white",
        fontsize=20,
    )
    _ = figure.text(
        0.04,
        0.925,
        INTERPRETATIONS[options.view],
        color="#aab4c8",
        fontsize=11,
    )
    _ = figure.text(
        0.5,
        0.018,
        CAUTION,
        ha="center",
        color="#9aa6bc",
        fontsize=11,
    )
    _ = figure.text(0.64, 0.89, "blue = state fell", color="#2e99ff", fontsize=11)
    _ = figure.text(0.84, 0.89, "red = state rose", color="#ff3d57", fontsize=11)

    def update(frame: int) -> None:
        values: FloatArray = changes[frame : frame + 1].reshape(-1).astype(np.float64)
        strength: FloatArray = emphasis[frame : frame + 1].reshape(-1)
        colors = _state_colors(values, strength)
        sizes: FloatArray = 2.0 + 44.0 * strength**1.45
        scene.rotate(frame)
        for collection in points:
            collection.set_facecolors(colors)
            collection.set_sizes(sizes)
        visible_token = (
            playback.token_labels[frame]
            .replace("Ġ", "[space]")
            .replace("Ċ", "[newline]")
        )
        visible_token = {'"': "[double quote]", "'": "[apostrophe]"}.get(
            visible_token, visible_token
        )
        token_text.set_text(f"BPE token selected   {visible_token!r}")
        metric_text.set_text(
            "".join(
                (
                    f"{frame + 1} / {len(playback.token_labels)}   ·   ",
                    f"probability {playback.probabilities[frame]:.1%}   ·   ",
                    f"{len(playback.coordinates):,} positioned neurons",
                )
            )
        )
        story_text.set_text(_recent_text(playback.prefix_texts[frame]))

    animation = FuncAnimation(
        native,
        update,
        frames=range(len(playback.token_labels)),
        interval=1000 / options.frames_per_second,
        blit=False,
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    writer_factories: dict[str, Callable[[], AbstractMovieWriter]] = {
        ".gif": lambda: PillowWriter(fps=options.frames_per_second),
        ".mp4": lambda: FFMpegWriter(
            fps=options.frames_per_second,
            codec="mpeg4",
            extra_args=["-pix_fmt", "yuv420p", "-qscale:v", "3"],
        ),
    }
    writer = writer_factories[options.output.suffix.lower()]()
    animation.save(options.output, writer=writer, dpi=options.dpi)
    native.clear()
    return options.output


def _figure_text(
    figure: BrainFigure, x: float, y: float, size: int, color: str
) -> Text:
    """Create centered figure text with stable styling."""
    return figure.text(x, y, "", ha="center", color=color, fontsize=size)


def _state_colors(values: FloatArray, strength: FloatArray) -> FloatArray:
    """Map signed state changes to a dark-center blue/red ramp."""
    neutral = np.asarray([0.12, 0.14, 0.19], dtype=np.float64)
    negative = np.asarray([0.18, 0.60, 1.00], dtype=np.float64)
    positive = np.asarray([1.00, 0.24, 0.34], dtype=np.float64)
    colors = np.empty((len(values), 4), dtype=np.float64)
    colors[:, :3] = neutral
    negative_rows = values < 0
    positive_rows = ~negative_rows
    magnitude = np.minimum(np.abs(values), 1.0)
    colors[negative_rows, :3] += magnitude[negative_rows, None] * (negative - neutral)
    colors[positive_rows, :3] += magnitude[positive_rows, None] * (positive - neutral)
    colors[:, 3] = 0.03 + 0.97 * strength
    return colors


def _recent_text(text: str) -> str:
    """Show a readable tail without cutting the first visible word."""
    cleaned = text.replace("\n", " [newline] ")
    if len(cleaned) <= STORY_CHARACTER_LIMIT:
        return cleaned
    tail = cleaned[-STORY_CHARACTER_LIMIT:]
    boundary = tail.find(" ")
    return "..." + tail[boundary + 1 :] if boundary >= 0 else "..." + tail
