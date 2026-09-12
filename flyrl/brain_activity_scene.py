"""Matplotlib scene construction for anatomical activity playback."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from matplotlib.figure import Figure
from matplotlib.text import Text
from pydantic import ConfigDict, TypeAdapter

from flyrl.brain_activity_data import FloatArray, RenderView


@runtime_checkable
class MutablePoints(Protocol):
    """Dynamic point styling needed by the animation callback."""

    def set_facecolors(self, colors: FloatArray) -> None:
        """Replace every point's RGBA color."""
        ...

    def set_sizes(self, sizes: FloatArray) -> None:
        """Replace every point's marker size."""
        ...


@runtime_checkable
class BrainAxes(Protocol):
    """The small dynamic Matplotlib axes surface used here."""

    def set_facecolor(self, color: str) -> None:
        """Set the axes background."""
        ...

    def scatter(
        self,
        x: FloatArray,
        y: FloatArray,
        /,
        **kwargs: str | float | FloatArray,
    ) -> MutablePoints:
        """Draw spatial points."""
        ...

    def set_title(self, title: str, **kwargs: str | float) -> Text:
        """Label the spatial view."""
        ...

    def set_aspect(self, aspect: str, *, adjustable: str) -> None:
        """Keep spatial axes equally scaled."""
        ...

    def set_axis_off(self) -> None:
        """Hide nonsemantic axes decoration."""
        ...

    def set_position(self, position: tuple[float, float, float, float]) -> None:
        """Place the axes in figure coordinates."""
        ...


@runtime_checkable
class BrainAxes3D(Protocol):
    """The dynamic 3D axes operations used by the rotating scene."""

    def set_facecolor(self, color: str) -> None:
        """Set the axes background."""
        ...

    def scatter(
        self,
        x: FloatArray,
        y: FloatArray,
        z: FloatArray,
        /,
        **kwargs: str | float | FloatArray,
    ) -> MutablePoints:
        """Draw three-dimensional spatial points."""
        ...

    def set_title(self, title: str, **kwargs: str | float) -> Text:
        """Label the spatial view."""
        ...

    def set_axis_off(self) -> None:
        """Hide nonsemantic axes decoration."""
        ...

    def set_box_aspect(
        self, aspect: tuple[float, float, float], *, zoom: float
    ) -> None:
        """Set the 3D box proportions."""
        ...

    def set_position(self, position: tuple[float, float, float, float]) -> None:
        """Place the axes in figure coordinates."""
        ...

    def view_init(self, *, elev: float, azim: float) -> None:
        """Move the camera for one frame."""
        ...


@dataclass(frozen=True, slots=True)
class Scene:
    """Mutable plot collections plus a frame-driven camera."""

    points: tuple[MutablePoints, ...]
    rotate: Callable[[int], None]


@dataclass(frozen=True, slots=True)
class CameraPlan:
    """A 3D camera schedule and view-specific framing."""

    azimuths: tuple[float, ...]
    zoom: float


def prepare_scene(
    native: Figure,
    coordinates: FloatArray,
    frames: int,
    view: RenderView,
) -> Scene:
    """Build the requested spatial scene with a stable frame contract."""
    builders: dict[RenderView, Callable[[], Scene]] = {
        "activity_3d": lambda: _prepare_three_dimensional_scene(
            native, coordinates, view_camera("activity_3d", frames)
        ),
        "projections": lambda: _prepare_projection_scene(native, coordinates),
        "rotating_3d": lambda: _prepare_three_dimensional_scene(
            native, coordinates, view_camera("rotating_3d", frames)
        ),
    }
    return builders[view]()


def view_camera(view: RenderView, frames: int) -> CameraPlan:
    """Return the camera plan consumed by a spatial view."""
    plans: dict[RenderView, CameraPlan] = {
        "activity_3d": CameraPlan(tuple(-35.0 for _ in range(frames)), 1.60),
        "projections": CameraPlan(tuple(0.0 for _ in range(frames)), 1.0),
        "rotating_3d": CameraPlan(
            tuple(
                -65.0 + 120.0 * frame / max(frames - 1, 1) for frame in range(frames)
            ),
            1.30,
        ),
    }
    return plans[view]


def view_azimuths(view: RenderView, frames: int) -> tuple[float, ...]:
    """Return the camera schedule consumed by a spatial view."""
    return view_camera(view, frames).azimuths


def _prepare_projection_scene(native: Figure, coordinates: FloatArray) -> Scene:
    """Create stable X-Y and X-Z soma projections."""
    top = TypeAdapter(
        BrainAxes, config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(native.add_subplot(121))
    side = TypeAdapter(
        BrainAxes, config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(native.add_subplot(122))
    top.set_position((0.04, 0.15, 0.43, 0.61))
    side.set_position((0.53, 0.15, 0.43, 0.61))
    collections: list[MutablePoints] = []
    foreground = np.zeros((len(coordinates), 4), dtype=np.float64)
    for axes, first, second, title in (
        (top, 0, 1, "Spatial projection X-Y"),
        (side, 0, 2, "Spatial projection X-Z"),
    ):
        axes.set_facecolor("#07090f")
        _ = axes.scatter(
            coordinates[:, first],
            coordinates[:, second],
            s=2,
            color="#4d5668",
            alpha=0.25,
            linewidths=0,
        )
        points = axes.scatter(
            coordinates[:, first],
            coordinates[:, second],
            s=2,
            c=foreground,
            linewidths=0,
        )
        _ = axes.set_title(title, color="#c6cfdf", fontsize=11, pad=8)
        axes.set_aspect("equal", adjustable="box")
        axes.set_axis_off()
        collections.append(points)
    return Scene((collections[0], collections[1]), lambda _frame: None)


def _prepare_three_dimensional_scene(
    native: Figure, coordinates: FloatArray, camera: CameraPlan
) -> Scene:
    """Create a rotating 3D soma cloud without inventing missing positions."""
    axes = TypeAdapter(
        BrainAxes3D, config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(native.add_subplot(111, projection="3d"))
    axes.set_facecolor("#07090f")
    _ = axes.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        s=3,
        color="#aab4c8",
        alpha=0.65,
        linewidths=0,
    )
    points = axes.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        s=2,
        c=np.zeros((len(coordinates), 4), dtype=np.float64),
        linewidths=0,
        depthshade=False,
    )
    axes.set_position((0.05, 0.12, 0.90, 0.66))
    axes.set_box_aspect((1.55, 1.25, 0.85), zoom=camera.zoom)
    axes.set_axis_off()

    def rotate(frame: int) -> None:
        axes.view_init(elev=18.0, azim=camera.azimuths[frame])

    return Scene((points,), rotate)
