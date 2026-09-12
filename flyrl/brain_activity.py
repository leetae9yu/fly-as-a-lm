"""CLI for offline token-synchronized anatomical activity playback."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal

import typer

from flyrl.brain_activity_data import (
    BrainPlayback,
    RenderOptions,
    activity_emphasis,
    align_soma_positions,
    load_playback,
)
from flyrl.brain_activity_plot import render_playback

__all__ = [
    "BrainPlayback",
    "RenderOptions",
    "activity_emphasis",
    "align_soma_positions",
    "render_playback",
]

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """Render an existing activation export without retraining."""

    activation_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)]
    annotations: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    tokenizer: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    output: Annotated[Path, typer.Option(dir_okay=False)]
    frames_per_second: Annotated[int, typer.Option(min=1, max=12)] = 3
    dpi: Annotated[int, typer.Option(min=50, max=200)] = 150
    view: Annotated[Literal["projections", "rotating_3d"], typer.Option()] = (
        "projections"
    )

    def __post_init__(self) -> None:
        """Execute the validated offline renderer."""
        playback = load_playback(self.activation_dir, self.annotations, self.tokenizer)
        rendered = render_playback(
            playback,
            RenderOptions(
                output=self.output,
                frames_per_second=self.frames_per_second,
                dpi=self.dpi,
                view=self.view,
            ),
        )
        typer.echo(rendered)


if __name__ == "__main__":
    app()
