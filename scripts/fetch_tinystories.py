# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2,<3", "pydantic>=2.10,<3", "typer>=0.15,<1",
#   "httpx2[http2,brotli,zstd]>=2.0.0,<3"
# ]
# ///
# Run from project root: uv run --no-sync python -m scripts.fetch_tinystories
"""Fetch bounded original TinyStories prefixes; no model or training is constructed."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal

import typer
from pydantic import TypeAdapter

from flyrl.story_fetch import PrefixBudget, fetch_prefix

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """Download first N complete stories at the embedded immutable source revision."""

    output: Annotated[Path, typer.Option(file_okay=False)]
    split: Annotated[str, typer.Option(help="Original source file: train or valid")] = (
        "train"
    )
    stories: Annotated[int, typer.Option(min=1)] = 1032
    max_bytes: Annotated[int, typer.Option(min=1)] = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        """Perform the explicitly requested bounded acquisition."""
        split = TypeAdapter[Literal["train", "valid"]](
            Literal["train", "valid"]
        ).validate_python(self.split)
        report = fetch_prefix(
            self.output,
            split,
            PrefixBudget(
                stories=self.stories,
                max_bytes=self.max_bytes,
            ),
        )
        typer.echo(report.model_dump_json())


if __name__ == "__main__":
    app()
