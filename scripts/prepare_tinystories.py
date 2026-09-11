# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2,<3", "pydantic>=2.10,<3", "tokenizers==0.22.2", "typer>=0.15,<1"
# ]
# ///
# Run from project root: uv run --no-sync python -m scripts.prepare_tinystories
"""Explicit local-only TinyStories preparation with retained licensing and hashes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import typer

from flyrl.bpe_data import save_bpe_corpus
from flyrl.story_data import prepare_stories
from flyrl.story_source import Preparation

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """Pool --raw files, deduplicate, then split a bounded unique-story selection.

    A delimiter is a line containing only <|endoftext|>. The final story may end
    at EOF. Source/revision/license describe caller-supplied files; no upstream
    authenticity is claimed and no network request is made.
    """

    raw: Annotated[list[Path], typer.Option(exists=True, dir_okay=False)]
    source: Annotated[str, typer.Option()]
    revision: Annotated[str, typer.Option()]
    license_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    output: Annotated[Path, typer.Option(file_okay=False)]
    train_stories: Annotated[int, typer.Option(min=1)] = 1000
    valid_stories: Annotated[int, typer.Option(min=1)] = 16
    test_stories: Annotated[int, typer.Option(min=1)] = 16
    vocab_size: Annotated[int, typer.Option(min=257, max=65536)] = 4096
    seed: Annotated[int, typer.Option(min=0, max=2**32 - 1)] = 0
    max_source_bytes: Annotated[int, typer.Option(min=1)] = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        """Publish a new directory; preserve all existing artifacts."""
        if self.output.exists():
            message = f"Preparation output already exists: {self.output}"
            raise FileExistsError(message)
        prepared = prepare_stories(
            Preparation(
                raw_files=tuple(self.raw),
                source=self.source,
                revision=self.revision,
                license_text=self.license_file.read_text(encoding="utf-8"),
                train_stories=self.train_stories,
                valid_stories=self.valid_stories,
                test_stories=self.test_stories,
                vocab_size=self.vocab_size,
                seed=self.seed,
                max_source_bytes=self.max_source_bytes,
            )
        )
        self.output.mkdir(parents=True, exist_ok=False)
        save_bpe_corpus(prepared.corpus, self.output / "corpus.npz")
        _ = (self.output / "tokenizer.json").write_text(
            prepared.corpus.tokenizer_json,
            encoding="utf-8",
        )
        _ = (self.output / "provenance.json").write_text(
            prepared.metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
        _ = (self.output / "SOURCE_LICENSE.txt").write_text(
            prepared.metadata.license_text,
            encoding="utf-8",
        )
        typer.echo(prepared.corpus.fingerprint)


if __name__ == "__main__":
    app()
