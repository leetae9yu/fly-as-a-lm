# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2,<3", "pydantic>=2.10,<3", "tokenizers==0.22.2", "typer>=0.15,<1"
# ]
# ///
# Run from project root: uv run --no-sync python -m scripts.prepare_quality_corpus
"""Prepare a new quality corpus from separate local official TinyStories splits."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import typer

from flyrl.bpe_data import save_bpe_corpus
from flyrl.quality_corpus import (
    QualityPreparation,
    prepare_quality_corpus,
    quality_report,
)

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """Select ordered official splits and fit fresh 4,096-token BPE on train only.

    Each raw file contains stories separated by a line containing <|endoftext|>;
    the final story may end at EOF. The supplied source/revision/license identify
    local files; this command neither fetches nor authenticates upstream data.
    """

    raw_train: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    raw_valid: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    source: Annotated[str, typer.Option()]
    revision: Annotated[str, typer.Option()]
    license_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    output: Annotated[Path, typer.Option(file_okay=False)]
    train_stories: Annotated[int, typer.Option(min=1)] = 50_000
    valid_stories: Annotated[int, typer.Option(min=1)] = 256
    test_stories: Annotated[int, typer.Option(min=1)] = 512
    max_source_bytes: Annotated[int, typer.Option(min=1)] = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        """Preserve existing directories and publish the four reproducible artifacts."""
        if self.output.exists():
            message = f"Preparation output already exists: {self.output}"
            raise FileExistsError(message)
        prepared = prepare_quality_corpus(
            QualityPreparation(
                raw_train=self.raw_train,
                raw_valid=self.raw_valid,
                source=self.source,
                revision=self.revision,
                license_text=self.license_file.read_text(encoding="utf-8"),
                train_stories=self.train_stories,
                valid_stories=self.valid_stories,
                test_stories=self.test_stories,
                max_source_bytes=self.max_source_bytes,
            )
        )
        report = quality_report(prepared)
        self.output.mkdir(parents=True, exist_ok=False)
        save_bpe_corpus(prepared.corpus, self.output / "corpus.npz")
        for name, contents in (
            ("tokenizer.json", prepared.corpus.tokenizer_json),
            ("provenance.json", report.model_dump_json(indent=2)),
            ("SOURCE_LICENSE.txt", prepared.metadata.license_text),
        ):
            _ = (self.output / name).write_text(contents, encoding="utf-8")
        typer.echo(
            " ".join(
                (
                    f"vocabulary={report.actual_vocab_size}",
                    f"train={report.token_counts['train']}",
                    f"valid={report.token_counts['valid']}",
                    f"test={report.token_counts['test']}",
                    f"fingerprint={report.fingerprint}",
                )
            )
        )


if __name__ == "__main__":
    app()
