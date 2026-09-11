# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6,<3", "numpy>=2.1,<3", "pydantic>=2.10,<3", "typer>=0.20"]
# ///
# python -m scripts.verify_ar_resume --help
"""Measure restoration and resumed CUDA updates without changing the published run."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import torch
import typer

from flyrl.ar_checkpoint import load_checkpoint, save_checkpoint
from flyrl.ar_corpus import load_ar_corpus
from flyrl.ar_framework import optimizer_tensors
from flyrl.ar_learning import ExperimentLearner, make_learner
from flyrl.ar_reporting import RunReport, continuations, file_identity
from flyrl.connectome import load_graph
from flyrl.language_checkpoint import atomic_text
from flyrl.language_models import Settings


class Difference(Settings):
    """Maximum absolute differences, not an assumed CUDA determinism guarantee."""

    parameters: float
    adam: float
    rng_equal: bool
    updates_equal: bool
    trace_equal: bool


class ResumeEvidence(Settings):
    """A read-only source checkpoint and a separately continued verification copy."""

    source_sha256: str
    source_updates: int
    continued_updates: int
    stored_generation_reproduced: bool
    restoration: Difference
    continuation: Difference
    continuation_loss_difference: float


def difference(first: ExperimentLearner, second: ExperimentLearner) -> Difference:
    """Compare corresponding parameters, moments, window RNG, progress and trace."""
    parameters = tuple(
        zip(first.model.parameters(), second.model.parameters(), strict=True)
    )
    first_adam = optimizer_tensors(first.optimizer)
    second_adam = optimizer_tensors(second.optimizer)
    adam_pairs = [
        (value, second_adam[right][key])
        for left, right in parameters
        if left in first_adam
        for key, value in first_adam[left].items()
    ]
    return Difference(
        parameters=max(float((a - b).abs().max().item()) for a, b in parameters),
        adam=max(float((a - b).abs().max().item()) for a, b in adam_pairs),
        rng_equal=torch.equal(
            first.window_rng.get_state(), second.window_rng.get_state()
        ),
        updates_equal=first.updates == second.updates,
        trace_equal=first.trace == second.trace,
    )


@dataclass
class Command:
    """Continue verification copies under the report's exact execution settings."""

    graph: Annotated[Path, typer.Option(exists=True)]
    corpus: Annotated[Path, typer.Option(exists=True)]
    run: Annotated[Path, typer.Option(exists=True)]
    output: Annotated[Path, typer.Option()]

    def __post_init__(self) -> None:
        """Verify exact restoration, then measure two additional continuation steps."""
        report = RunReport.model_validate_json((self.run / "report.json").read_bytes())
        torch.set_num_threads(1)
        graph, corpus = load_graph(self.graph), load_ar_corpus(self.corpus)
        source = self.run / "checkpoint.npz"
        source_hash = file_identity(source)
        first = make_learner(graph, report.config)
        load_checkpoint(first, source, corpus.fingerprint)
        if first.updates != report.updates:
            message = "Published report and checkpoint have different update counts"
            raise ValueError(message)
        generated = continuations(first, corpus)
        first.train(corpus.train, 2)
        midpoint = self.output / "midpoint.npz"
        save_checkpoint(first, midpoint, corpus.fingerprint)
        second = make_learner(graph, report.config)
        load_checkpoint(second, midpoint, corpus.fingerprint)
        restored = difference(first, second)
        if (
            restored.parameters != 0
            or restored.adam != 0
            or not restored.rng_equal
            or not restored.updates_equal
            or not restored.trace_equal
        ):
            message = "Checkpoint restoration did not preserve all saved state"
            raise RuntimeError(message)
        first.train(corpus.train, 2)
        second.train(corpus.train, 2)
        continued = difference(first, second)
        if not continued.rng_equal or not continued.updates_equal:
            message = "Resumed training consumed different windows or update counts"
            raise RuntimeError(message)
        evidence = ResumeEvidence(
            source_sha256=source_hash,
            source_updates=report.updates,
            continued_updates=second.updates,
            stored_generation_reproduced=(
                generated.text == report.generated
                and (
                    not report.generated_token_ids
                    or generated.token_ids == report.generated_token_ids
                )
            ),
            restoration=restored,
            continuation=continued,
            continuation_loss_difference=abs(
                first.trace[-1].nll - second.trace[-1].nll
            ),
        )
        if file_identity(source) != source_hash:
            message = "Verification modified the published checkpoint"
            raise RuntimeError(message)
        atomic_text(
            self.output / "verification.json", evidence.model_dump_json(indent=2)
        )
        typer.echo(evidence.model_dump_json())


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
_ = app.command()(Command)

if __name__ == "__main__":
    app()
