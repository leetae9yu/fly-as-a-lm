# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6,<3", "pydantic>=2.10,<3", "typer>=0.15,<1"]
# ///
# Run: uv run --no-sync python -m scripts.run_bpe_study --job 0
"""Execute one predeclared BPE comparison job without changing its budget."""

import json
from pathlib import Path
from typing import Annotated

import torch
import typer
from pydantic import Field

from flyrl.ar_config import ARConfig
from flyrl.ar_corpus import load_ar_corpus
from flyrl.ar_experiment import RunBudget, run
from flyrl.ar_reporting import file_identity
from flyrl.connectome import load_graph
from flyrl.language_data import CorpusError
from flyrl.language_models import Settings


class Study(Settings):
    """Frozen budgets, data identity and paired model configurations."""

    updates: Annotated[int, Field(ge=1)]
    seeds: tuple[Annotated[int, Field(ge=0, le=2**32 - 1)], ...]
    checkpoint_steps: Annotated[int, Field(ge=1)]
    graph: Path
    corpus: Path
    output: Path
    graph_sha256: str
    corpus_sha256: str
    rationale: str
    configurations: tuple[ARConfig, ...]

    def jobs(self) -> tuple[ARConfig, ...]:
        """Run the full seed-zero comparison before repeating every condition."""
        return tuple(
            ARConfig.model_validate({**config.model_dump(), "seed": seed})
            for seed in self.seeds
            for config in self.configurations
        )


def main(job: Annotated[int, typer.Option(min=0, max=14)] = 0) -> None:
    """Run one fixed-budget job or resume its existing checkpoint."""
    plan = Study.model_validate_json(Path("BPE_STUDY.json").read_bytes())
    for path, expected in (
        (plan.graph, plan.graph_sha256),
        (plan.corpus, plan.corpus_sha256),
    ):
        if file_identity(path) != expected:
            raise CorpusError(reason=f"Study source identity mismatch: {path}")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.sparse.check_sparse_tensor_invariants.enable()
    config = plan.jobs()[job]
    path = plan.output / f"seed-{config.seed}" / config.condition
    budget = RunBudget(
        output=plan.output,
        updates=plan.updates,
        checkpoint_steps=plan.checkpoint_steps,
        resume=(path / "checkpoint.npz").exists(),
        graph_file_sha256=plan.graph_sha256,
        corpus_file_sha256=plan.corpus_sha256,
        progress=True,
    )
    report = run(load_graph(plan.graph), load_ar_corpus(plan.corpus), config, budget)
    typer.echo(
        "BPE_JOB_DONE "
        + json.dumps(
            {
                "job": job,
                "seed": config.seed,
                "condition": config.condition,
                "updates": report.updates,
                "checkpoint_sha256": file_identity(path / "checkpoint.npz"),
            }
        )
    )


if __name__ == "__main__":
    typer.run(main)
