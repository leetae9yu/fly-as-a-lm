"""Train a sparse anatomical next-character LM: python -m flyrl.autoregressive."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import torch
import typer

from flyrl.ar_benchmark import benchmark
from flyrl.ar_config import ARConfig
from flyrl.ar_experiment import RunBudget, run
from flyrl.ar_learning import ARLearner
from flyrl.ar_reporting import file_identity
from flyrl.connectome import load_graph
from flyrl.language_checkpoint import atomic_text
from flyrl.language_data import load_corpus

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command(name="train")
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """Train requested controls lazily; --updates is a total target.

    Typer inspects the dataclass constructor's fully typed signature. Declarative
    fields keep the independent CLI options together without a giant hand-written
    function signature, untyped kwargs or a lint-rule exemption.
    """

    graph: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    corpus: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    output: Annotated[Path, typer.Option(file_okay=False)]
    device: Annotated[str, typer.Option()] = "cpu"
    updates: Annotated[int, typer.Option(min=0)] = 1000
    batch_size: Annotated[int, typer.Option(min=1)] = 8
    context: Annotated[int, typer.Option(min=1)] = 32
    seeds: Annotated[str, typer.Option()] = "0,1,2"
    controls: Annotated[str, typer.Option()] = "real,shuffled,frozen"
    resume: Annotated[bool, typer.Option()] = False
    progress: Annotated[
        bool, typer.Option(help="Write checkpoint progress as JSON records to stderr")
    ] = False
    learning_rate: Annotated[float, typer.Option(min=0.000001, max=1)] = 0.003
    weight_decay: Annotated[float, typer.Option(min=0)] = 0.0
    gradient_clip: Annotated[float, typer.Option(min=0.000001)] = 1.0
    initial_gain: Annotated[float, typer.Option(min=0.000001, max=1)] = 0.9
    leak: Annotated[float, typer.Option(min=0.000001, max=1)] = 0.5
    eval_windows: Annotated[int, typer.Option(min=1)] = 512
    sample_length: Annotated[int, typer.Option(min=0)] = 120
    checkpoint_steps: Annotated[int, typer.Option(min=1)] = 100
    readout_neurons: Annotated[int, typer.Option(min=1, max=1024)] = 256
    edge_chunk: Annotated[int, typer.Option(min=1)] = 65536
    cpu_threads: Annotated[int, typer.Option(min=1)] = 1
    benchmark_only: Annotated[
        bool, typer.Option(help="Disposable full-context forward/backward/AdamW probe")
    ] = False

    def __post_init__(self) -> None:
        """Execute the validated Typer invocation."""
        execute(self)


def execute(options: Command) -> None:
    """Execute one model at a time; real-only never constructs a shuffled graph."""
    torch.set_num_threads(options.cpu_threads)
    graph, corpus = load_graph(options.graph), load_corpus(options.corpus)
    template = ARConfig(
        alphabet_size=len(corpus.alphabet),
        device=options.device,
        context=options.context,
        batch_size=options.batch_size,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        gradient_clip=options.gradient_clip,
        initial_gain=options.initial_gain,
        leak=options.leak,
        eval_windows=options.eval_windows,
        sample_length=options.sample_length,
        readout_neurons=options.readout_neurons,
        edge_chunk=options.edge_chunk,
    )
    budget = RunBudget(
        output=options.output,
        updates=options.updates,
        checkpoint_steps=options.checkpoint_steps,
        resume=options.resume,
        graph_file_sha256=file_identity(options.graph),
        corpus_file_sha256=file_identity(options.corpus),
        progress=options.progress,
    )
    for seed in options.seeds.split(","):
        for control in options.controls.split(","):
            config = ARConfig.model_validate(
                {**template.model_dump(), "seed": int(seed), "control": control}
            )
            if options.benchmark_only:
                learner = ARLearner(graph, config)
                result = benchmark(learner)
                path = (
                    options.output / f"seed-{config.seed}" / control / "benchmark.json"
                )
                atomic_text(path, result.model_dump_json(indent=2))
                typer.echo(result.model_dump_json())
                del learner
            else:
                report = run(graph, corpus, config, budget)
                typer.echo(report.model_dump_json())


if __name__ == "__main__":
    app()
