"""Run character reward experiments with ``python -m flyrl.language``."""

from pathlib import Path
from typing import Annotated, Final

import typer

from flyrl.connectome import load_graph
from flyrl.language_data import load_corpus
from flyrl.language_experiment import Experiment, run_experiment
from flyrl.language_models import LanguageConfig

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(  # noqa: PLR0913 - Independent user-facing Typer options, not internal parameters.
    *,
    graph: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    corpus: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    device: Annotated[str, typer.Option()] = "cpu",
    updates: Annotated[int, typer.Option(min=0)] = 100,
    batch_size: Annotated[int, typer.Option(min=1)] = 64,
    context: Annotated[int, typer.Option(min=1)] = 16,
    seeds: Annotated[str, typer.Option()] = "0,1,2",
    resume: Annotated[bool, typer.Option()] = False,
    eval_windows: Annotated[int, typer.Option(min=1)] = 512,
    sample_length: Annotated[int, typer.Option(min=0)] = 120,
    learning_rate: Annotated[float, typer.Option(min=0.000001, max=1)] = 0.1,
    checkpoint_every: Annotated[int, typer.Option(min=1)] = 100,
) -> None:
    """Train next-character decisions using only terminal sampled correctness."""
    data = load_corpus(corpus)
    config = LanguageConfig(
        alphabet_size=len(data.alphabet),
        device=device,
        context=context,
        batch_size=batch_size,
        eval_windows=eval_windows,
        sample_length=sample_length,
        learning_rate=learning_rate,
    )
    try:
        parsed_seeds = tuple(int(seed) for seed in seeds.split(","))
    except ValueError as error:
        message = "seeds must be comma-separated integers"
        raise typer.BadParameter(message) from error
    _ = run_experiment(
        load_graph(graph),
        data,
        Experiment(
            output=output,
            learner=config,
            updates=updates,
            seeds=parsed_seeds,
            checkpoint_every=checkpoint_every,
            resume=resume,
        ),
    )
    typer.echo(str(output / "summary.json"))


if __name__ == "__main__":
    app()
