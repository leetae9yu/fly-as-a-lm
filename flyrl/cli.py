"""Command-line surface for a prepared graph and reproducible control suite."""

from pathlib import Path
from typing import Annotated, Final

import typer

from flyrl.connectome import load_graph
from flyrl.experiment import Experiment, run_experiment

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
DEFAULT_OUTPUT: Final = Path("results")


@app.command()
def main(
    *,
    graph: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)] = DEFAULT_OUTPUT,
    episodes: Annotated[int, typer.Option(min=0)] = 2000,
    seeds: Annotated[
        str, typer.Option(help="Comma-separated nonnegative seeds")
    ] = "0,1,2",
    delay: Annotated[int, typer.Option(min=1)] = 3,
    eval_trials: Annotated[int, typer.Option(min=2)] = 512,
    learning_rate: Annotated[float, typer.Option(min=0.000001, max=1)] = 0.03,
    checkpoint_every: Annotated[int, typer.Option(min=1)] = 250,
    resume: Annotated[bool, typer.Option(help="Continue to --episodes total")] = False,
) -> None:
    """Run both binary tasks with real, shuffled, and frozen controls.

    Evaluation uses separate fixed randomness and never trains the network.
    Checkpoints require the same graph, Python/NumPy versions, and settings.
    """
    try:
        parsed_seeds = tuple(int(seed) for seed in seeds.split(","))
    except ValueError as error:
        message = "seeds must be comma-separated integers"
        raise typer.BadParameter(message) from error
    summary = run_experiment(
        load_graph(graph),
        Experiment(
            output=output,
            episodes=episodes,
            seeds=parsed_seeds,
            delay=delay,
            eval_trials=eval_trials,
            learning_rate=learning_rate,
            checkpoint_every=checkpoint_every,
            resume=resume,
        ),
    )
    typer.echo(summary.model_dump_json(indent=2))
