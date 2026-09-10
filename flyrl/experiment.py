"""Paired-seed topology/plasticity controls and portable JSON summaries."""

from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Annotated

from pydantic import Field

from flyrl.checkpoint import atomic_write, load_checkpoint, save_checkpoint
from flyrl.connectome import Graph, shuffled_graph
from flyrl.learning import ExperimentError, Learner, select_ports
from flyrl.models import Config, Control, RunResult, Settings, Task


class Experiment(Settings):
    """Run budget and shared settings for both tasks and all three controls."""

    episodes: Annotated[int, Field(ge=0)] = 2000
    seeds: tuple[Annotated[int, Field(ge=0)], ...] = (0, 1, 2)
    delay: Annotated[int, Field(ge=1)] = 3
    eval_trials: Annotated[int, Field(ge=2)] = 512
    learning_rate: Annotated[float, Field(gt=0, le=1)] = 0.03
    checkpoint_every: Annotated[int, Field(ge=1)] = 250
    output: Path
    resume: bool = False


class Aggregate(Settings):
    """Seed-level means and sample standard deviations, not significance tests."""

    task: Task
    control: Control
    seeds: int
    initial_mean: float
    final_mean: float
    final_sd: float
    improvement_mean: float


class Summary(Settings):
    """Experiment report with paired individual runs and aggregate statistics."""

    schema_version: int = 1
    provenance: str
    nodes: int
    edges: int
    runs: tuple[RunResult, ...]
    aggregates: tuple[Aggregate, ...]
    topology_caveat: str = (
        "Ports are selected on an observed real edge. Shuffling may delete that "
        "edge and reduce task accessibility. This comparison is confounded by "
        "port selection, not evidence of a uniquely biological learning advantage."
    )
    weight_initialization: str = "signed source-normalized outgoing budget"
    evaluation_protocol: str = "balanced cues; fixed independent stochastic RNG"


def run_single(initial: Learner, experiment: Experiment) -> RunResult:
    """Train or resume one condition and measure it against initialization."""
    config, ports = initial.config, initial.ports
    checkpoint = (
        experiment.output / f"{config.task}-{config.control}-{config.seed}.json"
    )
    if checkpoint.exists() and not experiment.resume:
        raise ExperimentError(reason=f"checkpoint exists; use --resume: {checkpoint}")
    learner = load_checkpoint(checkpoint, initial) if checkpoint.exists() else initial
    if len(learner.rewards) > experiment.episodes:
        raise ExperimentError(reason="episode target precedes checkpoint progress")
    before = initial.evaluate()
    initial_weights = initial.weights.copy()
    while len(learner.rewards) < experiment.episodes:
        learner.train(
            min(experiment.checkpoint_every, experiment.episodes - len(learner.rewards))
        )
        save_checkpoint(learner, checkpoint)
    save_checkpoint(learner, checkpoint)
    difference = learner.weights - initial_weights
    direct = any(
        source == ports.sensory and target == ports.output
        for source, target in zip(
            learner.graph.source, learner.graph.target, strict=True
        )
    )
    return RunResult(
        config=config,
        ports=ports,
        episodes=len(learner.rewards),
        initial=before,
        final=learner.evaluate(),
        training_accuracy=mean(learner.rewards) if learner.rewards else 0.0,
        weight_change_l2=sqrt(float((difference * difference).sum())),
        checkpoint=checkpoint.name,
        direct_edge_present=direct,
    )


def run_experiment(graph: Graph, experiment: Experiment) -> Summary:
    """Execute paired controls; resume requires exact graph/config compatibility."""
    if not experiment.seeds or len(set(experiment.seeds)) != len(experiment.seeds):
        raise ExperimentError(reason="provide a nonempty list of distinct seeds")
    if experiment.eval_trials % 2:
        raise ExperimentError(reason="evaluation trials must be even")
    results: list[RunResult] = []
    for seed in experiment.seeds:
        ports = select_ports(graph, seed)
        shuffled = shuffled_graph(graph, seed)
        for task in Task:
            for control in Control:
                config = Config(
                    seed=seed,
                    task=task,
                    control=control,
                    delay=experiment.delay,
                    eval_trials=experiment.eval_trials,
                    learning_rate=experiment.learning_rate,
                )
                topology = shuffled if control == Control.SHUFFLED else graph
                initial = Learner(topology, config, ports)
                results.append(run_single(initial, experiment))
    aggregates: list[Aggregate] = []
    for task in Task:
        for control in Control:
            group = [
                r
                for r in results
                if r.config.task == task and r.config.control == control
            ]
            accuracy = [r.final.accuracy for r in group]
            aggregates.append(
                Aggregate(
                    task=task,
                    control=control,
                    seeds=len(group),
                    initial_mean=mean(r.initial.accuracy for r in group),
                    final_mean=mean(accuracy),
                    final_sd=stdev(accuracy) if len(accuracy) > 1 else 0.0,
                    improvement_mean=mean(
                        r.final.accuracy - r.initial.accuracy for r in group
                    ),
                )
            )
    summary = Summary(
        provenance=graph.provenance,
        nodes=len(graph.node_ids),
        edges=len(graph.source),
        runs=tuple(results),
        aggregates=tuple(aggregates),
    )
    atomic_write(experiment.output / "summary.json", summary.model_dump_json(indent=2))
    return summary
