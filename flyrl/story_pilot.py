"""Local TinyStories next-token pilot: python -m flyrl.story_pilot."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, assert_never

import torch
import typer

from flyrl.activations import ActivationOptions, export_activations
from flyrl.ar_checkpoint import load_checkpoint, save_checkpoint
from flyrl.ar_config import ARConfig, PortPolicy, TraceEntry
from flyrl.ar_learning import ARLearner
from flyrl.connectome import load_graph
from flyrl.language_checkpoint import atomic_text
from flyrl.language_data import CorpusError
from flyrl.language_models import Settings
from flyrl.language_runtime import graph_fingerprint
from flyrl.story_data import load_story_corpus
from flyrl.story_metrics import HeldoutMetrics, evaluate_heldout
from scripts.anatomy_port_artifacts import ConfiguredPorts, load_condition_ports
from scripts.connectome_source import file_digest

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
DEFAULT_GRAPH: Final = Path("data/large_connectome/malecns_v1_n16384.npz")


class PilotReport(Settings):
    """Portable pilot result with all-position metric denominators and resume trace."""

    format: Literal["flyrl-story-pilot-v1"] = "flyrl-story-pilot-v1"
    config: ARConfig
    updates: int
    graph_fingerprint: str
    corpus_fingerprint: str
    graph_nodes: int
    graph_edges: int
    training_windows: int
    trainable_parameters: int
    initial: HeldoutMetrics
    final: HeldoutMetrics
    prompt: str
    generated_token_ids: dict[str, tuple[int, ...]]
    generated_with_prompt: dict[str, str]
    activations: str | None
    trace: tuple[TraceEntry, ...]


@app.command()
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """Train one sparse anatomical condition; --updates is a total target.

    The intended pilot defaults to the existing real 16384-neuron graph. Tiny CPU
    checks must explicitly override --graph and --updates. Training samples only
    within-story windows; short training stories supply no full context windows.
    """

    corpus: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    output: Annotated[Path, typer.Option(file_okay=False)]
    graph: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_GRAPH
    control: Annotated[Literal["real", "shuffled"], typer.Option()] = "real"
    updates: Annotated[int, typer.Option(min=0)] = 1000
    checkpoint_steps: Annotated[int, typer.Option(min=1)] = 100
    resume: Annotated[bool, typer.Option()] = False
    device: Annotated[str, typer.Option()] = "cpu"
    seed: Annotated[int, typer.Option(min=0, max=2**32 - 1)] = 0
    context: Annotated[int, typer.Option(min=1)] = 64
    batch_size: Annotated[int, typer.Option(min=1)] = 8
    learning_rate: Annotated[float, typer.Option(min=0.000001, max=1)] = 0.003
    weight_decay: Annotated[float, typer.Option(min=0)] = 0.0
    readout_neurons: Annotated[int, typer.Option(min=1, max=1024)] = 256
    edge_chunk: Annotated[int, typer.Option(min=1)] = 65536
    trainable_codes: Annotated[bool, typer.Option()] = True
    sample_length: Annotated[int, typer.Option(min=0)] = 64
    prompt: Annotated[str, typer.Option()] = "Once upon a time"
    cpu_threads: Annotated[int, typer.Option(min=1, max=4)] = 1
    port_policy: Annotated[PortPolicy, typer.Option()] = "legacy_random"
    port_manifest: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = (
        None
    )

    def __post_init__(self) -> None:
        """Execute the validated explicit command."""
        typer.echo(run(self).model_dump_json())


def resolve_condition_ports(options: Command) -> ConfiguredPorts | None:
    """Resolve and verify one optional explicit port manifest at the CLI boundary."""
    match options.port_policy:
        case "legacy_random":
            if options.port_manifest is not None:
                message = "Legacy random policy cannot use a port manifest"
                raise ValueError(message)
            return None
        case "alpn_mbon" | "alpn_random" | "random_mbon" | "random_random":
            if options.port_manifest is None:
                message = "Explicit port policies require --port-manifest"
                raise ValueError(message)
            configured = load_condition_ports(
                options.port_manifest,
                options.port_policy,
                options.seed,
            )
            if file_digest(options.graph) != configured.graph_sha256:
                message = "Port manifest targets a different graph artifact"
                raise ValueError(message)
            return configured
        case _:
            assert_never(options.port_policy)


def run(options: Command) -> PilotReport:
    """Reuse ARLearner updates, its sole sampling RNG and atomic NPZ checkpoints."""
    torch.set_num_threads(options.cpu_threads)
    stories = load_story_corpus(options.corpus)
    corpus = stories.corpus
    graph = load_graph(options.graph)
    ports = resolve_condition_ports(options)
    config = ARConfig(
        alphabet_size=len(corpus.vocabulary),
        tokenization="bpe",
        seed=options.seed,
        device=options.device,
        control=options.control,
        context=options.context,
        batch_size=options.batch_size,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        readout_neurons=options.readout_neurons,
        edge_chunk=options.edge_chunk,
        trainable_codes=options.trainable_codes,
        sample_length=options.sample_length,
        port_policy=options.port_policy,
        port_manifest_sha256=None if ports is None else ports.manifest_sha256,
        sensory_indices=None if ports is None else ports.sensory_indices,
        readout_indices=None if ports is None else ports.readout_indices,
    )
    starts = stories.starts(stories.metadata.train, config.context)
    prompt_ids = corpus.encode(options.prompt)
    if not prompt_ids:
        raise CorpusError(reason="Generation prompt must encode at least one token")
    checkpoint = options.output / "checkpoint.npz"
    initial_path = options.output / "initial.json"
    if not options.resume and options.output.exists():
        message = (
            f"Run output already exists; use --resume or a new output: {options.output}"
        )
        raise FileExistsError(message)
    learner = ARLearner(graph, config)
    if options.resume:
        load_checkpoint(learner, checkpoint, corpus.fingerprint)
        initial = HeldoutMetrics.model_validate_json(initial_path.read_text())
    else:
        initial = evaluate_heldout(learner, stories)
        options.output.mkdir(parents=True, exist_ok=False)
        atomic_text(initial_path, initial.model_dump_json(indent=2))
        save_checkpoint(learner, checkpoint, corpus.fingerprint)
    if learner.updates > options.updates:
        raise CorpusError(reason="Requested total updates precede checkpoint progress")
    while learner.updates < options.updates:
        learner.train(
            corpus.train,
            min(
                options.checkpoint_steps,
                options.updates - learner.updates,
            ),
            starts=starts,
        )
        save_checkpoint(learner, checkpoint, corpus.fingerprint)
    final = evaluate_heldout(learner, stories)
    generated = {
        name: tuple(learner.generate(prompt_ids, options.sample_length, greedy=greedy))
        for name, greedy in (("greedy", True), ("sampled", False))
    }
    activation = None
    if options.sample_length:
        activation = (
            export_activations(
                learner,
                corpus,
                options.prompt,
                options.output,
                ActivationOptions(
                    length=options.sample_length, greedy=True, max_neurons=64
                ),
            )
            .relative_to(options.output)
            .as_posix()
        )
    report = PilotReport(
        config=config,
        updates=learner.updates,
        graph_fingerprint=graph_fingerprint(graph),
        corpus_fingerprint=corpus.fingerprint,
        graph_nodes=len(graph.node_ids),
        graph_edges=int(graph.source.size),
        training_windows=int(starts.size),
        trainable_parameters=sum(
            p.numel() for p in learner.model.parameters() if p.requires_grad
        ),
        initial=initial,
        final=final,
        prompt=options.prompt,
        generated_token_ids=generated,
        generated_with_prompt={
            name: corpus.decode(prompt_ids + list(ids))
            for name, ids in generated.items()
        },
        activations=activation,
        trace=tuple(learner.trace),
    )
    atomic_text(options.output / "report.json", report.model_dump_json(indent=2))
    return report


if __name__ == "__main__":
    app()
