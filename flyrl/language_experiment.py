"""Paired character-learning controls and fixed-final-step reporting."""

from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Annotated, Literal

from pydantic import Field

from flyrl.connectome import Graph, shuffled_graph
from flyrl.language_checkpoint import atomic_text, load_checkpoint, save_checkpoint
from flyrl.language_data import Corpus
from flyrl.language_learning import CharacterLearner
from flyrl.language_models import (
    DeviceEvidence,
    LanguageConfig,
    LanguageError,
    Metrics,
    Settings,
)
from flyrl.language_runtime import (
    device_evidence,
    graph_fingerprint,
    tensor_fingerprint,
)


class Experiment(Settings):
    """Paired-seed budget, independent of evaluation performance."""

    output: Path
    learner: LanguageConfig
    updates: Annotated[int, Field(ge=0)] = 100
    seeds: tuple[Annotated[int, Field(ge=0, le=2**32 - 1)], ...] = (0, 1, 2)
    checkpoint_every: Annotated[int, Field(ge=1)] = 100
    resume: bool = False


class SplitMetrics(Settings):
    """The same deterministic, non-overlapping window rule for each split."""

    train: Metrics
    valid: Metrics
    test: Metrics


class RunRecord(Settings):
    """One independently reproducible topology/plasticity condition."""

    seed: int
    control: Literal["real", "shuffled", "frozen"]
    config: LanguageConfig
    updates: int
    resumed_updates: int
    training_seconds_this_invocation: float
    training_rewards: tuple[float, ...]
    training_reward_mean: float
    initial: SplitMetrics
    final: SplitMetrics
    graph_fingerprint: str
    initial_weights: str
    final_weights: str
    weight_change_l2: float
    sensory: tuple[int, ...]
    output: tuple[int, ...]
    prompt: str
    continuation: str
    checkpoint: str
    device: DeviceEvidence


class Summary(Settings):
    """Machine-readable evidence, with no forced scientific success conclusion."""

    schema_version: Literal[1] = 1
    corpus_fingerprint: str
    corpus_provenance: str
    graph_provenance: str
    nodes: int
    edges: int
    alphabet: tuple[str, ...]
    runs: tuple[RunRecord, ...]
    selection: str = (
        "prespecified final update; no validation/test checkpoint selection"
    )
    evaluation_protocol: str = (
        "language_data.evaluation_starts; non-overlapping windows across each split; "
        "one seeded hidden trajectory per window; greedy uses conditional argmax; "
        "expected_reward integrates the categorical action; sampled_reward samples it"
    )
    nll_caveat: str = (
        "NLL/bits per character are mean conditional hidden-path NLL, an upper bound "
        "in expectation on marginal NLL; not exact marginal language likelihood"
    )
    learning_rule: str = (
        "trajectory Bernoulli score + final sampled categorical score; terminal "
        "sampled correctness only; preceding-batch EMA baseline; no autograd/BPTT; "
        "no learned biases, encoder or decoder; source-normalized initial weights"
    )
    topology_caveat: str = (
        "fixed edge-independent random node assignments, identical across controls; "
        "dense masked NxN float32 matmul on sparse anatomy; arbitrary roles; "
        "degree-preserving shuffle is a finite proposal process, not uniform sampling"
    )
    interpretation: str = (
        "Improvement may only reflect character-frequency learning. Compare held-out "
        "unigram and n-gram baselines before claiming contextual or linguistic ability."
    )


def split_metrics(learner: CharacterLearner, corpus: Corpus) -> SplitMetrics:
    """Report independent split metrics without any model selection."""
    return SplitMetrics(
        train=learner.evaluate(corpus.train),
        valid=learner.evaluate(corpus.valid),
        test=learner.evaluate(corpus.test),
    )


def run_experiment(graph: Graph, corpus: Corpus, experiment: Experiment) -> Summary:
    """Train all paired controls, checkpoint boundaries, then evaluate final weights."""
    if not experiment.seeds or len(set(experiment.seeds)) != len(experiment.seeds):
        raise LanguageError(reason="seeds must be nonempty and distinct")
    if experiment.learner.alphabet_size != len(corpus.alphabet):
        raise LanguageError(reason="configured alphabet size differs from corpus")
    records: list[RunRecord] = []
    for seed in experiment.seeds:
        shuffled = shuffled_graph(graph, seed)
        conditions: tuple[
            tuple[Literal["real", "shuffled", "frozen"], Graph, bool], ...
        ] = (
            ("real", graph, False),
            ("shuffled", shuffled, False),
            ("frozen", graph, True),
        )
        for control, topology, frozen in conditions:
            config = experiment.learner.model_copy(
                update={"seed": seed, "frozen": frozen}
            )
            learner = CharacterLearner(topology, config)
            path = experiment.output / f"{control}-{seed}.npz"
            if path.exists() and not experiment.resume:
                raise LanguageError(reason=f"checkpoint exists; use --resume: {path}")
            initial = split_metrics(learner, corpus)
            original = learner.weights.clone()
            if path.exists():
                load_checkpoint(learner, path, corpus.fingerprint)
            resumed = learner.updates
            if resumed > experiment.updates:
                raise LanguageError(reason="update target precedes checkpoint progress")
            started = perf_counter()
            while learner.updates < experiment.updates:
                learner.train(
                    corpus.train,
                    min(
                        experiment.checkpoint_every,
                        experiment.updates - learner.updates,
                    ),
                )
                save_checkpoint(learner, path, corpus.fingerprint)
            # Each learn() obtains a host reward, synchronizing CUDA; no async timing.
            elapsed = perf_counter() - started
            save_checkpoint(learner, path, corpus.fingerprint)
            prompt = tuple(corpus.valid.item(i) for i in range(config.context))
            generated = learner.generate(prompt, config.sample_length)
            records.append(
                RunRecord(
                    seed=seed,
                    control=control,
                    config=config,
                    updates=learner.updates,
                    resumed_updates=resumed,
                    training_seconds_this_invocation=elapsed,
                    training_rewards=tuple(learner.rewards),
                    training_reward_mean=mean(learner.rewards)
                    if learner.rewards
                    else 0.0,
                    initial=initial,
                    final=split_metrics(learner, corpus),
                    graph_fingerprint=graph_fingerprint(topology),
                    initial_weights=tensor_fingerprint(original),
                    final_weights=tensor_fingerprint(learner.weights),
                    weight_change_l2=float(
                        (learner.weights - original).square().sum().sqrt().item()
                    ),
                    sensory=learner.ports.sensory,
                    output=learner.ports.output,
                    prompt="".join(corpus.alphabet[i] for i in prompt),
                    continuation="".join(corpus.alphabet[i] for i in generated),
                    checkpoint=path.name,
                    device=device_evidence(learner.weights, config.device),
                )
            )
            atomic_text(
                experiment.output / f"{control}-{seed}.json",
                records[-1].model_dump_json(indent=2),
            )
    summary = Summary(
        corpus_fingerprint=corpus.fingerprint,
        corpus_provenance=corpus.provenance,
        graph_provenance=graph.provenance,
        nodes=len(graph.node_ids),
        edges=graph.source.size,
        alphabet=corpus.alphabet,
        runs=tuple(records),
    )
    atomic_text(experiment.output / "summary.json", summary.model_dump_json(indent=2))
    return summary
