"""Resumable fixed-budget runs; reporting never influences gradient updates."""

import json
import sys
from pathlib import Path
from time import perf_counter

import torch
from pydantic import TypeAdapter

from flyrl.ar_benchmark import synchronize
from flyrl.ar_checkpoint import load_checkpoint, save_checkpoint
from flyrl.ar_config import ARConfig, ARMetrics, TraceEntry
from flyrl.ar_corpus import ARCorpus, tokenization, vocabulary
from flyrl.ar_learning import ARLearner
from flyrl.ar_reporting import (
    RunReport,
    baselines,
    continuations,
    counts,
    device_evidence,
    evaluate_splits,
    identities,
)
from flyrl.connectome import Graph
from flyrl.language_baselines import MIN_CONTEXT
from flyrl.language_checkpoint import atomic_text
from flyrl.language_data import CorpusError
from flyrl.language_models import Settings


class RunBudget(Settings):
    """Output and stopping budget, deliberately outside checkpoint compatibility."""

    output: Path
    updates: int
    checkpoint_steps: int
    resume: bool
    graph_file_sha256: str
    corpus_file_sha256: str
    progress: bool = False


def run(
    graph: Graph, corpus: ARCorpus, config: ARConfig, budget: RunBudget
) -> RunReport:
    """Train to the total update target, preserving original initial scores."""
    if config.context < MIN_CONTEXT:
        raise CorpusError(
            reason="CLI comparison requires context >=2 for matched trigram targets"
        )
    if config.tokenization != tokenization(corpus) or config.alphabet_size != len(
        vocabulary(corpus)
    ):
        raise CorpusError(
            reason="Model vocabulary and prediction units differ from corpus"
        )
    if budget.updates < 0 or budget.checkpoint_steps < 1:
        message = "Need nonnegative updates and positive checkpoint steps"
        raise ValueError(message)
    start = perf_counter()
    learner = ARLearner(graph, config)
    device = learner.model.weight.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    path = budget.output / f"seed-{config.seed}" / config.control
    checkpoint, initial_path = path / "checkpoint.npz", path / "initial.json"
    adapter = TypeAdapter(dict[str, ARMetrics])
    if budget.resume:
        load_checkpoint(learner, checkpoint, corpus.fingerprint)
        initial = adapter.validate_json(initial_path.read_text())
    else:
        if checkpoint.exists() or initial_path.exists():
            message = f"Run already exists; use --resume or a new output: {path}"
            raise FileExistsError(message)
        initial = evaluate_splits(learner, corpus)
        atomic_text(initial_path, adapter.dump_json(initial, indent=2).decode())
        save_checkpoint(learner, checkpoint, corpus.fingerprint)
    if learner.updates > budget.updates:
        message = "Requested total updates precede checkpoint progress"
        raise ValueError(message)
    training_seconds = 0.0
    while learner.updates < budget.updates:
        synchronize(device)
        training_start = perf_counter()
        learner.train(
            corpus.train, min(budget.checkpoint_steps, budget.updates - learner.updates)
        )
        synchronize(device)
        training_seconds += perf_counter() - training_start
        save_checkpoint(learner, checkpoint, corpus.fingerprint)
        atomic_text(
            path / "trace.json",
            TypeAdapter(list[TraceEntry]).dump_json(learner.trace).decode(),
        )
        if budget.progress:
            event = {
                "seed": config.seed,
                "control": config.control,
                **learner.trace[-1].model_dump(),
            }
            _ = sys.stderr.write("AR_PROGRESS " + json.dumps(event) + "\n")
            _ = sys.stderr.flush()
    identity = identities(learner, corpus)
    identity.update(
        graph_file_sha256=budget.graph_file_sha256,
        corpus_file_sha256=budget.corpus_file_sha256,
    )
    final = evaluate_splits(learner, corpus)
    ablated = evaluate_splits(learner, corpus, zero_recurrent=True)
    generated = continuations(learner, corpus)
    synchronize(device)
    report = RunReport(
        config=config,
        updates=learner.updates,
        initial=initial,
        final=final,
        zero_recurrent=ablated,
        baselines=baselines(corpus, config),
        counts=counts(learner),
        identities=identity,
        device=device_evidence(learner),
        session_seconds=perf_counter() - start,
        training_seconds_this_session=training_seconds,
        generated=generated.text,
        generated_token_ids=generated.token_ids,
        generated_with_prompt=generated.complete_text,
        trace=tuple(learner.trace),
    )
    atomic_text(path / "report.json", report.model_dump_json(indent=2))
    return report
