"""Exact artifact identity, paired baselines, and auditable language-model reports."""

import hashlib
from math import log
from pathlib import Path

import torch

from flyrl.ar_config import ARConfig, ARMetrics, TraceEntry
from flyrl.ar_learning import ARLearner
from flyrl.language_baselines import fit_baselines, score_baselines
from flyrl.language_data import Corpus, evaluation_starts
from flyrl.language_models import Settings
from flyrl.language_runtime import graph_fingerprint


class RunReport(Settings):
    """Fixed-final-update scores; test metrics never select models or settings."""

    config: ARConfig
    updates: int
    initial: dict[str, ARMetrics]
    final: dict[str, ARMetrics]
    zero_recurrent: dict[str, ARMetrics]
    baselines: dict[str, dict[str, ARMetrics]]
    counts: dict[str, int]
    identities: dict[str, str]
    device: dict[str, str | int | bool | None]
    session_seconds: float
    training_seconds_this_session: float
    generated: dict[str, str]
    trace: tuple[TraceEntry, ...]
    selection: str = "fixed final update; no test or validation selection"
    shuffle: str = (
        "target-stub permutation; degrees preserved; multiedges/self-loops allowed"
    )
    resume_guarantee: str = (
        "CPU exact tested; CUDA sparse reductions not guaranteed bitwise"
    )


def file_identity(path: Path) -> str:
    """Hash the exact input file with bounded host memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate_splits(
    learner: ARLearner, corpus: Corpus, *, zero_recurrent: bool = False
) -> dict[str, ARMetrics]:
    """Evaluate immutable splits without sharing a hidden state across them."""
    return {
        name: learner.evaluate(tokens, zero_recurrent=zero_recurrent)
        for name, tokens in (
            ("train", corpus.train),
            ("valid", corpus.valid),
            ("test", corpus.test),
        )
    }


def baselines(corpus: Corpus, config: ARConfig) -> dict[str, dict[str, ARMetrics]]:
    """Fit only train and score at identical final targets; context must be >=2."""
    fitted = fit_baselines(corpus)
    result: dict[str, dict[str, ARMetrics]] = {}
    for name, tokens in (
        ("train", corpus.train),
        ("valid", corpus.valid),
        ("test", corpus.test),
    ):
        positions = (
            evaluation_starts(tokens, config.context, config.eval_windows)
            + config.context
        )
        result[name] = {
            score.name: ARMetrics(
                windows=score.trials,
                greedy_accuracy=score.greedy_accuracy,
                nll=score.bits_per_character * log(2),
                bits_per_character=score.bits_per_character,
            )
            for score in score_baselines(fitted, tokens, positions)
        }
    return result


def counts(learner: ARLearner) -> dict[str, int]:
    """Separate anatomical core, small decoder, and frozen versus trained counts."""
    model = learner.model
    return {
        "nodes": model.nodes,
        "edges": model.weight.numel(),
        "recurrent_parameters": model.weight.numel() + model.bias.numel(),
        "head_parameters": model.readout.numel() + model.output_bias.numel(),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "sensory_neurons": model.sensory.numel(),
        "readout_neurons": model.ports.numel(),
        "fixed_code_values": model.codes.numel(),
    }


def identities(learner: ARLearner, corpus: Corpus) -> dict[str, str]:
    """Identify the source graph and exact control's ordered recurrent edges."""
    edges = learner.model.edges.detach().cpu().numpy()
    return {
        "graph_fingerprint": graph_fingerprint(learner.graph),
        "graph_provenance": learner.graph.provenance,
        "control_edge_sha256": hashlib.sha256(edges.tobytes()).hexdigest(),
        "corpus_fingerprint": corpus.fingerprint,
        "corpus_provenance": corpus.provenance,
        "alphabet": "".join(corpus.alphabet),
    }


def device_evidence(learner: ARLearner) -> dict[str, str | int | bool | None]:
    """Capture observed placement, allocated GPU bytes, runtime and numeric flags."""
    device = learner.model.weight.device
    cuda = device.type == "cuda"
    return {
        "requested": learner.config.device,
        "actual": str(device),
        "name": torch.cuda.get_device_name(device) if cuda else "CPU",
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
        "allocated_gpu_bytes": torch.cuda.memory_allocated(device) if cuda else 0,
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "sparse_backend": (
            "custom autograd / native COO sparse-mm / chunked edge gradients"
        ),
    }


def continuations(learner: ARLearner, corpus: Corpus) -> dict[str, str]:
    """Generate freely from a training-only prefix; no test continuation is fed back."""
    prompt = [int(corpus.train.item(i)) for i in range(learner.config.context)]
    return {
        "prompt": "".join(corpus.alphabet[token] for token in prompt),
        **{
            name: "".join(
                corpus.alphabet[token]
                for token in learner.generate(
                    prompt, learner.config.sample_length, greedy=greedy
                )
            )
            for name, greedy in (("sampled", False), ("greedy", True))
        },
    }
