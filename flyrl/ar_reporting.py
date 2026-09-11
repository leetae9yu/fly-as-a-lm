"""Exact artifact identity, paired baselines, and auditable language-model reports."""

import hashlib
from math import log
from pathlib import Path
from typing import Final, assert_never

import torch
from pydantic import Field

from flyrl.ar_config import ARConfig, ARMetrics, TraceEntry
from flyrl.ar_corpus import ARCorpus, decode
from flyrl.ar_learning import ExperimentLearner
from flyrl.ar_model import ConnectomeLM
from flyrl.ar_ngrams import fit_ngrams
from flyrl.bpe_data import BPECorpus
from flyrl.gru_model import GRULM
from flyrl.language_baselines import fit_baselines, score_baselines
from flyrl.language_data import Corpus, evaluation_starts
from flyrl.language_models import Settings
from flyrl.language_runtime import graph_fingerprint
from flyrl.transformer_model import TransformerLM

SHUFFLE_DESCRIPTION: Final = (
    "target-stub permutation; degrees preserved; multiedges/self-loops allowed"
)


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
    generated_token_ids: dict[str, tuple[int, ...]] = Field(default_factory=dict)
    generated_with_prompt: dict[str, str] = Field(default_factory=dict)
    selection: str = "fixed final update; no test or validation selection"
    shuffle: str = SHUFFLE_DESCRIPTION
    resume_guarantee: str = "CPU exact tested; CUDA reductions not guaranteed bitwise"


def file_identity(path: Path) -> str:
    """Hash the exact input file with bounded host memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate_splits(
    learner: ExperimentLearner, corpus: ARCorpus, *, zero_recurrent: bool = False
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


def baselines(corpus: ARCorpus, config: ARConfig) -> dict[str, dict[str, ARMetrics]]:
    """Fit only train and score at identical final targets; context must be >=2."""
    match corpus:
        case BPECorpus():
            models = fit_ngrams(corpus.train, len(corpus.vocabulary))
            return {
                name: {
                    label: model.score(
                        tokens,
                        evaluation_starts(tokens, config.context, config.eval_windows)
                        + config.context,
                    )
                    for label, model in zip(
                        ("unigram", "bigram", "trigram"), models, strict=True
                    )
                }
                for name, tokens in (
                    ("train", corpus.train),
                    ("valid", corpus.valid),
                    ("test", corpus.test),
                )
            }
        case Corpus():
            pass
        case _:
            assert_never(corpus)
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


def counts(learner: ExperimentLearner) -> dict[str, int]:
    """Separate anatomical core, small decoder, and frozen versus trained counts."""
    match learner.model:
        case GRULM() as dense:
            return dense_counts(dense, dense.readout)
        case TransformerLM() as dense:
            return dense_counts(dense, dense.output)
        case ConnectomeLM() as model:
            pass
        case _:
            assert_never(learner.model)
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
        "vocabulary_tokens": learner.config.alphabet_size,
    }


def dense_counts(model: GRULM | TransformerLM, head: torch.nn.Linear) -> dict[str, int]:
    """Count ordinary model components without implying anatomical connectivity."""
    total = sum(p.numel() for p in model.parameters())
    embedding = model.weight.numel()
    head_count = sum(p.numel() for p in head.parameters())
    return {
        "trainable_parameters": total,
        "embedding_parameters": embedding,
        "head_parameters": head_count,
        "body_parameters": total - embedding - head_count,
        "vocabulary_tokens": model.config.alphabet_size,
    }


def identities(learner: ExperimentLearner, corpus: ARCorpus) -> dict[str, str]:
    """Identify the source graph and exact control's ordered recurrent edges."""
    match learner.model:
        case ConnectomeLM() as model:
            graph_identity = {
                "graph_fingerprint": graph_fingerprint(learner.graph),
                "graph_provenance": learner.graph.provenance,
                "control_edge_sha256": hashlib.sha256(
                    model.edges.detach().cpu().numpy().tobytes()
                ).hexdigest(),
            }
        case GRULM() | TransformerLM():
            graph_identity = {"graph_usage": "none"}
        case _:
            assert_never(learner.model)
    match corpus:
        case Corpus():
            text_identity = {
                "tokenization": "character",
                "alphabet": "".join(corpus.alphabet),
            }
        case BPECorpus():
            text_identity = {
                "tokenization": "bpe",
                "tokenizer_sha256": hashlib.sha256(
                    corpus.tokenizer_json.encode()
                ).hexdigest(),
            }
        case _:
            assert_never(corpus)
    return {
        **graph_identity,
        "architecture": learner.config.architecture,
        "corpus_fingerprint": corpus.fingerprint,
        "corpus_provenance": corpus.provenance,
        **text_identity,
    }


def device_evidence(learner: ExperimentLearner) -> dict[str, str | int | bool | None]:
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
            if learner.config.is_anatomical
            else "not applicable"
        ),
    }


class Continuations(Settings):
    """Decoded strings and exact IDs, including undecodable byte pieces."""

    text: dict[str, str]
    token_ids: dict[str, tuple[int, ...]]
    complete_text: dict[str, str]


def continuations(learner: ExperimentLearner, corpus: ARCorpus) -> Continuations:
    """Generate freely from a training-only prefix; no test continuation is fed back."""
    prompt = [int(corpus.train.item(i)) for i in range(learner.config.context)]
    ids = {
        "prompt": tuple(prompt),
        **{
            name: tuple(
                learner.generate(prompt, learner.config.sample_length, greedy=greedy)
            )
            for name, greedy in (("sampled", False), ("greedy", True))
        },
    }
    return Continuations(
        text={name: decode(corpus, tokens) for name, tokens in ids.items()},
        token_ids=ids,
        complete_text={
            name: decode(corpus, ids["prompt"] + ids[name])
            for name in ("sampled", "greedy")
        },
    )
