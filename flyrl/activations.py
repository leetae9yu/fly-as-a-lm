"""Token-aligned recordings of the anatomical circuit, not attention weights."""

import hashlib
from pathlib import Path
from typing import Literal, assert_never
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from numpy.lib.format import write_array
from pydantic import Field, TypeAdapter

from flyrl.activation_plot import render_heatmaps
from flyrl.ar_config import ARConfig, parameter_identity_json
from flyrl.ar_engine import select_token
from flyrl.ar_learning import ARLearner
from flyrl.bpe_data import BPECorpus
from flyrl.language_data import CorpusError
from flyrl.language_models import Settings
from flyrl.language_runtime import graph_fingerprint


class ActivationOptions(Settings):
    """Bounded output choices independent of the model's training configuration."""

    length: int = Field(default=32, gt=0)
    greedy: bool = True
    max_neurons: int = Field(default=64, gt=0)


class ActivationMetadata(Settings):
    """Portable token identity and provenance accompanying full signed states."""

    schema_version: Literal[1] = 1
    state_alignment: Literal["before_token_selection"] = "before_token_selection"
    config: ARConfig
    updates: int
    graph_fingerprint: str
    corpus_fingerprint: str
    parameter_fingerprint: str
    prompt: str
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    token_labels: tuple[str, ...]
    generated_text: str
    greedy: bool
    contexts: tuple[tuple[int, ...], ...]
    selected_node_ids: tuple[str, ...]
    sensory_node_ids: tuple[str, ...]
    readout_node_ids: tuple[str, ...]
    selection: Literal["mean_absolute_state"] = "mean_absolute_state"
    neurons_total: int = Field(gt=0)
    arrays: str = "activations.npz"
    figures: tuple[str, ...]


@torch.no_grad()
def export_activations(
    learner: ARLearner,
    corpus: BPECorpus,
    prompt: str,
    output: Path,
    options: ActivationOptions,
) -> Path:
    """Export full states without changing parameters, training RNG or progress.

    Each row is the state used to select its generated token, before feeding that
    token back. Windowed generation replays only the declared context. The export
    contains all neurons; the readable figure selects one stable subset across
    pages. Raw BPE labels are retained instead of decoding individual byte tokens.
    """
    if learner.config.alphabet_size != len(corpus.vocabulary):
        raise CorpusError(reason="Activation corpus and model vocabulary disagree")
    prompt_ids = corpus.encode(prompt)
    if not prompt_ids:
        raise CorpusError(reason="Activation export requires a nonempty prompt")
    model = learner.model
    device = model.weight.device
    rng = torch.Generator(device=device).manual_seed(learner.config.seed + 91)
    state = model.weight.new_zeros((model.nodes, 1))
    states = np.empty((options.length, model.nodes), dtype=np.float32)
    probabilities = np.empty(options.length, dtype=np.float32)
    generated: list[int] = []
    contexts: list[tuple[int, ...]] = []
    for index in range(options.length):
        prefix = prompt_ids + generated
        match learner.config.generation_context:
            case "windowed":
                context = prefix[-learner.config.context :]
                state = torch.zeros_like(state)
                consumed = context
            case "stateful":
                context = prefix
                consumed = prompt_ids if index == 0 else generated[-1:]
            case _:
                assert_never(learner.config.generation_context)
        logits = model.output_bias[None]
        for token in consumed:
            logits, state = model.step(torch.tensor([token], device=device), state)
        chosen = int(select_token(logits, rng, options.greedy).item())
        states[index] = state[:, 0].cpu().numpy()
        probabilities[index] = logits.softmax(dim=1)[0, chosen].item()
        generated.append(chosen)
        contexts.append(tuple(context))
    scores = TypeAdapter(tuple[float, ...]).validate_python(
        np.mean(np.abs(states), axis=0).tolist()
    )
    selected = sorted(range(model.nodes), key=lambda i: -scores[i])[
        : options.max_neurons
    ]
    node_ids = learner.graph.node_ids
    sensory = tuple(node_ids[int(i.item())] for i in model.sensory.unbind())
    readout = tuple(node_ids[int(i.item())] for i in model.ports.unbind())
    labels = tuple(corpus.vocabulary[i] for i in generated)
    selected_ids = tuple(node_ids[i] for i in selected)
    output.mkdir(parents=True, exist_ok=True)
    arrays = (
        ("states", states),
        ("generated_ids", np.asarray(generated, dtype=np.int64)),
        ("probabilities", probabilities),
        ("node_ids", np.asarray(node_ids)),
        ("selected_indices", np.asarray(selected, dtype=np.int64)),
    )
    with ZipFile(output / "activations.npz", "w", compression=ZIP_DEFLATED) as archive:
        for name, array in arrays:
            with archive.open(f"{name}.npy", "w") as stream:
                write_array(stream, array, allow_pickle=False)
    row_labels = tuple(
        f"{node} [{'I' if node in sensory else 'O' if node in readout else 'H'}]"
        for node in selected_ids
    )
    figures = render_heatmaps(states[:, selected], labels, row_labels, output)
    metadata = ActivationMetadata(
        config=learner.config,
        updates=learner.updates,
        graph_fingerprint=graph_fingerprint(learner.graph),
        corpus_fingerprint=corpus.fingerprint,
        parameter_fingerprint=_parameter_fingerprint(learner),
        prompt=prompt,
        prompt_ids=tuple(prompt_ids),
        generated_ids=tuple(generated),
        token_labels=labels,
        generated_text=corpus.decode(prompt_ids + generated),
        greedy=options.greedy,
        contexts=tuple(contexts),
        selected_node_ids=selected_ids,
        sensory_node_ids=sensory,
        readout_node_ids=readout,
        neurons_total=model.nodes,
        figures=tuple(path.name for path in figures),
    )
    path = output / "activations.json"
    _ = path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return path


def _parameter_fingerprint(learner: ARLearner) -> str:
    """Identify the exact learned parameters used for a recording."""
    digest = hashlib.sha256(parameter_identity_json(learner.config).encode())
    for name, parameter in learner.model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
