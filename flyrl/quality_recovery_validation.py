"""Strict generation and full-state activation evidence boundaries for recovery."""

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from pydantic import BaseModel, TypeAdapter

from flyrl.activation_plot import TOKENS_PER_PAGE
from flyrl.activations import ActivationMetadata
from flyrl.ar_learning import ARLearner
from flyrl.quality_generation import generate_quality_panel
from flyrl.quality_pilot_schema import (
    CheckpointResult,
    QualityProgress,
    QualityProtocol,
)
from flyrl.story_data import StoryCorpus
from scripts.connectome_source import file_digest
from scripts.image_artifact_validation import validate_figure

if TYPE_CHECKING:
    from numpy import generic
    from numpy.typing import NDArray

Model = TypeVar("Model", bound=BaseModel)


def require(condition: bool, message: str) -> None:
    """Fail closed at an untrusted artifact boundary."""
    if not condition:
        raise ValueError(message)


def read_model(model: type[Model], path: Path) -> Model:
    """Parse every nested JSON setting without coercion."""
    return model.model_validate_json(path.read_bytes(), strict=True)


def figures(length: int) -> tuple[str, ...]:
    """Enumerate the existing exporter's complete, paginated figure surface."""
    return tuple(
        f"activations-{page + 1:03}.{ext}"
        for page in range((length + TOKENS_PER_PAGE - 1) // TOKENS_PER_PAGE)
        for ext in ("png", "svg")
    )


def activation(
    path: Path,
    result: CheckpointResult,
    inputs: tuple[ARLearner, StoryCorpus],
    fingerprint: str,
    protocol: QualityProtocol,
) -> None:
    """Bind signed states, ranked neurons, tokens, contexts and decoded figures."""
    learner, stories = inputs
    metadata = read_model(ActivationMetadata, path)
    record = next(
        (
            r
            for r in result.generation
            if r.mode == "greedy" and r.prompt == metadata.prompt
        ),
        None,
    )
    if record is None:
        message = "Activation prompt mismatch"
        raise ValueError(message)
    graph, length = learner.graph, protocol.generation.length
    with (path.parent / "activations.npz").open("rb") as stream:
        data: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with data:
            expected = {
                "states": ((length, protocol.graph.nodes), np.dtype("float32")),
                "probabilities": ((length,), np.dtype("float32")),
                "generated_ids": ((length,), np.dtype("int64")),
                "selected_indices": (
                    (min(64, protocol.graph.nodes),),
                    np.dtype("int64"),
                ),
                "node_ids": ((protocol.graph.nodes,), np.asarray(graph.node_ids).dtype),
            }
            require(
                set(data.files) == set(expected) and len(data.files) == len(expected),
                "Activation array membership mismatch",
            )
            for key, (shape, dtype) in expected.items():
                require(
                    data[key].shape == shape and data[key].dtype == dtype,
                    "Activation shape/dtype mismatch",
                )
            states = np.asarray(data["states"], dtype=np.float32)
            probabilities = np.asarray(data["probabilities"], dtype=np.float32)
            require(
                bool(np.isfinite(states).all())
                and bool((np.abs(states) <= 1).all())
                and bool(np.isfinite(probabilities).all())
                and bool(((probabilities >= 0) & (probabilities <= 1)).all()),
                "Activation nonfinite or bounds mismatch",
            )
            averages = cast("NDArray[np.float32]", np.mean(np.abs(states), axis=0))
            scores = TypeAdapter(tuple[float, ...]).validate_python(averages.tolist())
            selected = sorted(range(protocol.graph.nodes), key=lambda i: -scores[i])[
                :64
            ]
            require(
                np.array_equal(data["selected_indices"], selected)
                and np.array_equal(data["node_ids"], np.asarray(graph.node_ids))
                and np.array_equal(data["generated_ids"], np.asarray(record.token_ids)),
                "Activation node/token/selection mismatch",
            )
            expected_metadata = ActivationMetadata(
                config=protocol.config,
                updates=result.update,
                graph_fingerprint=protocol.graph.fingerprint,
                corpus_fingerprint=protocol.corpus.fingerprint,
                parameter_fingerprint=fingerprint,
                prompt=record.prompt,
                prompt_ids=record.prompt_ids,
                generated_ids=record.token_ids,
                token_labels=tuple(
                    stories.corpus.vocabulary[t] for t in record.token_ids
                ),
                generated_text=record.text,
                greedy=True,
                contexts=tuple(
                    (*record.prompt_ids, *record.token_ids[:i]) for i in range(length)
                ),
                selected_node_ids=tuple(graph.node_ids[i] for i in selected),
                sensory_node_ids=tuple(
                    graph.node_ids[int(i.item())]
                    for i in learner.model.sensory.unbind()
                ),
                readout_node_ids=tuple(
                    graph.node_ids[int(i.item())] for i in learner.model.ports.unbind()
                ),
                neurons_total=protocol.graph.nodes,
                figures=figures(length),
            )
            require(
                metadata == expected_metadata,
                "Activation metadata/context/fingerprint mismatch",
            )
    for name in metadata.figures:
        validate_figure(path.parent / name)


def validate_result(
    root: Path,
    result: CheckpointResult,
    report: QualityProgress,
    inputs: tuple[ARLearner, StoryCorpus],
    fingerprint: str,
) -> None:
    """Validate all fixed draws; CPU replay is not a CUDA generation attestation."""
    learner, stories = inputs
    protocol = report.protocol
    draws = 3 if result.update == report.selected_update else 1
    seeds = tuple(
        tuple(10_000 + 100 * i + j for j in range(draws))
        for i in range(len(protocol.prompts))
    )
    identities = tuple(
        (prompt, "greedy" if j == 0 else "sampled", max(0, j - 1), seed)
        for prompt, prompt_seeds in zip(protocol.prompts, seeds, strict=True)
        for j, seed in enumerate((None, *prompt_seeds))
    )
    require(
        tuple((r.prompt, r.mode, r.draw, r.seed) for r in result.generation)
        == identities,
        "Generation identity/count/seed/mode mismatch",
    )
    for record in result.generation:
        require(
            record.prompt_ids == tuple(stories.corpus.encode(record.prompt))
            and len(record.token_ids) == protocol.generation.length
            and all(
                0 <= token < protocol.corpus.vocabulary for token in record.token_ids
            ),
            "Generation token identity/count mismatch",
        )
        require(
            record.text
            == stories.corpus.decode([*record.prompt_ids, *record.token_ids]),
            "Generation decoded text mismatch",
        )
    if torch.device(protocol.config.device).type == "cpu":
        require(
            result.generation
            == generate_quality_panel(
                learner, stories.corpus, protocol.prompts, seeds, protocol.generation
            ),
            "Generation reproduction mismatch",
        )
    for artifact in result.artifacts:
        require(
            file_digest(root / artifact.path) == artifact.sha256,
            "Activation artifact hash mismatch",
        )
    for path_string, index in zip(
        result.activations,
        protocol.activation_prompt_indices if result.activations else (),
        strict=True,
    ):
        path = root / path_string
        require(
            read_model(ActivationMetadata, path).prompt == protocol.prompts[index],
            "Activation prompt index mismatch",
        )
        activation(path, result, inputs, fingerprint, protocol)
