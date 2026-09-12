"""Typed loading and spatial alignment for brain-activity playback."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pyarrow as pa
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from pyarrow import ipc
from pydantic import TypeAdapter
from typing_extensions import override

from flyrl.activations import ActivationMetadata
from flyrl.bpe_tokenizer import restore_tokenizer

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
RenderView = Literal["projections", "rotating_3d"]
MATRIX_DIMENSIONS: Final = 2
SOMA_DIMENSIONS: Final = 3
MINIMUM_POSITIONED_NEURONS: Final = 2


@dataclass(frozen=True, slots=True)
class BrainActivityError(ValueError):
    """Invalid activation, coordinate or rendering input."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class SomaPositions:
    """Activation-column indices with measured soma coordinates."""

    activation_indices: IntArray
    coordinates: FloatArray
    missing_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BrainPlayback:
    """All frame-aligned data needed by the renderer."""

    states: NDArray[np.float32]
    probabilities: NDArray[np.float32]
    token_labels: tuple[str, ...]
    prefix_texts: tuple[str, ...]
    coordinates: FloatArray
    missing_neurons: int

    def __post_init__(self) -> None:
        """Reject mismatched recordings before Matplotlib sees them."""
        frames = len(self.token_labels)
        neurons = len(self.coordinates)
        if (
            self.states.ndim != MATRIX_DIMENSIONS
            or self.states.shape != (frames, neurons)
            or self.probabilities.shape != (frames,)
            or len(self.prefix_texts) != frames
            or self.coordinates.shape != (neurons, 3)
            or frames == 0
            or neurons < MINIMUM_POSITIONED_NEURONS
            or not np.isfinite(self.states).all()
            or not np.isfinite(self.probabilities).all()
            or not np.isfinite(self.coordinates).all()
        ):
            message = "Playback arrays must be finite and frame/neuron aligned"
            raise BrainActivityError(message)


@dataclass(frozen=True, slots=True)
class RenderOptions:
    """Video dimensions and playback rate."""

    output: Path
    frames_per_second: int = 3
    dpi: int = 150
    view: RenderView = "projections"

    def __post_init__(self) -> None:
        """Accept only supported animation containers and useful dimensions."""
        if self.frames_per_second <= 0 or self.dpi <= 0:
            message = "Frame rate and DPI must be positive"
            raise BrainActivityError(message)
        if self.output.suffix.lower() not in {".gif", ".mp4"}:
            message = "Brain activity output must end in .gif or .mp4"
            raise BrainActivityError(message)


def align_soma_positions(
    node_ids: tuple[str, ...],
    annotation_ids: IntArray,
    soma_locations: Sequence[Sequence[int] | None],
) -> SomaPositions:
    """Join soma coordinates while preserving activation-column order."""
    if annotation_ids.ndim != 1 or len(soma_locations) != annotation_ids.size:
        message = (
            "Annotation IDs and soma locations must be one-dimensional and aligned"
        )
        raise BrainActivityError(message)
    parsed_ids = TypeAdapter(list[int]).validate_python(annotation_ids.tolist())
    annotation_lookup = {body_id: index for index, body_id in enumerate(parsed_ids)}
    if len(annotation_lookup) != annotation_ids.size:
        message = "Annotation body IDs must be unique"
        raise BrainActivityError(message)
    indices: list[int] = []
    coordinates: list[tuple[float, float, float]] = []
    missing: list[str] = []
    for activation_index, node_id in enumerate(node_ids):
        try:
            body_id = int(node_id.rsplit(":", 1)[1])
            location = soma_locations[annotation_lookup[body_id]]
        except (IndexError, KeyError, ValueError) as error:
            message = f"Cannot align anatomical node identity: {node_id}"
            raise BrainActivityError(message) from error
        if location is None:
            missing.append(node_id)
            continue
        if len(location) != SOMA_DIMENSIONS:
            message = f"Soma coordinate must have three values: {node_id}"
            raise BrainActivityError(message)
        indices.append(activation_index)
        coordinates.append((float(location[0]), float(location[1]), float(location[2])))
    if len(coordinates) < MINIMUM_POSITIONED_NEURONS:
        message = "At least two neurons need measured soma coordinates"
        raise BrainActivityError(message)
    return SomaPositions(
        np.asarray(indices, dtype=np.int64),
        np.asarray(coordinates, dtype=np.float64),
        tuple(missing),
    )


def activity_emphasis(states: NDArray[np.float32]) -> FloatArray:
    """Scale per-token state changes for point brightness and size."""
    if (
        states.ndim != MATRIX_DIMENSIONS
        or states.size == 0
        or not np.isfinite(states).all()
    ):
        message = "Activity states must be a finite nonempty matrix"
        raise BrainActivityError(message)
    previous = np.empty_like(states)
    previous[0] = 0
    previous[1:] = states[:-1]
    change: FloatArray = np.abs(states - previous).astype(np.float64)
    ranked = TypeAdapter(list[float]).validate_python(change.reshape(-1).tolist())
    ranked.sort()
    scale = ranked[int(0.99 * (len(ranked) - 1))]
    if scale == 0:
        return np.zeros_like(change)
    bounded = np.minimum(np.maximum(change / scale, 0.0), 1.0)
    return np.sqrt(bounded)


def load_playback(
    activation_dir: Path, annotations: Path, tokenizer_path: Path
) -> BrainPlayback:
    """Parse a recovered activation export and join pinned soma coordinates."""
    metadata = ActivationMetadata.model_validate_json(
        (activation_dir / "activations.json").read_text(encoding="utf-8")
    )
    with (
        (activation_dir / metadata.arrays).open("rb") as stream,
        NpzFile[np.generic](stream, allow_pickle=False) as archive,
    ):
        states = np.asarray(archive["states"], dtype=np.float32)
        probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
        generated = TypeAdapter(tuple[int, ...]).validate_python(
            archive["generated_ids"].tolist()
        )
        node_ids = TypeAdapter(tuple[str, ...]).validate_python(
            archive["node_ids"].tolist()
        )
    if generated != metadata.generated_ids:
        message = "Activation JSON and NPZ generated tokens disagree"
        raise BrainActivityError(message)
    with pa.memory_map(str(annotations), "r") as mapped:
        table = ipc.open_file(mapped).read_all()
    positions = align_soma_positions(
        node_ids,
        np.asarray(table.column("bodyId").to_numpy(), dtype=np.int64),
        TypeAdapter(list[list[int] | None]).validate_python(
            table.column("somaLocation").to_pylist()
        ),
    )
    tokenizer = restore_tokenizer(tokenizer_path.read_text(encoding="utf-8"))
    prefix_texts = tuple(
        tokenizer.decode(list(metadata.prompt_ids + generated[: index + 1]))
        for index in range(len(generated))
    )
    if prefix_texts[-1] != metadata.generated_text:
        message = "Tokenizer does not reproduce the recorded generated text"
        raise BrainActivityError(message)
    return BrainPlayback(
        states=states[:, positions.activation_indices],
        probabilities=probabilities,
        token_labels=metadata.token_labels,
        prefix_texts=prefix_texts,
        coordinates=positions.coordinates,
        missing_neurons=len(positions.missing_node_ids),
    )
