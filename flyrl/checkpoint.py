"""Atomic, validated JSON checkpoints; no pickle or opaque Python objects."""

import os
import platform
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from flyrl.connectome import Graph
from flyrl.learning import ExperimentError, Learner
from flyrl.models import Config, Ports, Settings


class PCGState(Settings):
    """PCG64's two exact 128-bit integers."""

    state: Annotated[int, Field(ge=0, lt=2**128)]
    inc: Annotated[int, Field(ge=0, lt=2**128)]


class RandomState(Settings):
    """NumPy PCG64 state, including the cached 32-bit integer."""

    bit_generator: Literal["PCG64"]
    state: PCGState
    has_uint32: Literal[0, 1]
    uinteger: Annotated[int, Field(ge=0, lt=2**32)]


class GraphData(Settings):
    """Portable sparse graph representation."""

    node_ids: tuple[str, ...]
    source: tuple[int, ...]
    target: tuple[int, ...]
    weight: tuple[float, ...]
    provenance: str

    @classmethod
    def from_graph(cls, graph: Graph) -> "GraphData":
        """Freeze array values into JSON-compatible typed tuples."""
        return cls(
            node_ids=graph.node_ids,
            source=tuple(int(x) for x in graph.source),
            target=tuple(int(x) for x in graph.target),
            weight=tuple(float(x) for x in graph.weight),
            provenance=graph.provenance,
        )

    def graph(self) -> Graph:
        """Parse sparse arrays through Graph's validation boundary."""
        return Graph(
            self.node_ids,
            np.array(self.source, dtype=np.int64),
            np.array(self.target, dtype=np.int64),
            np.array(self.weight, dtype=np.float64),
            self.provenance,
        )


class Checkpoint(Settings):
    """Complete versioned state at an episode boundary."""

    format_version: Literal[1] = 1
    numpy_version: str
    python_version: str
    config: Config
    graph: GraphData
    ports: Ports
    weights: tuple[float, ...]
    eligibility: tuple[float, ...]
    baseline: Annotated[float, Field(ge=0, le=1)]
    rewards: tuple[Literal[0, 1], ...]
    rng: RandomState


def atomic_write(path: Path, content: str) -> None:
    """Flush a sibling temporary file, then atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            _ = stream.write(content.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
            _ = temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def save_checkpoint(learner: Learner, path: Path) -> None:
    """Atomically save complete learning state without pickle."""
    checkpoint = Checkpoint(
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        config=learner.config,
        graph=GraphData.from_graph(learner.graph),
        ports=learner.ports,
        weights=tuple(float(x) for x in learner.weights),
        eligibility=tuple(float(x) for x in learner.eligibility),
        baseline=learner.baseline,
        rewards=tuple(1 if reward else 0 for reward in learner.rewards),
        rng=RandomState.model_validate(learner.rng.bit_generator.state),
    )
    atomic_write(path, checkpoint.model_dump_json())


def load_checkpoint(path: Path, expected: Learner | None = None) -> Learner:
    """Restore exact state; reject incompatible versions, shapes, or settings."""
    saved = Checkpoint.model_validate_json(path.read_bytes())
    if (
        saved.numpy_version != np.__version__
        or saved.python_version != platform.python_version()
    ):
        raise ExperimentError(
            reason="checkpoint requires matching NumPy and Python versions"
        )
    if expected is not None and (
        saved.config != expected.config
        or saved.ports != expected.ports
        or saved.graph != GraphData.from_graph(expected.graph)
    ):
        raise ExperimentError(reason="checkpoint graph, ports, or config mismatch")
    learner = Learner(saved.graph.graph(), saved.config, saved.ports)
    weights = np.array(saved.weights, dtype=np.float64)
    eligibility = np.array(saved.eligibility, dtype=np.float64)
    if weights.shape != learner.weights.shape or eligibility.shape != weights.shape:
        raise ExperimentError(reason="checkpoint synapse-state shape mismatch")
    if (
        np.not_equal(np.sign(weights), np.sign(learner.graph.weight)).any()
        or (np.abs(weights) > saved.config.max_weight).any()
    ):
        raise ExperimentError(
            reason="checkpoint weights violate original signs or bounds"
        )
    learner.weights = weights
    learner.eligibility = eligibility
    learner.baseline = saved.baseline
    learner.rewards = list(saved.rewards)
    learner.rng.bit_generator.state = saved.rng.model_dump()
    return learner
