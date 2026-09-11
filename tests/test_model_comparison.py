"""Shared optimization, generation context and checkpoint seams for comparisons."""

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from flyrl import ar_learning
from flyrl.ar_benchmark import benchmark
from flyrl.ar_checkpoint import load_checkpoint, save_checkpoint
from flyrl.ar_config import ARConfig
from flyrl.connectome import load_graph


@pytest.mark.parametrize("architecture", ["gru", "transformer"])
def test_dense_architecture_requires_windowed_generation(architecture: str) -> None:
    # Given: a dense model without a defined unbounded generation state.
    # When: parsing the shared model configuration.
    config = ARConfig.model_validate(
        {
            "alphabet_size": 8,
            "architecture": architecture,
            "generation_context": "windowed",
        }
    )
    # Then: its architecture and finite inference context are explicit.
    assert config.architecture == architecture
    with pytest.raises(ValidationError):
        _ = ARConfig.model_validate(
            {**config.model_dump(), "generation_context": "stateful"}
        )
    with pytest.raises(ValidationError):
        _ = ARConfig.model_validate({**config.model_dump(), "control": "frozen"})


@pytest.mark.parametrize("architecture", ["connectome", "gru", "transformer"])
def test_shared_training_and_checkpoint_continuation(
    architecture: str, tmp_path: Path
) -> None:
    # Given: deterministic tokens and the same private window stream.
    config = ARConfig.model_validate(
        {
            "alphabet_size": 8,
            "architecture": architecture,
            "generation_context": "windowed",
            "context": 4,
            "batch_size": 2,
            "eval_windows": 4,
        }
    )
    graph = load_graph(Path("data/large_connectome/malecns_v1_n256.npz"))
    tokens = np.arange(256, dtype=np.int64) % 8
    original = ar_learning.make_learner(graph, config)
    original.train(tokens, 2)
    checkpoint = tmp_path / "checkpoint.npz"
    save_checkpoint(original, checkpoint, "fixture")
    restored = ar_learning.make_learner(graph, config)
    load_checkpoint(restored, checkpoint, "fixture")
    # When: both copies consume additional training windows.
    original.train(tokens, 2)
    restored.train(tokens, 2)
    # Then: parameters, progress, window RNG and generated IDs agree exactly.
    assert original.trace == restored.trace
    assert torch.equal(original.window_rng.get_state(), restored.window_rng.get_state())
    for a, b in zip(
        original.model.parameters(), restored.model.parameters(), strict=True
    ):
        assert torch.equal(a, b)
    assert original.generate([0, 1, 2, 3], 9) == restored.generate([0, 1, 2, 3], 9)


def test_windowed_generation_discards_tokens_before_context() -> None:
    # Given: equal final contexts preceded by different older tokens.
    graph = load_graph(Path("data/large_connectome/malecns_v1_n256.npz"))
    config = ARConfig.model_validate(
        {"alphabet_size": 8, "context": 3, "generation_context": "windowed"}
    )
    learner = ar_learning.make_learner(graph, config)
    # When: generating greedily with the explicitly bounded context.
    a = learner.generate([7, 7, 1, 2, 3], 5, greedy=True)
    b = learner.generate([0, 0, 1, 2, 3], 5, greedy=True)
    # Then: inaccessible history cannot influence any generated token.
    assert a == b


@pytest.mark.parametrize("architecture", ["gru", "transformer"])
def test_dense_capacity_report_does_not_invent_anatomy(architecture: str) -> None:
    # Given: a dense reference with no use of the provided connectome graph.
    config = ARConfig.model_validate(
        {
            "alphabet_size": 8,
            "architecture": architecture,
            "generation_context": "windowed",
            "context": 3,
            "batch_size": 2,
        }
    )
    graph = load_graph(Path("data/large_connectome/malecns_v1_n256.npz"))
    learner = ar_learning.make_learner(graph, config)
    # When: measuring its real forward/backward/update path.
    measured = benchmark(learner)
    # Then: capacity evidence names the architecture without claiming neural edges.
    assert measured.architecture == architecture
    assert measured.vocabulary_tokens == 8
    assert measured.nodes == measured.edges == 0
