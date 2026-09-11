"""Story-safe sampling and opt-in sensory learning use the original optimizer."""

from pathlib import Path

import numpy as np
import pytest
import torch

from flyrl.ar_checkpoint import load_checkpoint, save_checkpoint
from flyrl.ar_config import ARConfig
from flyrl.ar_framework import optimizer_tensors
from flyrl.ar_learning import ARLearner
from flyrl.connectome import Graph


@pytest.fixture
def graph() -> Graph:
    source, target = np.nonzero(np.ones((12, 12)) - np.eye(12))
    return Graph(
        tuple(f"synthetic:{i}" for i in range(12)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size, dtype=np.float64),
        "synthetic test circuit",
    )


def test_trainable_codes_are_causal_and_receive_gradients(graph: Graph) -> None:
    # Given: opt-in and otherwise identically initialized legacy models.
    config = ARConfig(alphabet_size=2, context=3, trainable_codes=True)
    learner = ARLearner(graph, config)
    fixed = ARLearner(graph, ARConfig(alphabet_size=2, context=3))
    assert torch.equal(learner.model.codes, fixed.model.codes)
    assert "codes" in dict(fixed.model.named_buffers())
    assert "codes" not in dict(fixed.model.named_parameters())
    tokens = torch.tensor([[0, 1, 0, 1]])
    changed = torch.tensor([[0, 1, 1, 0]])
    original = learner.model.codes.detach().clone()
    # When: scoring a changed suffix and learning on next-token targets.
    logits = learner.model.forward(tokens)
    alternative = learner.model.forward(changed)
    learner.train(np.array([0, 1] * 12, dtype=np.int64), 1)
    # Then: the prefix is causal and sensory codes actually learn.
    assert torch.equal(logits[:, :2], alternative[:, :2])
    assert not torch.equal(original, learner.model.codes)
    assert learner.model.codes.grad is not None
    assert torch.count_nonzero(learner.model.codes.grad).item() > 0


def test_story_starts_and_trainable_codes_resume_exactly(
    graph: Graph,
    tmp_path: Path,
) -> None:
    # Given: only two legal story windows with forbidden transitions between them.
    config = ARConfig(alphabet_size=2, context=2, batch_size=3, trainable_codes=True)
    tokens = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    starts = np.array([0, 3], dtype=np.int64)
    full, partial = ARLearner(graph, config), ARLearner(graph, config)
    full.train(tokens, 4, starts=starts)
    partial.train(tokens, 2, starts=starts)
    checkpoint = tmp_path / "checkpoint.npz"
    save_checkpoint(partial, checkpoint, "stories-with-boundaries")
    # When: loading through the original checkpoint interface and continuing.
    resumed = ARLearner(graph, config)
    load_checkpoint(resumed, checkpoint, "stories-with-boundaries")
    resumed.train(tokens, 2, starts=starts)
    # Then: parameters, Adam moments and RNG match exactly.
    assert resumed.trace == full.trace
    assert torch.equal(full.window_rng.get_state(), resumed.window_rng.get_state())
    for name, value in full.model.named_parameters():
        assert torch.equal(value, resumed.model.get_parameter(name))
    for left, right in zip(
        optimizer_tensors(full.optimizer).values(),
        optimizer_tensors(resumed.optimizer).values(),
        strict=True,
    ):
        for key in left:
            assert torch.equal(left[key], right[key])


def test_sampler_uses_only_supplied_starts(graph: Graph) -> None:
    # Given: a single allowed window in the second story.
    config = ARConfig(alphabet_size=2, context=2, batch_size=2)
    learner, reference = ARLearner(graph, config), ARLearner(graph, config)
    tokens = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    # When: restricting training versus training the isolated story directly.
    learner.train(tokens, 1, starts=np.array([3], dtype=np.int64))
    reference.train(tokens[3:], 1)
    # Then: the shared optimizer saw the same batch, not a crossing window.
    assert learner.trace == reference.trace
    for name, value in reference.model.named_parameters():
        assert torch.equal(value, learner.model.get_parameter(name))
