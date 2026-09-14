"""Independent dense recurrence checks for immutable heldout story evidence."""

from math import exp

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from flyrl.ar_config import ARConfig
from flyrl.ar_framework import optimizer_tensors
from flyrl.ar_learning import ARLearner
from flyrl.connectome import Graph
from flyrl.language_data import IntVector
from flyrl.quality_metrics import QualityMetrics, StoryEvidence, evaluate_quality_split
from flyrl.story_data import StorySplit


@pytest.fixture
def learner() -> ARLearner:
    source, target = np.nonzero(np.ones((4, 4)) - np.eye(4))
    graph = Graph(
        tuple(f"synthetic:{i}" for i in range(4)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size, dtype=np.float64),
        "synthetic quality metrics test",
    )
    return ARLearner(
        graph,
        ARConfig(alphabet_size=2, context=2, batch_size=2, trainable_codes=True),
    )


@torch.no_grad()
def dense_score(learner: ARLearner, tokens: IntVector) -> tuple[float, int]:
    model = learner.model
    matrix = torch.zeros((model.nodes, model.nodes))
    matrix[model.edges[0], model.edges[1]] = model.weight
    state = torch.zeros((model.nodes, 1))
    loss, correct = 0.0, 0
    for position in range(tokens.size - 1):
        current, target = int(tokens.item(position)), int(tokens.item(position + 1))
        drive = torch.zeros_like(state)
        drive[model.sensory] = model.codes[current, :, None]
        state = (1 - learner.config.leak) * state + learner.config.leak * torch.tanh(
            matrix @ (state + drive) + model.bias[:, None] + drive
        )
        logits = (state[model.ports].T @ model.readout + model.output_bias)[0]
        loss += float((logits.logsumexp(dim=0) - logits[target]).item())
        correct += int(int(logits.argmax().item()) == target)
    return loss, correct


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 20])
def test_each_pair_matches_dense_equations_with_story_resets(
    learner: ARLearner, chunk_size: int
) -> None:
    tokens = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1], dtype=np.int64)
    split = StorySplit(offsets=(0, 6, 9), sha256=("a" * 64, "b" * 64))
    result = evaluate_quality_split(learner, tokens, split, chunk_size=chunk_size)
    assert result.count == 7
    assert tuple(story.sha256 for story in result.per_story) == split.sha256
    for story, start, stop in zip(
        result.per_story, split.offsets[:-1], split.offsets[1:], strict=True
    ):
        loss, correct = dense_score(learner, tokens[start:stop])
        assert story.loss_sum == pytest.approx(loss)
        assert story.count == stop - start - 1
        assert story.correct == correct
    assert result == evaluate_quality_split(learner, tokens, split, chunk_size=1)
    assert result == evaluate_quality_split(learner, tokens, split)


def test_empty_and_singleton_stories_remain_explicit(learner: ARLearner) -> None:
    tokens = np.array([1, 0, 1], dtype=np.int64)
    split = StorySplit(offsets=(0, 0, 1, 3, 3), sha256=("a", "b", "c", "d"))
    result = evaluate_quality_split(learner, tokens, split)
    assert len(result.per_story) == 4
    assert result.count == 1
    for index in (0, 1, 3):
        story = result.per_story[index]
        assert story.sha256 == split.sha256[index]
        assert (story.loss_sum, story.count, story.correct) == (0.0, 0, 0)


@pytest.mark.parametrize("offsets", [(0,), (0, 0), (0, 1), (0, 0, 1, 1)])
def test_no_targets_have_no_fabricated_mean(
    learner: ARLearner, offsets: tuple[int, ...]
) -> None:
    split = StorySplit(offsets=offsets, sha256=tuple("a" for _ in offsets[1:]))
    result = evaluate_quality_split(
        learner, np.zeros(offsets[-1], dtype=np.int64), split
    )
    assert len(result.per_story) == len(split.sha256)
    assert (result.loss_sum, result.count, result.correct) == (0.0, 0, 0)
    assert result.nll is None
    assert result.perplexity is None
    assert result.accuracy is None


def test_aggregation_uses_additive_token_weights() -> None:
    result = QualityMetrics(
        per_story=(
            StoryEvidence(sha256="a", loss_sum=1.0, count=1, correct=1),
            StoryEvidence(sha256="b", loss_sum=27.0, count=9, correct=3),
            StoryEvidence(sha256="c", loss_sum=0.0, count=0, correct=0),
        )
    )
    assert result.loss_sum == 28.0
    assert result.count == 10
    assert result.correct == 4
    assert result.nll == 2.8
    assert result.accuracy == 0.4
    assert result.perplexity == exp(2.8)
    assert result.perplexity != pytest.approx((exp(1) + exp(3)) / 2)


def test_large_finite_loss_does_not_overflow(learner: ARLearner) -> None:
    with torch.no_grad():
        _ = learner.model.readout.zero_()
        _ = learner.model.output_bias.copy_(torch.tensor([1000.0, 0.0]))
    result = evaluate_quality_split(
        learner,
        np.array([1, 1], dtype=np.int64),
        StorySplit(offsets=(0, 2), sha256=("a",)),
    )
    assert result.nll == 1000.0
    assert result.perplexity is None


@pytest.mark.parametrize(
    ("offsets", "identities"),
    [
        ((), ()),
        ((0,), ("a",)),
        ((0, 2, 3), ("a",)),
        ((1, 3), ("a",)),
        ((0, 2), ("a",)),
        ((0, 4, 3), ("a", "b")),
        ((0, -1, 3), ("a", "b")),
        ((0, 2, 1, 3), ("a", "b", "c")),
    ],
)
def test_invalid_boundaries_rejected(
    learner: ARLearner, offsets: tuple[int, ...], identities: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="boundaries"):
        _ = evaluate_quality_split(
            learner,
            np.array([0, 1, 0], dtype=np.int64),
            StorySplit(offsets=offsets, sha256=identities),
        )


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_chunk_size_rejected(learner: ARLearner, chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk"):
        _ = evaluate_quality_split(
            learner,
            np.array([0, 1], dtype=np.int64),
            StorySplit(offsets=(0, 2), sha256=("a",)),
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize("training", [True, False])
def test_evaluation_preserves_trained_learner_and_inputs(
    learner: ARLearner, training: bool
) -> None:
    tokens = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    learner.train(tokens, 1)
    _ = learner.model.train(training)
    split = StorySplit(offsets=(0, 3, 6), sha256=("a", "b"))
    before_tokens = tokens.copy()
    model_tensors = dict(learner.model.named_parameters()) | dict(
        learner.model.named_buffers()
    )
    before_model = {
        name: value.detach().clone() for name, value in model_tensors.items()
    }
    before_gradients = tuple(
        parameter.grad.clone() if parameter.grad is not None else None
        for parameter in learner.model.parameters()
    )
    before_optimizer = {
        parameter: {name: value.clone() for name, value in state.items()}
        for parameter, state in optimizer_tensors(learner.optimizer).items()
    }
    before_rng = learner.window_rng.get_state().clone()
    before_global_rng = torch.get_rng_state().clone()
    before_trace = tuple(learner.trace)
    result = evaluate_quality_split(learner, tokens, split)
    assert result.count == 4
    assert learner.model.training is training
    assert learner.updates == 1
    assert tuple(learner.trace) == before_trace
    assert torch.equal(before_rng, learner.window_rng.get_state())
    assert torch.equal(before_global_rng, torch.get_rng_state())
    assert np.array_equal(tokens, before_tokens)
    after_model = dict(learner.model.named_parameters()) | dict(
        learner.model.named_buffers()
    )
    assert after_model.keys() == before_model.keys()
    for name, value in after_model.items():
        assert torch.equal(value, before_model[name])
    for parameter, gradient in zip(
        learner.model.parameters(), before_gradients, strict=True
    ):
        if gradient is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, gradient)
    after_optimizer = optimizer_tensors(learner.optimizer)
    assert before_optimizer.keys() == after_optimizer.keys()
    for parameter, state in before_optimizer.items():
        for name, value in state.items():
            assert torch.equal(value, after_optimizer[parameter][name])


def test_evidence_is_deeply_immutable(learner: ARLearner) -> None:
    result = evaluate_quality_split(
        learner,
        np.array([0, 1], dtype=np.int64),
        StorySplit(offsets=(0, 2), sha256=("a",)),
    )
    with pytest.raises(ValidationError, match="frozen"):
        result.per_story[0].loss_sum = 0.0
    with pytest.raises(ValidationError, match="frozen"):
        result.per_story = ()
    assert isinstance(result.per_story, tuple)
    assert QualityMetrics.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    ("loss_sum", "count", "correct"),
    [(1.0, 0, 0), (0.0, 0, 1), (1.0, 1, 2), (-1.0, 1, 0), (0.0, -1, 0)],
)
def test_impossible_evidence_rejected(
    loss_sum: float, count: int, correct: int
) -> None:
    with pytest.raises(ValidationError):
        _ = StoryEvidence(sha256="a", loss_sum=loss_sum, count=count, correct=correct)


@pytest.mark.parametrize("token", [-1, 2])
def test_out_of_vocabulary_tokens_rejected(learner: ARLearner, token: int) -> None:
    with pytest.raises(ValueError, match="in-vocabulary"):
        _ = evaluate_quality_split(
            learner,
            np.array([0, token], dtype=np.int64),
            StorySplit(offsets=(0, 2), sha256=("a",)),
        )
