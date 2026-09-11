"""Exact all-position denominators, evaluation isolation and finite-loss overflow."""

from math import log

import numpy as np
import pytest
import torch

from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.connectome import Graph
from flyrl.story_data import StorySplit
from flyrl.story_metrics import evaluate_story_split


@pytest.fixture
def learner() -> ARLearner:
    source, target = np.nonzero(np.ones((4, 4)) - np.eye(4))
    graph = Graph(
        tuple(f"synthetic:{i}" for i in range(4)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size, dtype=np.float64),
        "synthetic metrics test",
    )
    return ARLearner(
        graph,
        ARConfig(
            alphabet_size=2,
            context=2,
            trainable_codes=True,
        ),
    )


def test_evaluation_scores_every_pair_without_crossing_or_resetting_chunks(
    learner: ARLearner,
) -> None:
    # Given: two unequal stories and independently computed full-story logits.
    tokens = np.array([0, 1, 1, 0, 1, 1, 0], dtype=np.int64)
    split = StorySplit(offsets=(0, 4, 7), sha256=("first", "second"))
    logits = torch.cat(
        [
            learner.model.forward(torch.tensor([[0, 1, 1]]))[0],
            learner.model.forward(torch.tensor([[1, 1]]))[0],
        ]
    )
    targets = torch.tensor([1, 1, 0, 1, 0])
    expected = torch.nn.functional.cross_entropy(logits, targets).item()
    rng = learner.window_rng.get_state().clone()
    parameters = {
        name: value.detach().clone() for name, value in learner.model.named_parameters()
    }
    # When: evaluating through bounded chunks smaller than the first story.
    result = evaluate_story_split(learner, tokens, split)
    # Then: every within-story pair is counted once and state/RNG are isolated.
    assert result.tokens == 5
    assert result.stories == 2
    assert result.nll == pytest.approx(expected)
    assert result.accuracy == pytest.approx(
        (logits.argmax(dim=1) == targets).float().mean().item()
    )
    assert torch.equal(rng, learner.window_rng.get_state())
    assert learner.updates == 0
    for name, value in parameters.items():
        assert torch.equal(value, learner.model.get_parameter(name))


def test_finite_large_nll_reports_null_perplexity(learner: ARLearner) -> None:
    # Given: finite logits with loss larger than the floating-point exp domain.
    with torch.no_grad():
        _ = learner.model.readout.zero_()
        _ = learner.model.output_bias.copy_(torch.tensor([1000.0, 0.0]))
    # When: all next-token targets have negligible modeled probability.
    result = evaluate_story_split(
        learner,
        np.array([1, 1, 1], dtype=np.int64),
        StorySplit(offsets=(0, 3), sha256=("one",)),
    )
    # Then: finite NLL survives reporting without an exp overflow crash.
    assert result.nll == 1000.0
    assert result.perplexity is None
    assert result.nll > log(2)


@pytest.mark.parametrize("architecture", ["gru", "transformer"])
def test_trainable_sensory_codes_reject_dense_architectures(architecture: str) -> None:
    # Given/When/Then: opting into anatomical inputs on a dense model is an error.
    with pytest.raises(ValueError, match="sensory"):
        _ = ARConfig.model_validate(
            {
                "alphabet_size": 2,
                "architecture": architecture,
                "generation_context": "windowed",
                "trainable_codes": True,
            }
        )
