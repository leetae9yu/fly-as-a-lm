"""Answer-only optimization on a real, small anatomical recurrent fixture."""

from collections import Counter

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from flyrl.ar_config import ARConfig
from flyrl.ar_framework import optimizer_tensors
from flyrl.babi_learning import (
    BabiExample,
    BabiLearner,
    answer_loss,
    make_batch,
)
from flyrl.babi_metrics import evaluate_nll
from flyrl.connectome import Graph
from flyrl.language_data import CorpusError


def make_learner(*, context: int = 12) -> BabiLearner:
    source, target = np.nonzero(np.ones((12, 12)) - np.eye(12))
    graph = Graph(
        tuple(f"fixture:{i}" for i in range(12)),
        source.astype(np.int64),
        target.astype(np.int64),
        np.ones(source.size, dtype=np.float64),
        "synthetic bAbI core fixture",
    )
    return BabiLearner(
        graph,
        ARConfig(
            alphabet_size=4096,
            tokenization="bpe",
            trainable_codes=True,
            context=context,
            batch_size=2,
        ),
    )


def examples() -> tuple[BabiExample, ...]:
    return (
        BabiExample(example_id="a", prompt_ids=(10, 11, 12), answer_ids=(20, 199)),
        BabiExample(example_id="b", prompt_ids=(13, 14), answer_ids=(21, 22, 199)),
        BabiExample(example_id="c", prompt_ids=(15,), answer_ids=(23, 199)),
    )


def test_first_answer_and_terminator_mask() -> None:
    # Given: both two- and three-token answers, including hallway's split spelling.
    data = examples()[:2]
    # When: shifted sequences are padded to the training context.
    batch = make_batch(data, 6)
    # Then: only first-answer through terminator targets carry loss.
    assert batch.inputs == ((10, 11, 12, 20, 199, 199), (13, 14, 21, 22, 199, 199))
    assert batch.targets == (
        (-100, -100, 20, 199, -100, -100),
        (-100, 21, 22, 199, -100, -100),
    )
    assert batch.answer_counts == (2, 3)


def test_hallway_has_equal_example_weight() -> None:
    # Given: deliberately unequal token difficulty and answer lengths.
    batch = make_batch(examples()[:2], 6)
    logits = torch.zeros((2, 6, 4096), requires_grad=True)
    with torch.no_grad():
        logits[0, :, 199] = 4
    targets = torch.tensor(batch.targets)
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 4096), targets.flatten(), reduction="none"
    ).reshape(2, 6)
    expected = (losses[0].sum() / 2 + losses[1].sum() / 3) / 2
    # When: the core reduces the answer-only objective.
    actual = answer_loss(logits, batch)
    # Then: it is a mean of question means, not pooled token NLL.
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, losses.sum() / 5)
    gradient = torch.autograd.grad(actual, logits)[0]
    assert torch.count_nonzero(gradient[targets == -100]) == 0
    assert torch.count_nonzero(gradient[targets != -100]) > 0


def test_padding_invariance_and_full_prompt_gradient() -> None:
    # Given: a real model and distinct token IDs at every prompt position.
    learner = make_learner()
    data = examples()[:1]
    short, padded = make_batch(data, 4), make_batch(data, 12)
    # When: computing losses and gradients with different right padding lengths.
    first = learner.loss(short)
    first_gradient = torch.autograd.grad(first, learner.model.codes)[0]
    second = learner.loss(padded)
    second_gradient = torch.autograd.grad(second, learner.model.codes)[0]
    # Then: padding is irrelevant and every prompt token receives recurrent gradient.
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_gradient, second_gradient, rtol=0, atol=0)
    for token in data[0].prompt_ids:
        assert torch.count_nonzero(first_gradient[token]) > 0
    assert torch.count_nonzero(first_gradient[199]) == 0


def test_replacement_samples_exposures_and_evaluation_isolation() -> None:
    # Given: two identically initialized learners and the specified private RNG.
    left, right = make_learner(), make_learner()
    data = examples()
    rng = torch.Generator().manual_seed(37)
    expected = tuple(
        tuple(data[int(i)].example_id for i in torch.randint(3, (2,), generator=rng))
        for _ in range(3)
    )
    # When: evaluation is interleaved with one learner's updates only.
    for _ in range(3):
        _ = left.update(data)
        state = left.window_rng.get_state().clone()
        adam = tuple(
            t.clone()
            for s in optimizer_tensors(left.optimizer).values()
            for t in s.values()
        )
        _ = evaluate_nll(left.model, data)
        assert torch.equal(state, left.window_rng.get_state())
        for before, after in zip(
            adam,
            (t for s in optimizer_tensors(left.optimizer).values() for t in s.values()),
            strict=True,
        ):
            assert torch.equal(before, after)
        _ = right.update(data)
    # Then: samples, exposure accounting, all trainable tensors and traces agree.
    assert tuple(entry.sampled_ids for entry in left.trace) == expected
    assert left.exposure_counts == Counter(
        identity for batch in expected for identity in batch
    )
    assert left.updates == 3
    assert left.trace == right.trace
    assert sum(left.exposure_counts.values()) == 6
    for a, b in zip(left.model.parameters(), right.model.parameters(), strict=True):
        assert torch.equal(a, b)
    assert left.optimizer.defaults["betas"] == (0.9, 0.999)
    assert left.optimizer.defaults["eps"] == 1e-8
    assert left.optimizer.defaults["foreach"] is False


@pytest.mark.parametrize(
    ("prompt", "answer"),
    [
        ((), (20, 199)),
        ((10,), ()),
        ((10,), (199,)),
        ((10,), (20,)),
        ((10,), (199, 20, 199)),
        ((-1,), (20, 199)),
        ((4096,), (20, 199)),
        ((10,), (4096, 199)),
        ((True,), (20, 199)),
    ],
)
def test_invalid_example_rejected(
    prompt: tuple[int, ...], answer: tuple[int, ...]
) -> None:
    # Given/When/Then: malformed boundary tokens cannot become a core example.
    with pytest.raises(ValidationError):
        _ = BabiExample(example_id="bad", prompt_ids=prompt, answer_ids=answer)


def test_context_boundary_empty_batches_and_duplicate_identity() -> None:
    # Given: exactly four causal inputs, plus a terminator target.
    data = examples()[:1]
    # When/Then: the exact boundary succeeds; overflow and empty batches fail.
    assert len(make_batch(data, 4).inputs[0]) == 4
    with pytest.raises(CorpusError):
        _ = make_batch(data, 3)
    with pytest.raises(CorpusError):
        _ = make_batch((), 4)
    with pytest.raises(CorpusError):
        _ = make_batch(data, 0)
    learner = make_learner()
    with pytest.raises(CorpusError):
        _ = learner.update((data[0], data[0]))


@pytest.mark.parametrize("corrupt_adam", [False, True])
def test_nonfinite_update_rejected(*, corrupt_adam: bool) -> None:
    # Given: a nonfinite parameter or Adam moment at an update boundary.
    learner = make_learner()
    _ = learner.update(examples())
    with torch.no_grad():
        tensor = (
            optimizer_tensors(learner.optimizer)[learner.model.weight]["exp_avg"]
            if corrupt_adam
            else learner.model.weight
        )
        _ = tensor.fill_(float("nan"))
    # When/Then: invalid state is rejected rather than recorded as a valid update.
    with pytest.raises(CorpusError):
        _ = learner.update(examples())
    assert learner.updates == 1
