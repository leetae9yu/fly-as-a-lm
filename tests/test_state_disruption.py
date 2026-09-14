"""Dense oracles for outgoing centering, story alignment and frozen replay."""

from itertools import pairwise
from typing import Literal

import numpy as np
import pytest
import torch
from torch.nn.functional import cross_entropy

from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.ar_model import ConnectomeLM
from flyrl.connectome import Graph
from flyrl.regional_probe import extract_features, fit_probe
from flyrl.regional_probe_types import ExtractionConfig, FeatureCache, ProbeConfig
from flyrl.state_disruption import (
    CenteredOutgoing,
    evaluate_intervention,
    replay_original_readout,
    replay_probe,
)
from flyrl.story_data import StorySplit


@pytest.fixture
def learner() -> ARLearner:
    edges = np.arange(16, dtype=np.int64)
    graph = Graph(
        tuple(map(str, range(4))),
        edges % 4,
        edges // 4,
        np.ones(16, dtype=np.float64),
        "Dense synthetic graph",
    )
    config = ARConfig(
        alphabet_size=3,
        leak=0.4,
        readout_neurons=2,
        port_policy="random_random",
        port_manifest_sha256="a" * 64,
        sensory_indices=(0, 1),
        readout_indices=(3, 2),
    )
    learner = ARLearner(graph, config)
    m = learner.model
    with torch.no_grad():
        _ = m.weight.copy_(torch.arange(16) * 0.025 - 0.15)
        _ = m.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.1]))
        _ = m.codes.copy_(torch.tensor([[1.0, -0.5], [-0.2, 0.7], [0.3, 0.4]]))
        _ = m.readout.copy_(torch.tensor([[0.4, -0.7, 0.2], [-0.3, 0.8, 0.1]]))
        _ = m.output_bias.copy_(torch.tensor([0.2, -0.1, 0.3]))
    return learner


@pytest.fixture
def arm() -> CenteredOutgoing:
    return CenteredOutgoing((0, 2), (0.35, -0.2, -0.45, 0.1))


@torch.no_grad()
def dense(
    m: ConnectomeLM,
    tokens: torch.Tensor,
    arm: CenteredOutgoing | None,
    wrong: Literal["none", "post", "incoming", "drop_group", "first"] = "none",
) -> torch.Tensor:
    # Fixture edges enumerate each dense [recipient, source] entry once.
    matrix = m.weight.reshape(4, 4)
    h = torch.zeros((4, tokens.shape[0]))
    states: list[torch.Tensor] = []
    for t, token in enumerate(tokens.unbind(dim=1)):
        drive = torch.zeros_like(h)
        drive[m.sensory] = m.codes[token].T
        signal, weights = h + drive, matrix.clone()
        active = arm is not None and (t > 0 or wrong == "first")
        group = [] if arm is None else list(arm.indices)
        mean = (
            torch.zeros((0, 1))
            if arm is None
            else torch.tensor(arm.training_mean)[group, None]
        )
        if active:
            if wrong in {"none", "first", "drop_group"}:
                signal[group] = mean + drive[group]
            if wrong == "incoming":
                weights[group] = 0
            if wrong == "drop_group":
                for recipient in group:
                    weights[recipient, group] = 0
        h = 0.6 * h + 0.4 * torch.tanh(weights @ signal + m.bias[:, None] + drive)
        if active and wrong == "post":
            h[group] = mean
        states.append(h.T)
    return torch.stack(states, dim=1)


def test_default_and_empty_are_bitwise_legacy(learner: ARLearner) -> None:
    m, tokens, selected = learner.model, torch.tensor([[0, 1, 2, 1]]), torch.arange(4)
    baseline, final = m.selected_states(tokens, selected)
    empty, end = m.selected_states(
        tokens, selected, intervention=CenteredOutgoing((), (0.3,) * 4)
    )
    assert torch.equal(baseline, empty)
    assert torch.equal(final, end)
    h = torch.zeros_like(final)
    with torch.no_grad():
        for t, token in enumerate(tokens.unbind(dim=1)):
            _, h = m.step(token, h)
            assert torch.equal(baseline[:, t], h.T)
        assert torch.equal(
            m.forward(tokens), baseline[:, :, m.ports] @ m.readout + m.output_bias
        )


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_dense_mechanism_first_token_and_chunk_carry(
    learner: ARLearner, arm: CenteredOutgoing, cut: int
) -> None:
    m, tokens, selected = (
        learner.model,
        torch.tensor([[0, 1, 2, 1], [2, 0, 1, 0]]),
        torch.arange(4),
    )
    actual, final = m.selected_states(tokens, selected, intervention=arm)
    expected = dense(m, tokens, arm)
    torch.testing.assert_close(actual, expected, rtol=0, atol=6e-8)
    torch.testing.assert_close(final.T, expected[:, -1], rtol=0, atol=6e-8)
    baseline = dense(m, tokens, None)
    assert torch.equal(actual[:, 0], baseline[:, 0])
    assert not torch.allclose(actual[:, 1:], baseline[:, 1:])
    for wrong in ("post", "incoming", "drop_group", "first"):
        assert not torch.allclose(actual, dense(m, tokens, arm, wrong))
    first, carry = m.selected_states(tokens[:, :cut], selected, intervention=arm)
    saved = carry.clone()
    second, end = m.selected_states(
        tokens[:, cut:], selected, state=carry, intervention=arm
    )
    assert torch.equal(carry, saved)
    assert torch.equal(final, end)
    assert torch.equal(actual, torch.cat((first, second), dim=1))
    # A supplied zero state still denotes a consumed prefix, not a story reset.
    zero = torch.zeros_like(carry)
    active, _ = m.selected_states(tokens[:, :1], selected, state=zero, intervention=arm)
    assert not torch.equal(active, actual[:, :1])


@pytest.mark.parametrize(("chunk", "batch"), [(1, 1), (2, 3), (8, 6)])
def test_stories_padding_lanes_and_exact_replay(
    learner: ARLearner, arm: CenteredOutgoing, chunk: int, batch: int
) -> None:
    m = learner.model
    tokens = np.array([0, 1, 2, 0, 2, 1, 2, 0, 1, 2], dtype=np.int64)
    split = StorySplit(offsets=(0, 4, 4, 5, 7, 10), sha256=("a", "b", "c", "d", "e"))
    config = ExtractionConfig(chunk_size=chunk, story_batch_size=batch)
    result = evaluate_intervention(m, tokens, split, intervention=arm, config=config)
    expected = [
        dense(m, torch.tensor(tokens[a : b - 1][None]), arm)[0, :, m.ports]
        for a, b in pairwise(split.offsets)
        if b - a > 1
    ]
    cache, scores = result.cache, result.scores
    torch.testing.assert_close(
        torch.tensor(cache.features), torch.cat(expected), rtol=0, atol=6e-8
    )
    np.testing.assert_array_equal(cache.labels, [1, 2, 0, 2, 1, 2])
    assert not cache.features.flags.writeable
    assert not cache.labels.flags.writeable
    assert scores.story_sha256 == split.sha256
    assert tuple(s.count for s in scores.stories) == (3, 0, 0, 1, 2)
    head = m.readout.detach().numpy().copy(), m.output_bias.detach().numpy().copy()
    assert scores == replay_original_readout(cache, split, head)
    logits = torch.tensor(cache.features) @ m.readout.detach() + m.output_bias.detach()
    targets = torch.tensor(cache.labels)
    losses = cross_entropy(logits, targets, reduction="none")
    for s, (a, b) in zip(scores.stories, pairwise((0, 3, 3, 3, 4, 6)), strict=True):
        assert s.loss_sum == float(losses[a:b].double().sum())
        assert s.correct == int((logits[a:b].argmax(1) == targets[a:b]).sum())
    assert scores.total.count == 6
    assert scores.total.nll == float(losses.double().sum()) / 6
    assert scores.stories[1].nll is None
    baseline = evaluate_intervention(m, tokens, split)
    empty = evaluate_intervention(
        m, tokens, split, intervention=CenteredOutgoing((), arm.training_mean)
    )
    np.testing.assert_array_equal(baseline.cache.features, empty.cache.features)
    assert baseline.scores == empty.scores
    np.testing.assert_array_equal(
        baseline.cache.features, extract_features(m, tokens, split, (3, 2)).features
    )


def test_saved_probe_replay_uses_frozen_normalization() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    features = np.arange(6 * 97, dtype=np.float32).reshape(6, 97) / np.float32(100)
    snapshot, cache = features.copy(), FeatureCache(features, labels)
    fitted = fit_probe(
        cache, cache, cache, ProbeConfig(vocab_size=3, updates=2, batch_size=2)
    )
    split, p = StorySplit(offsets=(0, 4, 8), sha256=("a", "b")), fitted.parameters
    scores = replay_probe(cache, split, p, batch_size=2)
    x = (torch.tensor(features) - torch.tensor(p.mean)) / torch.tensor(p.std)
    losses = cross_entropy(
        x @ torch.tensor(p.weight) + torch.tensor(p.bias),
        torch.tensor(labels),
        reduction="none",
    )
    for i, score in enumerate(scores.stories):
        assert score.loss_sum == float(losses[i * 3 : (i + 1) * 3].double().sum())
        assert score.count == 3
    assert scores.total.nll == pytest.approx(fitted.test.nll, abs=1e-7)
    assert scores.total.accuracy == fitted.test.accuracy
    assert scores == replay_probe(cache, split, p, batch_size=2)
    np.testing.assert_array_equal(features, snapshot)
    with pytest.raises(ValueError, match="normalization"):
        _ = replay_probe(cache, split, p.model_copy(update={"std": (0.0,) * 97}))


def test_evaluation_preserves_source_and_rng(
    learner: ARLearner, arm: CenteredOutgoing
) -> None:
    m = learner.model
    torch.autograd.backward(m.forward(torch.tensor([[0, 1]])).sum())
    params = tuple(m.parameters())
    tensors = (*params, *m.buffers(), *(p.grad for p in params if p.grad is not None))
    saved = tuple(t.detach().clone() for t in tensors)
    modes = tuple(module.training for module in m.modules())
    rng, private = torch.get_rng_state().clone(), learner.window_rng.get_state().clone()
    progress = learner.updates, tuple(learner.trace)
    tokens = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    original = tokens.copy()
    split = StorySplit(offsets=(0, 3, 5), sha256=("a", "b"))
    _ = evaluate_intervention(m, tokens, split, intervention=arm)
    params = tuple(m.parameters())
    after = (*params, *m.buffers(), *(p.grad for p in params if p.grad is not None))
    assert all(torch.equal(a, b) for a, b in zip(after, saved, strict=True))
    assert modes == tuple(module.training for module in m.modules())
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(private, learner.window_rng.get_state())
    assert progress == (learner.updates, tuple(learner.trace))
    np.testing.assert_array_equal(tokens, original)


@pytest.mark.parametrize("indices", [(0, 0), (-1,), (4,)])
def test_invalid_group_is_rejected(indices: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="indices"):
        _ = CenteredOutgoing(indices, (0.0,) * 4)


def test_invalid_mean_is_rejected(learner: ARLearner) -> None:
    with pytest.raises(ValueError, match="finite"):
        _ = CenteredOutgoing((0,), (float("nan"),) * 4)
    bad = CenteredOutgoing((0,), (0.0,))
    with pytest.raises(ValueError, match="mean"):
        _ = learner.model.selected_states(
            torch.tensor([[0, 1]]), torch.tensor([0]), intervention=bad
        )


def test_replay_validates_boundaries_and_shapes(learner: ARLearner) -> None:
    cache = FeatureCache(
        np.zeros((2, 2), dtype=np.float32), np.array([0, 1], dtype=np.int64)
    )
    split = StorySplit(offsets=(0, 3), sha256=("a",))
    head = (
        learner.model.readout.detach().numpy(),
        learner.model.output_bias.detach().numpy(),
    )
    with pytest.raises(ValueError, match="boundaries"):
        _ = replay_original_readout(
            cache, split.model_copy(update={"offsets": (0, 4)}), head
        )
    invalid = learner.model.readout.detach().double().numpy()
    with pytest.raises(ValueError, match="float32"):
        _ = replay_original_readout(cache, split, (invalid, head[1]))
    with pytest.raises(ValueError, match="batch_size"):
        _ = replay_original_readout(cache, split, head, batch_size=0)
