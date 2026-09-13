"""Frozen regional state alignment and reproducible equal-capacity probes."""

from math import log

import numpy as np
import pytest
import torch

from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.connectome import Graph
from flyrl.regional_probe import (
    extract_features,
    fit_probe,
)
from flyrl.regional_probe_types import (
    ExtractionConfig,
    FeatureCache,
    ProbeConfig,
    ProbeResult,
)
from flyrl.story_data import StorySplit


@pytest.fixture
def learner() -> ARLearner:
    source = np.arange(104, dtype=np.int64)
    graph = Graph(
        tuple(f"synthetic:{i}" for i in source),
        source,
        (source + 1) % source.size,
        np.ones(source.size, dtype=np.float64),
        "synthetic probe fixture, not anatomy",
    )
    return ARLearner(graph, ARConfig(alphabet_size=3, context=2, batch_size=2))


@pytest.fixture
def signal() -> FeatureCache:
    labels = np.array([0, 1, 1, 0] * 12, dtype=np.int64)
    features = np.full((labels.size, 97), 0.75, dtype=np.float32)
    features[:, 0] = 2 * labels - 1
    return FeatureCache(features, labels)


@pytest.mark.parametrize(("chunk_size", "story_batch_size"), [(1, 1), (2, 2), (8, 3)])
def test_alignment_padding_pair_order_and_reset(
    learner: ARLearner, chunk_size: int, story_batch_size: int
) -> None:
    tokens = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2, 2], dtype=np.int64)
    split = StorySplit(offsets=(0, 5, 7, 10), sha256=("a", "b", "c"))
    indices = tuple(range(100, 3, -1))
    expected: list[torch.Tensor] = []
    with torch.no_grad():
        for a, b in zip(split.offsets[:-1], split.offsets[1:], strict=True):
            state = torch.zeros((learner.model.nodes, 1))
            for token in tokens[a : b - 1].flat:
                _, state = learner.model.step(torch.tensor([token]), state)
                expected.append(state[list(indices), 0].clone())
    cache = extract_features(
        learner.model,
        tokens,
        split,
        indices,
        config=ExtractionConfig(
            chunk_size=chunk_size, story_batch_size=story_batch_size
        ),
    )
    assert cache.features.dtype == np.float32
    assert cache.labels.dtype == np.int64
    assert not cache.features.flags.writeable
    assert not cache.labels.flags.writeable
    np.testing.assert_array_equal(cache.features, torch.stack(expected).numpy())
    np.testing.assert_array_equal(cache.labels, [1, 2, 0, 1, 1, 2, 2])
    assert cache.features.shape == (7, 97)


def test_extraction_and_fitting_preserve_source(learner: ARLearner) -> None:
    tokens = np.array([0, 1, 2] * 5, dtype=np.int64)
    learner.train(tokens, 1)
    values = {
        name: value.detach().clone()
        for name, value in (
            *learner.model.named_parameters(),
            *learner.model.named_buffers(),
        )
    }
    gradients = tuple(
        None if value.grad is None else value.grad.clone()
        for value in learner.model.parameters()
    )
    modes = tuple(module.training for module in learner.model.modules())
    rng = torch.get_rng_state().clone()
    private_rng = learner.window_rng.get_state().clone()
    progress = learner.updates, tuple(learner.trace)
    split = StorySplit(offsets=(0, 8, 15), sha256=("a", "b"))
    cache = extract_features(learner.model, tokens, split, tuple(range(97)))
    _ = fit_probe(cache, cache, cache, ProbeConfig(vocab_size=3, updates=3))
    for name, value in (
        *learner.model.named_parameters(),
        *learner.model.named_buffers(),
    ):
        assert torch.equal(value, values[name])
    for value, before in zip(learner.model.parameters(), gradients, strict=True):
        if before is None:
            assert value.grad is None
        else:
            assert value.grad is not None
            assert torch.equal(value.grad, before)
    assert modes == tuple(module.training for module in learner.model.modules())
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(private_rng, learner.window_rng.get_state())
    assert progress == (learner.updates, tuple(learner.trace))


def test_deterministic_fit_learns_signal_and_serializes(signal: FeatureCache) -> None:
    config = ProbeConfig(
        vocab_size=2,
        updates=60,
        batch_size=16,
        learning_rate=0.1,
        min_learning_rate=0.02,
        device="cpu",
    )
    global_rng = torch.get_rng_state().clone()
    first = fit_probe(signal, signal, signal, config)
    second = fit_probe(signal, signal, signal, config)
    assert first == second
    assert torch.equal(global_rng, torch.get_rng_state())
    assert first.valid.nll < 0.15 < log(2)
    assert first.test.accuracy == 1
    assert first.test.perplexity == pytest.approx(np.exp(first.test.nll))
    assert first.parameter_count == (97 + 1) * 2
    assert len(first.parameters.weight) == 97
    assert len(first.parameters.bias) == 2
    assert len(first.trace) == config.updates
    assert first.trace[0].learning_rate == config.learning_rate
    assert 0.02 < first.trace[-1].learning_rate < first.trace[0].learning_rate
    assert first == ProbeResult.model_validate_json(first.model_dump_json())
    different = fit_probe(signal, signal, signal, config.model_copy(update={"seed": 9}))
    assert different.parameters != first.parameters


def test_train_only_normalization_unigram_and_feature_evidence(
    signal: FeatureCache,
) -> None:
    train = FeatureCache(signal.features[:3].copy(), signal.labels[:3].copy())
    heldout = FeatureCache(signal.features + 10, signal.labels.copy())
    config = ProbeConfig(vocab_size=3, updates=0, std_floor=0.02)
    result = fit_probe(train, heldout, heldout, config)
    expected_bias = np.log(np.array([1.5, 2.5, 0.5]) / 4.5)
    np.testing.assert_allclose(result.parameters.bias, expected_bias, rtol=1e-6)
    np.testing.assert_array_equal(result.parameters.weight, np.zeros((97, 3)))
    np.testing.assert_allclose(result.parameters.mean, train.features.mean(axis=0))
    np.testing.assert_allclose(
        result.parameters.std, np.maximum(train.features.std(axis=0), 0.02)
    )
    assert result.valid.nll == pytest.approx(
        float(-expected_bias[signal.labels].mean())
    )
    assert result.valid.tokens == signal.labels.size
    assert result.valid.accuracy == 0.5
    assert result.features[0].saturation_fraction == pytest.approx(1 / 97)
    assert result.features[0].variance[0] == pytest.approx(8 / 9)
    assert result.features[0].variance[1] == 0
    assert result.features[0].feature_sha256 != result.features[1].feature_sha256
    assert result.features[1] == result.features[2]
    assert len(result.features[0].label_sha256) == 64


def test_probe_rejects_wrong_capacity_and_invalid_targets(signal: FeatureCache) -> None:
    wrong = FeatureCache(signal.features[:, :2].copy(), signal.labels.copy())
    with pytest.raises(ValueError, match="97"):
        _ = fit_probe(wrong, signal, signal, ProbeConfig(vocab_size=2))
    invalid = FeatureCache(signal.features.copy(), signal.labels + 2)
    with pytest.raises(ValueError, match="vocabulary"):
        _ = fit_probe(signal, signal, invalid, ProbeConfig(vocab_size=2))


def test_extraction_rejects_invalid_boundaries_and_options(learner: ARLearner) -> None:
    tokens = np.array([0, 1, 2, 0], dtype=np.int64)
    split = StorySplit(offsets=(0, 2, 4), sha256=("a", "b"))
    with pytest.raises(ValueError, match="greater than"):
        _ = ExtractionConfig(chunk_size=0)
    with pytest.raises(ValueError, match="boundaries"):
        _ = extract_features(
            learner.model, tokens, split.model_copy(update={"offsets": (1, 4)}), (0,)
        )
