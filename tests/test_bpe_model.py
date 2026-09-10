"""Large token vocabularies must not change the anatomical input-port budget."""

from math import exp, log
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from flyrl.ar_checkpoint import load_checkpoint, save_checkpoint
from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.connectome import Graph


@pytest.fixture
def ring() -> Graph:
    source = np.arange(512, dtype=np.int64)
    return Graph(
        tuple(f"synthetic:{i}" for i in source),
        source,
        (source + 1) % source.size,
        np.ones(source.size, dtype=np.float64),
        "synthetic ring, not anatomical data",
    )


@pytest.mark.parametrize("size", [4096, 8192])
def test_bpe_vocabulary_keeps_fixed_sensory_budget(ring: Graph, size: int) -> None:
    # Given: a token vocabulary much larger than the character alphabet.
    config = ARConfig.model_validate({"alphabet_size": size, "tokenization": "bpe"})
    # When: constructing the same anatomical circuit for each vocabulary.
    learner = ARLearner(ring, config)
    # Then: input ports stay fixed while codes and the decoder cover the vocabulary.
    assert learner.model.sensory.numel() == 192
    assert learner.model.codes.shape == (size, 192)
    assert learner.model.readout.shape == (256, size)
    assert not bool(
        (learner.model.sensory[:, None] == learner.model.ports[None, :]).any()
    )


def test_character_configuration_retains_its_vocabulary_limit() -> None:
    # Given/When: a large vocabulary mislabeled as the legacy character format.
    with pytest.raises(ValidationError):
        _ = ARConfig(alphabet_size=4096)


def test_bpe_learning_reports_token_metrics(ring: Graph) -> None:
    # Given: high token IDs that cannot be represented by the old 48-way head.
    config = ARConfig.model_validate(
        {
            "alphabet_size": 4096,
            "tokenization": "bpe",
            "context": 3,
            "batch_size": 2,
            "eval_windows": 4,
        }
    )
    learner = ARLearner(ring, config)
    tokens = np.array([3500, 4000] * 20, dtype=np.int64)
    before = learner.evaluate(tokens)
    # When: optimizing strict next-token likelihood.
    learner.train(tokens, 3)
    after = learner.evaluate(tokens)
    # Then: likelihood improves and token units are never mislabeled as characters.
    assert after.nll < before.nll
    assert after.bits_per_character is None
    assert after.bits_per_token == pytest.approx(after.nll / log(2))
    assert after.perplexity == pytest.approx(exp(after.nll))


def test_bpe_update_boundary_resumes_exactly(ring: Graph, tmp_path: Path) -> None:
    # Given: identical full and interrupted token-model trajectories.
    config = ARConfig.model_validate(
        {"alphabet_size": 4096, "tokenization": "bpe", "context": 3, "batch_size": 2}
    )
    tokens = np.array([3500, 4000] * 20, dtype=np.int64)
    full, partial = ARLearner(ring, config), ARLearner(ring, config)
    full.train(tokens, 3)
    partial.train(tokens, 1)
    checkpoint = tmp_path / "checkpoint.npz"
    save_checkpoint(partial, checkpoint, "tokenizer-and-corpus-fingerprint")
    resumed = ARLearner(ring, config)
    # When: restoring and continuing the interrupted copy.
    load_checkpoint(resumed, checkpoint, "tokenizer-and-corpus-fingerprint")
    resumed.train(tokens, 2)
    # Then: optimizer continuation produces the identical final model and trace.
    assert full.trace == resumed.trace
    assert torch.equal(full.window_rng.get_state(), resumed.window_rng.get_state())
    for name, parameter in full.model.named_parameters():
        assert torch.equal(parameter, resumed.model.get_parameter(name))


def test_large_token_loss_remains_reportable(ring: Graph) -> None:
    # Given: finite logits whose token perplexity exceeds float64's range.
    learner = ARLearner(
        ring,
        ARConfig(alphabet_size=4096, tokenization="bpe", context=3, eval_windows=2),
    )
    with torch.no_grad():
        _ = learner.model.readout.zero_()
        _ = learner.model.output_bias.zero_()
        learner.model.output_bias[0] = 1000
    # When: scoring deliberately improbable, but valid, target tokens.
    metrics = learner.evaluate(np.ones(12, dtype=np.int64))
    # Then: retain the finite log loss instead of failing during exponentiation.
    assert metrics.nll == 1000
    assert metrics.bits_per_token == pytest.approx(1000 / log(2))
    assert metrics.perplexity is None
