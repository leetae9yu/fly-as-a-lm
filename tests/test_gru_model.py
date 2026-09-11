"""Ordinary GRU causality, parameter matching, private RNG and CPU continuation."""

from copy import deepcopy

import pytest
import torch

from flyrl.ar_config import ARConfig
from flyrl.ar_framework import optimizer_step, optimizer_tensors
from flyrl.gru_model import GRULM, UnsupportedGRUAblationError


def _update(model: GRULM, optimizer: torch.optim.AdamW, windows: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model.forward(windows[:, :-1])
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), windows[:, 1:].flatten()
    )
    torch.autograd.backward(loss)
    optimizer_step(optimizer)
    return float(loss.detach().item())


def test_shape_and_parameter_match_when_vocabulary_is_4096() -> None:
    # Given: the parameter-matched BPE vocabulary.
    config = ARConfig(alphabet_size=4096, tokenization="bpe")
    model = GRULM(config)
    tokens = torch.tensor([[0, 4095, 23], [3500, 12, 4]])
    # When: computing all-position logits.
    logits = model.forward(tokens)
    # Then: the ordinary untied architecture matches the requested budget.
    assert logits.shape == (2, 3, 4096)
    assert bool(torch.isfinite(logits).all())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 2271104
    assert model.embedding.weight.shape == (4096, 128)
    assert model.recurrent.input_size == 128
    assert model.recurrent.hidden_size == 320
    assert model.recurrent.num_layers == 1
    assert model.recurrent.batch_first
    assert not model.recurrent.bidirectional
    assert model.recurrent.dropout == 0
    assert model.readout.weight.shape == (4096, 320)
    assert model.readout.weight is not model.embedding.weight
    assert model.config is config


def test_weight_exposes_device_without_registering_an_alias() -> None:
    # Given: an ordinary CPU GRU.
    model = GRULM(ARConfig(alphabet_size=4))
    # When: inspecting the compatibility property and serialized parameters.
    state: dict[str, torch.Tensor] = {}
    _ = model.state_dict(destination=state)
    # Then: the device handle is the existing embedding, not duplicate state.
    assert model.weight is model.embedding.weight
    assert model.weight.device == torch.device("cpu")
    assert "weight" not in state
    assert len(state) == 7


@pytest.mark.parametrize("prefix_length", [1, 3, 5])
def test_prefix_equivalence_when_future_tokens_change(prefix_length: int) -> None:
    # Given: two windows with identical prefixes but different futures.
    model = GRULM(ARConfig(alphabet_size=4))
    tokens = torch.tensor([[0, 1, 2, 3, 0, 1], [3, 2, 1, 0, 3, 2]])
    changed = tokens.clone()
    changed[:, prefix_length:] = (changed[:, prefix_length:] + 1) % 4
    # When: evaluating both full windows and the isolated prefix.
    with torch.no_grad():
        full = model.forward(tokens)
        altered = model.forward(changed)
        prefix = model.forward(tokens[:, :prefix_length])
    # Then: neither future tokens nor earlier calls change prefix logits.
    assert torch.equal(full[:, :prefix_length], altered[:, :prefix_length])
    torch.testing.assert_close(full[:, :prefix_length], prefix, rtol=1e-6, atol=1e-7)


def test_recurrent_history_changes_logits_when_current_token_is_identical() -> None:
    # Given: distinct histories ending in the same token.
    model = GRULM(ARConfig(alphabet_size=4))
    tokens = torch.tensor([[0, 1, 2], [3, 1, 2]])
    # When: consuming both histories with identical zero initial state.
    logits = model.forward(tokens)
    # Then: prediction uses recurrent history, not just the current embedding.
    assert not torch.equal(logits[0, -1], logits[1, -1])


def test_zero_recurrent_raises_typed_error_when_requested() -> None:
    # Given: the anatomical ablation has no definition for an ordinary GRU.
    model = GRULM(ARConfig(alphabet_size=4))
    # When/Then: requesting it fails explicitly rather than scoring the real model.
    with pytest.raises(UnsupportedGRUAblationError):
        _ = model.forward(torch.tensor([[0, 1]]), zero_recurrent=True)


def test_global_cpu_rng_is_unchanged_when_constructing_and_running() -> None:
    # Given: the caller owns the global CPU random stream.
    before = torch.random.get_rng_state().clone()
    # When: initializing and evaluating in both training and evaluation modes.
    model = GRULM(ARConfig(alphabet_size=4, seed=91))
    tokens = torch.tensor([[0, 1, 2]])
    train_logits = model.forward(tokens)
    _ = model.eval()
    eval_logits = model.forward(tokens)
    # Then: initialization and forward consume no caller randomness or dropout.
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(train_logits, eval_logits)


def test_initialization_is_reproducible_when_seed_matches() -> None:
    # Given: matching and different private seeds.
    config = ARConfig(alphabet_size=4, seed=13)
    # When: independently constructing models.
    first, second = GRULM(config), GRULM(config)
    different = GRULM(ARConfig(alphabet_size=4, seed=14))
    # Then: every matching parameter is bitwise equal, but seeds affect weights.
    for name, parameter in first.named_parameters():
        assert torch.equal(parameter, second.get_parameter(name))
    assert not torch.equal(first.weight, different.weight)


def test_optimizer_learns_when_sequence_is_deterministic() -> None:
    # Given: a repeated four-token cycle with all next-token targets defined.
    model = GRULM(ARConfig(alphabet_size=4, seed=7))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, foreach=False)
    windows = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0]] * 2)
    initial = model.weight.detach().clone()
    # When: training through ordinary all-position likelihood and AdamW.
    losses = [_update(model, optimizer, windows) for _ in range(12)]
    predictions = model.forward(windows[:, :-1]).argmax(dim=-1)
    # Then: trainable embeddings change and the sequence is learned.
    assert losses[-1] < losses[0] * 0.1
    assert losses[-1] < 0.1
    assert torch.equal(predictions, windows[:, 1:])
    assert not torch.equal(initial, model.weight)


def test_cpu_continuation_is_exact_when_state_dicts_are_restored() -> None:
    # Given: an interrupted CPU trajectory with populated AdamW moments.
    config = ARConfig(alphabet_size=4, seed=23)
    original = GRULM(config)
    optimizer = torch.optim.AdamW(original.parameters(), lr=0.003, foreach=False)
    windows = torch.tensor([[0, 1, 2, 3, 0], [3, 2, 1, 0, 3]])
    for _ in range(2):
        _ = _update(original, optimizer, windows)
    state: dict[str, torch.Tensor] = {}
    _ = original.state_dict(destination=state)
    saved_model = deepcopy(state)
    saved_optimizer = deepcopy(optimizer.state_dict())
    resumed = GRULM(config)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.003, foreach=False)
    # When: restoring both state dictionaries and continuing identical batches.
    _ = resumed.load_state_dict(saved_model)
    resumed_optimizer.load_state_dict(saved_optimizer)
    expected = [_update(original, optimizer, windows) for _ in range(3)]
    actual = [_update(resumed, resumed_optimizer, windows) for _ in range(3)]
    # Then: losses, model weights, optimizer moments and predictions agree exactly.
    assert actual == expected
    for name, parameter in original.named_parameters():
        assert torch.equal(parameter, resumed.get_parameter(name))
    for left, right in zip(
        optimizer_tensors(optimizer).values(),
        optimizer_tensors(resumed_optimizer).values(),
        strict=True,
    ):
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key])
    assert torch.equal(original.forward(windows), resumed.forward(windows))


def test_cuda_request_fails_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: CUDA cannot run on this execution environment.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # When/Then: a CUDA request is rejected instead of silently using CPU.
    with pytest.raises(ValueError, match="CUDA"):
        _ = GRULM(ARConfig(alphabet_size=4, device="cuda"))
