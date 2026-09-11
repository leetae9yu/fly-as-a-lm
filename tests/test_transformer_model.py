"""Causality, matched capacity, deterministic learning and isolated initialization."""

import pytest
import torch

from flyrl import transformer_model as transformer
from flyrl.ar_config import ARConfig
from flyrl.ar_framework import optimizer_step


def test_vocabulary_shape_and_parameter_budget() -> None:
    # Given: the parameter-matched BPE configuration.
    config = ARConfig(alphabet_size=4096, tokenization="bpe", context=32)
    # When: constructing and executing a complete context.
    model = transformer.TransformerLM(config)
    logits = model.forward(torch.zeros((2, 32), dtype=torch.long))
    # Then: every position covers the vocabulary at the fixed capacity.
    assert logits.shape == (2, 32, 4096)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_248_096
    assert model.config is config
    assert model.weight.shape == (4096, 160)
    assert model.weight.device == torch.device("cpu")
    assert "weight" not in model.state_dict()
    assert model.output.weight is not model.weight


@pytest.mark.parametrize("prefix", [1, 3, 7])
def test_future_tokens_cannot_change_prefix_logits(prefix: int) -> None:
    # Given: a shared prefix with different tokens at every future position.
    model = transformer.TransformerLM(ARConfig(alphabet_size=4, context=8))
    tokens = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3], [3, 2, 1, 0, 3, 2, 1, 0]])
    changed = tokens.clone()
    changed[:, prefix:] = (changed[:, prefix:] + 1) % 4
    # When: predicting the full windows in training mode.
    original = model.forward(tokens)
    altered = model.forward(changed)
    # Then: future edits cannot affect any earlier prediction.
    torch.testing.assert_close(
        original[:, :prefix], altered[:, :prefix], rtol=0, atol=0
    )
    assert not torch.equal(original[:, prefix:], altered[:, prefix:])


@pytest.mark.parametrize("prefix", [1, 3, 7])
def test_prefix_execution_matches_full_window(prefix: int) -> None:
    # Given: one context and its independently executed prefix.
    model = transformer.TransformerLM(ARConfig(alphabet_size=4, context=8))
    tokens = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    # When: executing different lengths with the same positional origin.
    full = model.forward(tokens)
    shortened = model.forward(tokens[:, :prefix])
    # Then: causal attention produces the same prefix up to reduction rounding.
    torch.testing.assert_close(shortened, full[:, :prefix], rtol=1e-5, atol=1e-6)


def test_initialization_preserves_cpu_rng_and_is_seeded() -> None:
    # Given: a CPU RNG state that model initialization must not consume.
    config = ARConfig(alphabet_size=4, seed=17)
    before = torch.random.get_rng_state().clone()
    # When: constructing same-seed and different-seed models.
    first = transformer.TransformerLM(config)
    second = transformer.TransformerLM(config)
    different = transformer.TransformerLM(config.model_copy(update={"seed": 18}))
    # Then: initialization is private, repeatable and seed-sensitive.
    assert torch.equal(before, torch.random.get_rng_state())
    for name, parameter in first.named_parameters():
        assert torch.equal(parameter, second.get_parameter(name))
    assert not torch.equal(first.weight, different.weight)


def test_initialization_never_seeds_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: an observable CUDA seeding boundary, including on CPU-only builds.
    seeded: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seeded.append)
    # When: constructing the CPU model.
    _ = transformer.TransformerLM(ARConfig(alphabet_size=2))
    # Then: initialization never invokes torch.manual_seed's CUDA side effect.
    assert seeded == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_construction_preserves_all_rng_states() -> None:
    # Given: initialized CUDA generators and the CPU generator.
    cuda_before = torch.cuda.get_rng_state_all()
    cpu_before = torch.random.get_rng_state().clone()
    # When: constructing on CPU and transferring to the requested GPU.
    model = transformer.TransformerLM(ARConfig(alphabet_size=2, device="cuda"))
    # Then: neither generator family changes, and execution stays on the GPU.
    assert model.weight.device.type == "cuda"
    assert model.forward(torch.zeros((1, 2), dtype=torch.long, device="cuda")).is_cuda
    assert torch.equal(cpu_before, torch.random.get_rng_state())
    for before, after in zip(cuda_before, torch.cuda.get_rng_state_all(), strict=True):
        assert torch.equal(before, after)


def test_training_forward_is_deterministic_and_preserves_rng() -> None:
    # Given: a model left in training mode and a fixed input.
    model = transformer.TransformerLM(ARConfig(alphabet_size=2, context=4))
    tokens = torch.tensor([[0, 1, 0, 1]])
    before = torch.random.get_rng_state().clone()
    # When: executing training-mode predictions repeatedly.
    first, second = model.forward(tokens), model.forward(tokens)
    # Then: dropout introduces neither noise nor global RNG consumption.
    assert model.training
    assert torch.equal(first, second)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("context", [1, 4])
def test_context_limit_accepts_boundary_and_rejects_overflow(context: int) -> None:
    # Given: a model with exactly the configured learned positions.
    model = transformer.TransformerLM(ARConfig(alphabet_size=2, context=context))
    # When: executing the boundary length, then one position beyond it.
    boundary = model.forward(torch.zeros((1, context), dtype=torch.long))
    with pytest.raises(transformer.ContextLengthError) as caught:
        _ = model.forward(torch.zeros((1, context + 1), dtype=torch.long))
    # Then: the limit is inclusive and the error exposes structured lengths.
    assert boundary.shape == (1, context, 2)
    assert caught.value.length == context + 1
    assert caught.value.context == context


def test_recurrent_ablation_is_rejected() -> None:
    # Given: a Transformer, where anatomical recurrence ablation has no meaning.
    model = transformer.TransformerLM(ARConfig(alphabet_size=2))
    # When/Then: requesting that ablation raises its own typed error.
    with pytest.raises(transformer.RecurrentAblationError):
        _ = model.forward(torch.tensor([[0]]), zero_recurrent=True)


def test_deterministic_next_token_sequence_learning() -> None:
    # Given: identical models and a deterministic repeating next-token task.
    config = ARConfig(alphabet_size=4, context=8, seed=9)
    models = [transformer.TransformerLM(config), transformer.TransformerLM(config)]
    windows = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0], [2, 3, 0, 1, 2, 3, 0, 1, 2]])
    targets = windows[:, 1:].reshape(-1)
    initial = torch.nn.functional.cross_entropy(
        models[0].forward(windows[:, :-1]).reshape(-1, 4), targets
    ).item()
    final_losses: list[float] = []
    # When: optimizing each model on exactly the same batches and update count.
    for model in models:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, foreach=False)
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward(windows[:, :-1])
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 4), targets)
            torch.autograd.backward(loss)
            optimizer_step(optimizer)
        final = model.forward(windows[:, :-1]).reshape(-1, 4)
        final_losses.append(torch.nn.functional.cross_entropy(final, targets).item())
        assert torch.equal(final.argmax(dim=-1), targets)
    # Then: real gradient updates learn the sequence identically, well below chance.
    assert final_losses[0] < min(initial, 0.1)
    assert final_losses[0] == final_losses[1]
    for name, parameter in models[0].named_parameters():
        assert torch.equal(parameter, models[1].get_parameter(name))
