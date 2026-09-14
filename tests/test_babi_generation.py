"""Unrestricted greedy decoding, using the unchanged ConnectomeLM recurrence."""

import pytest
import torch
from pydantic import ValidationError

from flyrl.babi_generation import BabiPrompt, greedy_generate
from flyrl.language_data import CorpusError
from tests.test_babi_learning import make_learner


def test_step_matches_full_causal_forward() -> None:
    # Given: a real model, including a final short batch.
    model = make_learner().model
    tokens = torch.tensor([[10, 11, 12, 20], [13, 14, 21, 22]])
    state = model.weight.new_zeros((model.nodes, 2))
    # When: online recurrence consumes the same causal sequence as forward.
    online: list[torch.Tensor] = []
    for token in tokens.unbind(dim=1):
        logits, state = model.step(token, state)
        online.append(logits)
    # Then: all 4096 logits at every position are equivalent.
    torch.testing.assert_close(torch.stack(online, dim=1), model.forward(tokens))


def test_generation_matches_prompt_only_forward_and_resets_state() -> None:
    # Given: unequal prompt lengths, no gold IDs anywhere in the decoder API.
    model = make_learner(context=3).model
    prompts = (
        BabiPrompt(example_id="a", prompt_ids=(10, 11, 12)),
        BabiPrompt(example_id="b", prompt_ids=(13,)),
        BabiPrompt(example_id="c", prompt_ids=(14, 15, 16)),
    )
    expected: list[tuple[int, ...]] = []
    for prompt in prompts:
        generated: list[int] = []
        for _ in range(8):
            logits = model.forward(torch.tensor([(*prompt.prompt_ids, *generated)]))
            token = int(logits[0, -1].argmax().item())
            generated.append(token)
            if token == 199:
                break
        expected.append(tuple(generated))
    # When: batched stateful decoding continues beyond the training context.
    actual = greedy_generate(model, prompts)
    # Then: tokens agree in source order, without cross-example state.
    assert tuple(record.generated_ids for record in actual) == tuple(expected)
    assert tuple(record.example_id for record in actual) == ("a", "b", "c")
    assert greedy_generate(model, prompts[::-1]) == actual[::-1]


@pytest.mark.parametrize("token", [0, 4095, 199])
def test_unrestricted_wrong_tokens_and_terminator(token: int) -> None:
    # Given: an actual model whose highest output is outside the answer vocabulary.
    learner = make_learner(context=1)
    model = learner.model
    with torch.no_grad():
        _ = model.readout.zero_()
        _ = model.output_bias.zero_()
        model.output_bias[token] = 100
    prompt = BabiPrompt(example_id="wrong", prompt_ids=(10,))
    rng = learner.window_rng.get_state().clone()
    # When: greedy decoding sees the entire 4096-way output.
    record = greedy_generate(model, (prompt,))[0]
    # Then: wrong tokens are neither masked nor replaced; only LF terminates.
    assert record.generated_ids == ((199,) if token == 199 else (token,) * 8)
    assert record.terminated == (token == 199)
    assert torch.equal(rng, learner.window_rng.get_state())


def test_generation_rejects_empty_invalid_and_overflow_prompts() -> None:
    # Given: a small configured context.
    model = make_learner(context=2).model
    # When/Then: direct decoder boundaries reject invalid input without truncation.
    with pytest.raises(CorpusError):
        _ = greedy_generate(model, ())
    with pytest.raises(ValidationError):
        _ = BabiPrompt(example_id="x", prompt_ids=())
    with pytest.raises(ValidationError):
        _ = BabiPrompt(example_id="x", prompt_ids=(4096,))
    with pytest.raises(CorpusError):
        _ = greedy_generate(model, (BabiPrompt(example_id="x", prompt_ids=(1, 2, 3)),))
