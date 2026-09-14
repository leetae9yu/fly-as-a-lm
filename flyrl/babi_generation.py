"""All-vocabulary greedy decoding with local, zero-initialized anatomical state."""

from collections import defaultdict
from typing import Annotated, Final, Self

import torch
from pydantic import Field, model_validator

from flyrl.ar_model import ConnectomeLM
from flyrl.babi_learning import TERMINATOR, VOCABULARY, ExampleId, TokenId, TokenTuple
from flyrl.language_data import CorpusError
from flyrl.language_models import Settings

MAX_GENERATED: Final = 8


class BabiPrompt(Settings):
    """Generation input deliberately has no gold token or answer-label field."""

    example_id: ExampleId
    prompt_ids: TokenTuple


class Generation(Settings):
    """Raw generated IDs in order; termination is observed, never inferred from text."""

    example_id: ExampleId
    generated_ids: Annotated[
        tuple[TokenId, ...], Field(min_length=1, max_length=MAX_GENERATED)
    ]

    @model_validator(mode="after")
    def stopped_at_terminator(self) -> Self:
        """Reject evidence containing any token generated after the first LF."""
        if TERMINATOR in self.generated_ids[:-1]:
            raise CorpusError(reason="Generated tokens after terminator")
        return self

    @property
    def terminated(self) -> bool:
        """Report whether an actual LF token occurred within the generation budget."""
        return TERMINATOR in self.generated_ids


@torch.no_grad()
def greedy_generate(
    model: ConnectomeLM, prompts: tuple[BabiPrompt, ...]
) -> tuple[Generation, ...]:
    """Consume only prompts, then at most eight unrestricted greedy tokens.

    Equal-length prompts share batched step calls, with a fresh zero state per
    example. Finished rows are removed, so neither LF nor any post-LF token is
    ever fed back. Stateful continuation is not truncated at the training context.
    No model mode, optimizer, gradient, or random generator is changed.
    """
    if not prompts or model.config.alphabet_size != VOCABULARY:
        raise CorpusError(reason="Generation requires prompts and 4096 logits")
    groups: defaultdict[int, list[int]] = defaultdict(list)
    for index, prompt in enumerate(prompts):
        if len(prompt.prompt_ids) > model.config.context:
            raise CorpusError(reason=f"Prompt context overflow: {prompt.example_id}")
        groups[len(prompt.prompt_ids)].append(index)
    generated: list[tuple[int, ...]] = [() for _ in prompts]
    for indices in groups.values():
        for start in range(0, len(indices), model.config.batch_size):
            selected = indices[start : start + model.config.batch_size]
            result = _generate_group(model, tuple(prompts[i] for i in selected))
            for index, tokens in zip(selected, result, strict=True):
                generated[index] = tokens
    return tuple(
        Generation(example_id=prompt.example_id, generated_ids=tuple(tokens))
        for prompt, tokens in zip(prompts, generated, strict=True)
    )


def _generate_group(
    model: ConnectomeLM,
    prompts: tuple[BabiPrompt, ...],
) -> tuple[tuple[int, ...], ...]:
    """Advance one same-length batch, removing finished rows from local state."""
    device = model.weight.device
    inputs = torch.tensor([prompt.prompt_ids for prompt in prompts], device=device)
    state = model.weight.new_zeros((model.nodes, len(prompts)))
    logits = model.output_bias[None]
    for token in inputs.unbind(dim=1):
        logits, state = model.step(token, state)
    generated: list[list[int]] = [[] for _ in prompts]
    active = list(range(len(prompts)))
    for iteration in range(MAX_GENERATED):
        if not bool(torch.isfinite(logits).all()):
            raise CorpusError(reason="Nonfinite generation logits")
        tokens = logits.argmax(dim=1)
        remaining: list[int] = []
        positions: list[int] = []
        for position, (index, token) in enumerate(zip(active, tokens, strict=True)):
            token_id = int(token.item())
            generated[index].append(token_id)
            if token_id != TERMINATOR:
                remaining.append(index)
                positions.append(position)
        if not remaining or iteration == MAX_GENERATED - 1:
            break
        active = remaining
        logits, state = model.step(tokens[positions], state[:, positions])
    return tuple(tuple(tokens) for tokens in generated)
