"""Deterministic, unfiltered quality panels from the anatomical recurrent model."""

import re
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import Field

from flyrl.ar_learning import ARLearner
from flyrl.bpe_data import BPECorpus
from flyrl.language_data import CorpusError
from flyrl.language_models import Settings


class QualityGenerationOptions(Settings):
    """Fixed continuation length and temperature/nucleus-only sampling choices."""

    length: int = Field(default=192, ge=0)
    temperature: float = Field(default=0.8, gt=0)
    nucleus_p: float = Field(default=0.9, gt=0, le=1)


@dataclass(frozen=True, slots=True)
class QualityGenerationRecord:
    """One published draw; token IDs exclude the prompt, decoded text includes it.

    Draw indices are zero-based within each mode. Greedy has draw zero and no
    seed; sampled seeds are supplied explicitly by the caller, not derived from
    learner or global RNG state.
    """

    prompt: str
    prompt_ids: tuple[int, ...]
    mode: Literal["greedy", "sampled"]
    draw: int
    seed: int | None
    token_ids: tuple[int, ...]
    text: str


def nucleus_probabilities(
    logits: torch.Tensor, options: QualityGenerationOptions
) -> torch.Tensor:
    """Return a renormalized one-dimensional distribution in original token order.

    Temperature precedes stable descending sorting (ties use token ID order).
    Keep exactly the smallest prefix reaching p; p=1 keeps the full vocabulary,
    even when cumulative floating-point rounding would otherwise drop its tail.
    """
    probabilities = (logits / options.temperature).softmax(dim=0)
    if options.nucleus_p == 1:
        return probabilities
    ordered, indices = probabilities.sort(descending=True, stable=True)
    preceding_mass = torch.cat((ordered.new_zeros(1), ordered.cumsum(dim=0)[:-1]))
    retained = ordered * (preceding_mass < options.nucleus_p)
    retained = retained / retained.sum()
    return torch.zeros_like(probabilities).scatter(0, indices, retained)


@torch.no_grad()
def generate_quality_panel(
    learner: ARLearner,
    corpus: BPECorpus,
    prompts: tuple[str, ...],
    sampled_seeds: tuple[tuple[int, ...], ...],
    options: QualityGenerationOptions,
) -> tuple[QualityGenerationRecord, ...]:
    """Generate every fixed prompt, greedy first then all caller-seeded draws.

    Each prompt has one seed tuple, whose length determines its sampled draws.
    Empty seed tuples request greedy only. Each continuation starts at zero state,
    consumes the entire prompt, and keeps recurrent state while feeding back only
    generated tokens, regardless of the learner's windowed generation setting.
    Parameters, gradients, training progress and global RNGs are unchanged.
    """
    if learner.config.alphabet_size != len(corpus.vocabulary):
        raise CorpusError(reason="Generation corpus and model vocabulary disagree")
    if len(prompts) != len(sampled_seeds) or any(
        not -(2**63) <= seed < 2**64 for seeds in sampled_seeds for seed in seeds
    ):
        message = "Generation needs one valid torch seed tuple per prompt"
        raise ValueError(message)
    encoded = tuple(tuple(corpus.encode(prompt)) for prompt in prompts)
    if any(not ids for ids in encoded):
        raise CorpusError(reason="Generation requires a nonempty prompt")
    records: list[QualityGenerationRecord] = []
    for prompt, prompt_ids, seeds in zip(prompts, encoded, sampled_seeds, strict=True):
        for position, seed in enumerate((None, *seeds)):
            token_ids = _continue(learner, prompt_ids, seed, options)
            records.append(
                QualityGenerationRecord(
                    prompt=prompt,
                    prompt_ids=prompt_ids,
                    mode="greedy" if seed is None else "sampled",
                    draw=max(0, position - 1),
                    seed=seed,
                    token_ids=token_ids,
                    text=corpus.decode([*prompt_ids, *token_ids]),
                )
            )
    return tuple(records)


def _continue(
    learner: ARLearner,
    prompt_ids: tuple[int, ...],
    seed: int | None,
    options: QualityGenerationOptions,
) -> tuple[int, ...]:
    """Keep one local circuit state and one private device-local RNG per draw."""
    model = learner.model
    device = model.weight.device
    rng = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    state = model.weight.new_zeros((model.nodes, 1))
    logits = model.output_bias[None]
    for token_id in prompt_ids:
        logits, state = model.step(torch.tensor([token_id], device=device), state)
    generated: list[int] = []
    for _ in range(options.length):
        token = (
            logits.argmax(dim=1)
            if rng is None
            else torch.multinomial(
                nucleus_probabilities(logits[0], options), 1, generator=rng
            )
        )
        generated.append(int(token.item()))
        logits, state = model.step(token, state)
    return tuple(generated)


def has_repeated_word_block(text: str) -> bool:
    """Detect any lowercase 1-10-word block occurring at least three times in a row."""
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    return any(
        words[start : start + width]
        == words[start + width : start + 2 * width]
        == words[start + 2 * width : start + 3 * width]
        for width in range(1, 11)
        for start in range(len(words) - 3 * width + 1)
    )
