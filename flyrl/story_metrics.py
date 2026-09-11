"""All-position next-token metrics with recurrent state reset only between stories."""

from math import exp, log
from sys import float_info
from typing import Literal

import torch

from flyrl.ar_learning import ARLearner
from flyrl.language_data import IntVector
from flyrl.language_models import Settings
from flyrl.story_data import StoryCorpus, StorySplit


class StoryMetrics(Settings):
    """Mean natural-log loss and greedy accuracy over every within-story pair."""

    tokens: int
    stories: int
    nll: float
    perplexity: float | None
    accuracy: float
    denominator: Literal["within_story_next_token_pairs"] = (
        "within_story_next_token_pairs"
    )
    context_policy: Literal["full_story_recurrent_state"] = "full_story_recurrent_state"


class HeldoutMetrics(Settings):
    """Validation and test scores; neither selects updates or tokenizer merges."""

    valid: StoryMetrics
    test: StoryMetrics


@torch.no_grad()
def evaluate_story_split(
    learner: ARLearner,
    tokens: IntVector,
    split: StorySplit,
) -> StoryMetrics:
    """Score each target once, excluding first tokens and cross-story transitions.

    Computation is chunked only to bound logits storage; hidden state is preserved
    across chunks. Unlike training's truncated windows, evaluation sees the full
    causal prefix within each story, including short stories.
    """
    model = learner.model
    device = model.weight.device
    nll, correct, count = 0.0, 0, 0
    for a, b in zip(split.offsets[:-1], split.offsets[1:], strict=True):
        state = model.weight.new_zeros((model.nodes, 1))
        for start in range(a, b - 1, learner.config.context):
            stop = min(start + learner.config.context, b - 1)
            inputs = torch.tensor(tokens[start:stop], device=device)
            targets = torch.tensor(tokens[start + 1 : stop + 1], device=device)
            outputs: list[torch.Tensor] = []
            for token in inputs.unbind():
                logits, state = model.step(token.reshape(1), state)
                outputs.append(logits)
            stacked = torch.cat(outputs)
            nll += float(
                torch.nn.functional.cross_entropy(
                    stacked,
                    targets,
                    reduction="sum",
                ).item()
            )
            correct += int((stacked.argmax(dim=1) == targets).sum().item())
            count += stop - start
    mean = nll / count
    return StoryMetrics(
        tokens=count,
        stories=len(split.sha256),
        nll=mean,
        perplexity=exp(mean) if mean <= log(float_info.max) else None,
        accuracy=correct / count,
    )


def evaluate_heldout(learner: ARLearner, stories: StoryCorpus) -> HeldoutMetrics:
    """Evaluate both heldout splits without consuming the checkpoint training RNG."""
    return HeldoutMetrics(
        valid=evaluate_story_split(
            learner, stories.corpus.valid, stories.metadata.valid
        ),
        test=evaluate_story_split(learner, stories.corpus.test, stories.metadata.test),
    )
