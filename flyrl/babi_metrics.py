"""Literal exact-answer parsing and additive, equal-question NLL evidence."""

from collections.abc import Callable
from math import fsum
from typing import Annotated, Final, TypeAlias

import torch
from pydantic import Field

from flyrl.ar_model import ConnectomeLM
from flyrl.babi_generation import Generation
from flyrl.babi_learning import (
    TERMINATOR,
    BabiExample,
    ExampleId,
    TokenId,
    answer_token_losses,
    make_batch,
)
from flyrl.language_data import CorpusError
from flyrl.language_models import Settings

LOCATIONS: Final = frozenset(
    {"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"}
)
Decode: TypeAlias = Callable[[tuple[int, ...]], str]


class QuestionNLL(Settings):
    """One question's additive natural-log token loss and supervised-token count."""

    example_id: ExampleId
    loss_sum: Annotated[float, Field(ge=0)]
    count: Annotated[int, Field(ge=1)]


class AnswerNLL(Settings):
    """Ordered question evidence; the primary NLL weights questions equally."""

    per_question: Annotated[tuple[QuestionNLL, ...], Field(min_length=1)]

    @property
    def loss_sum(self) -> float:
        """Pool additive token losses, independently of the question mean."""
        return fsum(question.loss_sum for question in self.per_question)

    @property
    def count(self) -> int:
        """Count only answer tokens, including every terminator."""
        return sum(question.count for question in self.per_question)

    @property
    def nll(self) -> float:
        """Return mean question NLL, not a token-weighted corpus average."""
        return fsum(q.loss_sum / q.count for q in self.per_question) / len(
            self.per_question
        )


class Prediction(Settings):
    """Published machine evidence without any output repair or normalization."""

    example_id: ExampleId
    generated_ids: tuple[TokenId, ...]
    terminated: bool
    decoded_answer: str
    gold_answer: str

    @property
    def correct(self) -> bool:
        """Require observed termination and exactly one leading space plus gold."""
        return self.terminated and self.decoded_answer == f" {self.gold_answer}"


def validate_answer_encoding(example: BabiExample, gold: str, decode: Decode) -> None:
    """Bind supplied tokens to the six legal decoded labels before any training."""
    if gold not in LOCATIONS:
        raise CorpusError(reason="Malformed gold answer")
    expected_count = 3 if gold == "hallway" else 2
    if (
        len(example.answer_ids) != expected_count
        or decode(example.answer_ids) != f" {gold}\n"
    ):
        raise CorpusError(reason=f"Changed answer encoding: {example.example_id}")


def score_prediction(generation: Generation, gold: str, decode: Decode) -> Prediction:
    """Decode only pre-terminator tokens; malformed outputs remain wrong answers."""
    if gold not in LOCATIONS:
        raise CorpusError(reason="Malformed gold answer")
    tokens = generation.generated_ids
    if generation.terminated:
        tokens = tokens[: tokens.index(TERMINATOR)]
    return Prediction(
        example_id=generation.example_id,
        generated_ids=generation.generated_ids,
        terminated=generation.terminated,
        decoded_answer=decode(tokens),
        gold_answer=gold,
    )


@torch.no_grad()
def evaluate_nll(model: ConnectomeLM, examples: tuple[BabiExample, ...]) -> AnswerNLL:
    """Teacher-force answers without touching model mode, gradients, Adam or RNG."""
    if not examples:
        raise CorpusError(reason="Evaluation requires at least one question")
    evidence: list[QuestionNLL] = []
    for start in range(0, len(examples), model.config.batch_size):
        batch = make_batch(
            examples[start : start + model.config.batch_size], model.config.context
        )
        inputs = torch.tensor(batch.inputs, device=model.weight.device)
        losses = answer_token_losses(model.forward(inputs), batch).double().sum(dim=1)
        evidence.extend(
            QuestionNLL(example_id=identity, loss_sum=float(loss.item()), count=count)
            for identity, loss, count in zip(
                batch.example_ids, losses, batch.answer_counts, strict=True
            )
        )
    return AnswerNLL(per_question=tuple(evidence))
