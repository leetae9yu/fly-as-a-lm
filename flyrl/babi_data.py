"""Strict Task 1 episodes, source-grounded questions, and immutable lexical IDs."""

import hashlib
import re
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar, Final, Literal, Self

import numpy as np
from pydantic import ConfigDict, Field, ValidationError, model_validator

from flyrl.bpe_tokenizer import BPETokenizer, restore_tokenizer
from flyrl.language_models import LanguageError, Settings

BabiError = LanguageError
CONTEXT: Final = 160
VOCABULARY: Final = 4096
TRAIN_EPISODES: Final = 2000
QUESTION_FIELDS: Final = 3
Location = Literal["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
Person = Literal["Mary", "John", "Daniel", "Sandra"]
Token = Annotated[int, Field(ge=0, lt=4096)]
Positive = Annotated[int, Field(gt=0)]
TOKENIZER_HASH: Final = (
    "f626d6a0653d63580a9aad06dd5dd19bd314514dc861d271254867dfb476d4ff"
)
ANSWER_IDS: Final[Mapping[str, tuple[int, ...]]] = MappingProxyType(
    {
        "bathroom": (2811, 199),
        "bedroom": (2683, 199),
        "garden": (851, 199),
        "hallway": (3885, 453, 199),
        "kitchen": (1198, 199),
        "office": (3175, 199),
    }
)
FACT_PATTERN: Final = (
    r"(Mary|John|Daniel|Sandra) (?:moved|went|went back|journeyed|travelled) "
    r"to the (bathroom|bedroom|garden|hallway|kitchen|office)\."
)
FACT: Final = re.compile(FACT_PATTERN)
QUERY: Final = re.compile(r"Where is (Mary|John|Daniel|Sandra)\?")


class Record(Settings):
    """Reject coercions and unknown fields at persisted-data boundaries."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )


class Fact(Record):
    """One declarative source line and its exact semantic interpretation."""

    line_id: Positive
    text: str
    person: Person
    location: Location

    @model_validator(mode="after")
    def grammar(self) -> Self:
        """Bind structured labels to the unchanged factual sentence."""
        match = FACT.fullmatch(self.text)
        if match is None or match.groups() != (self.person, self.location):
            raise BabiError(reason="Invalid fact grammar or labels")
        return self


class Question(Record):
    """A source-local question, preceding facts, and separately encoded segments."""

    episode_id: Annotated[int, Field(ge=0)]
    question_line: Positive
    queried_person: Person
    answer: Location
    support_line: Positive
    facts: tuple[Fact, ...]
    prompt_ids: tuple[Token, ...] = ()
    answer_ids: tuple[Token, ...] = ()
    removed_prompt_ids: tuple[Token, ...] = ()

    @property
    def prompt(self) -> str:
        """Serialize without previous QA records, source IDs, or future facts."""
        return self._format(self.facts)

    def _format(self, facts: tuple[Fact, ...]) -> str:
        return "\n".join(
            (
                *[f.text for f in facts],
                f"Question: Where is {self.queried_person}?",
                "Answer:",
            )
        )

    @property
    def removed_line_ids(self) -> tuple[int, ...]:
        """Identify every removed queried-person fact, not only the support."""
        return tuple(f.line_id for f in self.facts if f.person == self.queried_person)

    @property
    def removed_prompt(self) -> str:
        """Keep only other people's factual history with the ordinary question."""
        return self._format(
            tuple(f for f in self.facts if f.person != self.queried_person)
        )

    @property
    def prompt_hash(self) -> str:
        """Hash the exact formatted UTF-8 prompt, without its answer."""
        return hashlib.sha256(self.prompt.encode()).hexdigest()

    @model_validator(mode="after")
    def grounded(self) -> Self:
        """Require chronological facts and agreement of support and latest state."""
        ids = tuple(f.line_id for f in self.facts)
        history = tuple(f for f in self.facts if f.person == self.queried_person)
        if not ids or ids != tuple(sorted(set(ids))) or ids[-1] >= self.question_line:
            raise BabiError(reason="Invalid preceding fact source IDs")
        if not history or (history[-1].line_id, history[-1].location) != (
            self.support_line,
            self.answer,
        ):
            raise BabiError(
                reason="Supporting fact and latest-location oracle mismatch"
            )
        if self.answer_ids and self.answer_ids != ANSWER_IDS[self.answer]:
            raise BabiError(reason="Changed answer encoding")
        if len(self.prompt_ids) + len(self.answer_ids) - 1 > CONTEXT:
            raise BabiError(reason="Supervised input overflow; truncation forbidden")
        return self


def _parse_question(
    fields: list[str], facts: list[Fact], episode: int, line: int
) -> Question:
    match = QUERY.fullmatch(fields[0])
    if (
        len(fields) != QUESTION_FIELDS
        or match is None
        or re.fullmatch(r"[1-9][0-9]*", fields[2]) is None
    ):
        raise BabiError(reason="Invalid question grammar or single support ID")
    return Question.model_validate(
        {
            "episode_id": episode,
            "question_line": line,
            "queried_person": match[1],
            "answer": fields[1],
            "support_line": int(fields[2]),
            "facts": tuple(facts),
        }
    )


def parse_episodes(source: bytes) -> tuple[tuple[Question, ...], ...]:
    """Parse contiguous source IDs, resetting all facts at each episode boundary."""
    episodes: list[tuple[Question, ...]] = []
    questions: list[Question] = []
    facts: list[Fact] = []
    previous = 0
    try:
        lines = source.decode("utf-8").removesuffix("\n").split("\n")
        for raw in lines:
            match = re.fullmatch(r"([1-9][0-9]*) (.+)", raw.strip(" \r"))
            if match is None:
                raise BabiError(reason="Invalid source record")
            line, body = int(match[1]), match[2]
            if line == 1 and previous:
                if not questions:
                    raise BabiError(reason="Episode has no questions")
                episodes.append(tuple(questions))
                questions, facts, previous = [], [], 0
            if line != previous + 1:
                raise BabiError(
                    reason="Noncontiguous source IDs or missing episode start"
                )
            previous = line
            fields = [field.strip() for field in body.split("\t")]
            if len(fields) > 1:
                questions.append(_parse_question(fields, facts, len(episodes), line))
            else:
                fact = FACT.fullmatch(body.strip())
                if fact is None:
                    raise BabiError(reason="Invalid declarative grammar")
                facts.append(
                    Fact.model_validate(
                        {
                            "line_id": line,
                            "text": body.strip(),
                            "person": fact[1],
                            "location": fact[2],
                        }
                    )
                )
        if not questions:
            raise BabiError(reason="Empty source or episode without questions")
        episodes.append(tuple(questions))
    except (UnicodeDecodeError, ValidationError) as error:
        raise BabiError(reason="Malformed Task 1 source") from error
    return tuple(episodes)


def split_training(
    episodes: tuple[tuple[Question, ...], ...],
) -> tuple[tuple[Question, ...], tuple[Question, ...]]:
    """Apply PCG64(0) episode allocation and validation-only prompt exclusion."""
    if len(episodes) != TRAIN_EPISODES:
        raise BabiError(reason="Official training requires 2000 episodes")
    order: np.ndarray[tuple[int, ...], np.dtype[np.int64]] = np.random.Generator(
        np.random.PCG64(0)
    ).permutation(TRAIN_EPISODES)
    heldout = {int(order.item(i)) for i in range(1800, TRAIN_EPISODES)}
    valid = tuple(
        q for i, episode in enumerate(episodes) if i in heldout for q in episode
    )
    prompts = {q.prompt for q in valid}
    train = tuple(
        q
        for i, episode in enumerate(episodes)
        if i not in heldout
        for q in episode
        if q.prompt not in prompts
    )
    return train, valid


def load_tokenizer(path: Path) -> BPETokenizer:
    """Load only the exact externally fitted tokenizer; never train or add tokens."""
    serialized = path.read_bytes()
    if (
        hashlib.sha256(serialized).hexdigest() != TOKENIZER_HASH
        or version("tokenizers") != "0.22.2"
    ):
        raise BabiError(reason="Changed tokenizer bytes or implementation version")
    tokenizer = restore_tokenizer(serialized.decode("utf-8"))
    if len(tokenizer.get_vocab()) != VOCABULARY or any(
        tuple(tokenizer.encode(" " + answer + "\n", add_special_tokens=False).ids)
        != ids
        for answer, ids in ANSWER_IDS.items()
    ):
        raise BabiError(reason="Changed tokenizer vocabulary or answer encodings")
    return tokenizer


def encode_question(question: Question, tokenizer: BPETokenizer) -> Question:
    """Encode segments separately, preserving the answer boundary without truncation."""
    try:
        return Question.model_validate(
            {
                **question.model_dump(),
                "prompt_ids": tuple(
                    tokenizer.encode(question.prompt, add_special_tokens=False).ids
                ),
                "answer_ids": tuple(
                    tokenizer.encode(
                        " " + question.answer + "\n", add_special_tokens=False
                    ).ids
                ),
                "removed_prompt_ids": tuple(
                    tokenizer.encode(
                        question.removed_prompt, add_special_tokens=False
                    ).ids
                ),
            }
        )
    except ValidationError as error:
        raise BabiError(reason=f"Invalid encoding or overflow: {error}") from error
