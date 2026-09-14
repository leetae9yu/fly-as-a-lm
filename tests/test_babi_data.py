"""Strict bAbI parsing, lexical boundaries, isolation, and real local archive IO."""

import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from flyrl.babi_data import (
    ANSWER_IDS,
    BabiError,
    Question,
    encode_question,
    load_tokenizer,
    parse_episodes,
    split_training,
)
from flyrl.babi_storage import MEMBERS, Metadata, load_test, load_train_valid
from flyrl.bpe_tokenizer import BPETokenizer
from scripts.prepare_babi_task1 import app, parse_official, read_members, read_source

ARCHIVE = Path("artifacts/babi_tasks_1-20_v1-2-parlai.tar.gz")
TOKENIZER = Path("data/tinystories_quality/tokenizer.json")
STORY = (
    "1 Mary moved to the bathroom.\n2 John went to the hallway.\n"
    "3 Where is Mary? \tbathroom\t1\n4 Mary moved to the garden.\n"
    "5 Where is Mary?\tgarden\t4\n"
)


@dataclass(frozen=True, slots=True)
class Encoding:
    ids: list[int]


def test_episode_prompts_and_history_removal() -> None:
    episodes = parse_episodes((STORY + STORY).encode())
    assert len(episodes) == 2
    first, second = episodes[0]
    assert (first.episode_id, episodes[1][0].episode_id) == (0, 1)
    assert (first.question_line, first.support_line) == (3, 1)
    assert first.prompt == (
        "Mary moved to the bathroom.\nJohn went to the hallway.\n"
        "Question: Where is Mary?\nAnswer:"
    )
    assert second.prompt == (
        "Mary moved to the bathroom.\nJohn went to the hallway.\n"
        "Mary moved to the garden.\nQuestion: Where is Mary?\nAnswer:"
    )
    assert second.removed_line_ids == (1, 4)
    assert second.removed_prompt == (
        "John went to the hallway.\nQuestion: Where is Mary?\nAnswer:"
    )
    assert second.queried_person == "Mary"
    assert tuple(f.line_id for f in second.facts) == (1, 2, 4)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "2 Mary moved to the garden.\n",
        "1 Mary vanished.\n",
        "1 Mary moved to the garden.\n3 Where is Mary?\tgarden\t1\n",
        "1 Mary moved to the garden.\n2 Where is Mary?\tbathroom\t1\n",
        "1 Mary moved to the garden.\n2 Where is John?\tgarden\t1\n",
        "1 Mary moved to the garden.\n2 Where is Mary?\tgarden\t1 1\n",
        "1 Mary moved to the garden.\n2 Where is Mary?\tgarden\t3\n",
        "1 Mary moved to the garden.\n2 Where is Mary?\tgarden\t0\n",
        "1 Mary moved to the garden.\n2 Where is Mary?\tgarden\t1\textra\n",
        "1 Mary moved to the garden.\n2 Where is Mary?\tgarden\t1\t\n",
        "1 Mary moved to the garden.\n2 Where was Mary?\tgarden\t1\n",
        STORY + "6 Where is Mary?\tbathroom\t1\n",
        STORY + "6 Where is Mary?\tgarden\t5\n",
        STORY + "1 John went to the office.\n2 Where is Mary?\tgarden\t4\n",
        "1 Mary moved to the garden.\n\n2 Where is Mary?\tgarden\t1\n",
        "1 Mary moved to the garden.\n",
        STORY.replace("\n", "\v"),
        STORY + "1 John went to the office.\n",
    ],
)
def test_malformed_and_inconsistent_records(text: str) -> None:
    with pytest.raises(BabiError):
        _ = parse_episodes(text.encode())


def test_encoding_separation_and_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = load_tokenizer(TOKENIZER)
    question = parse_episodes(STORY.encode())[0][0]
    original = tokenizer.encode
    calls: list[str] = []

    def encode(
        _self: BPETokenizer, sequence: str, *, add_special_tokens: bool = True
    ) -> Encoding:
        assert not add_special_tokens
        calls.append(sequence)
        return Encoding(original(sequence, add_special_tokens=add_special_tokens).ids)

    monkeypatch.setattr(type(tokenizer), "encode", encode)
    encoded = encode_question(question, tokenizer)
    assert calls == [question.prompt, " bathroom\n", question.removed_prompt]
    assert encoded.prompt_ids == tuple(
        tokenizer.encode(question.prompt, add_special_tokens=False).ids
    )
    assert encoded.answer_ids == (2811, 199)
    assert encoded.removed_prompt_ids == tuple(
        tokenizer.encode(question.removed_prompt, add_special_tokens=False).ids
    )
    assert tuple(ANSWER_IDS.values()) == (
        (2811, 199),
        (2683, 199),
        (851, 199),
        (3885, 453, 199),
        (1198, 199),
        (3175, 199),
    )
    for answer, expected in ANSWER_IDS.items():
        assert (
            tuple(tokenizer.encode(" " + answer + "\n", add_special_tokens=False).ids)
            == expected
        )
    with pytest.raises(ValidationError, match="answer encoding"):
        _ = Question.model_validate({**encoded.model_dump(), "answer_ids": (851, 199)})
    text = "".join(f"{i} Mary moved to the garden.\n" for i in range(1, 41))
    overflow = parse_episodes((text + "41 Where is Mary?\tgarden\t40\n").encode())[0][0]
    with pytest.raises(BabiError, match="overflow"):
        _ = encode_question(overflow, tokenizer)
    changed = tmp_path / "tokenizer.json"
    _ = changed.write_bytes(TOKENIZER.read_bytes() + b" ")
    with pytest.raises(BabiError, match="tokenizer"):
        _ = load_tokenizer(changed)


def test_validation_only_split() -> None:
    episodes = parse_episodes((STORY * 2000).encode())
    train, valid = split_training(episodes)
    assert train == ()
    assert len(valid) == 400
    order: np.ndarray[tuple[int, ...], np.dtype[np.int64]] = np.random.Generator(
        np.random.PCG64(0)
    ).permutation(2000)
    expected = {int(order.item(i)) for i in range(1800, 2000)}
    assert {q.episode_id for q in valid} == expected
    assert [(q.episode_id, q.question_line) for q in valid] == sorted(
        (q.episode_id, q.question_line) for q in valid
    )
    with pytest.raises(BabiError, match="episodes"):
        _ = split_training(episodes[:1])


@pytest.mark.parametrize(
    "case",
    [
        ("", tarfile.REGTYPE, 2, False, "Duplicate"),
        ("../", tarfile.REGTYPE, 1, False, "unsafe"),
        ("./", tarfile.REGTYPE, 1, False, "unsafe"),
        ("/", tarfile.REGTYPE, 1, False, "unsafe"),
        ("x\\", tarfile.REGTYPE, 1, False, "unsafe"),
        ("", tarfile.SYMTYPE, 1, False, "regular"),
        ("", tarfile.CONTTYPE, 1, False, "regular"),
        ("", tarfile.LNKTYPE, 1, False, "regular"),
        ("", tarfile.REGTYPE, 1, False, "Missing"),
        ("", tarfile.REGTYPE, 1, True, "hash"),
    ],
)
def test_archive_member_rejection(
    tmp_path: Path, case: tuple[str, bytes, int, bool, str]
) -> None:
    prefix, kind, copies, corrupt, message = case
    archive = tmp_path / "bad.tar.gz"
    data = read_source(ARCHIVE).train
    with tarfile.open(archive, "w:gz") as stream:
        for _ in range(copies):
            member = tarfile.TarInfo(prefix + MEMBERS[0].name)
            member.type, member.linkname, member.size = kind, "/etc/passwd", len(data)
            stream.addfile(member, io.BytesIO(b"!" + data[1:] if corrupt else data))
    with tarfile.open(archive) as stream, pytest.raises(BabiError, match=message):
        _ = read_members(stream)
    with pytest.raises(BabiError, match="archive size/hash"):
        _ = read_source(archive)
    with pytest.raises(BabiError, match="counts"):
        _ = parse_official(STORY.encode(), 1, 3)


def _questions_identity(questions: tuple[Question, ...]) -> tuple[tuple[int, int], ...]:
    return tuple((q.episode_id, q.question_line) for q in questions)


def test_real_cli_determinism_and_tampering(tmp_path: Path) -> None:
    runner = CliRunner()
    outputs = (tmp_path / "a", tmp_path / "b")
    for output in outputs:
        result = runner.invoke(
            app,
            ["--archive", str(ARCHIVE), "--output", str(output)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout.splitlines()) == 1
        assert Metadata.model_validate_json(
            result.stdout
        ) == Metadata.model_validate_json((output / "provenance.json").read_bytes())
    first, second = outputs
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
    corpus = load_train_valid(first)
    test = load_test(first)
    splits = (corpus.train, corpus.valid, test)
    assert tuple(len(s.questions) for s in splits) == (8983, 1000, 1000)
    assert tuple(s.unique_prompts for s in splits) == (8909, 999, 999)
    train_hashes = {q.prompt_hash for q in corpus.train.questions}
    assert not train_hashes.intersection(q.prompt_hash for q in corpus.valid.questions)
    assert sum(q.prompt_hash in train_hashes for q in test.questions) == 16
    source = read_source(ARCHIVE)
    expected_train, expected_valid = split_training(parse_episodes(source.train))
    assert _questions_identity(corpus.train.questions) == _questions_identity(
        expected_train
    )
    assert _questions_identity(corpus.valid.questions) == _questions_identity(
        expected_valid
    )
    assert _questions_identity(test.questions) == _questions_identity(
        tuple(q for e in parse_episodes(source.test) for q in e)
    )
    assert (first / "SOURCE_LICENSE.txt").read_bytes() == source.license_bytes
    assert (
        hashlib.sha256(source.license_bytes).hexdigest()
        == "d12f09a636365a040fd581ebab6bf018d6fe61973c6d4862f841a6aca2f53efa"
    )
    for split in splits:
        assert sum(entry.count for entry in split.answer_histogram) == len(
            split.questions
        )
        assert (
            max(len(q.prompt_ids) + len(q.answer_ids) - 1 for q in split.questions)
            <= 160
        )
    with pytest.raises(ValidationError):
        corpus.train.questions[0].answer = "office"
    refused = runner.invoke(app, ["--archive", str(ARCHIVE), "--output", str(first)])
    assert isinstance(refused.exception, FileExistsError)
    metadata_path = first / "provenance.json"
    metadata = Metadata.model_validate_json(metadata_path.read_bytes())
    changed = metadata.model_copy(
        update={"train": metadata.train.model_copy(update={"questions": 0})}
    )
    _ = metadata_path.write_text(changed.model_dump_json())
    with pytest.raises(ValidationError, match="counts"):
        _ = load_train_valid(first)
    _ = metadata_path.write_text(metadata.model_dump_json())
    _ = (first / "test.npz").write_bytes(b"unopened test")
    assert load_train_valid(first) == corpus
    with pytest.raises(BabiError, match="hash"):
        _ = load_test(first)
    _ = (first / "train_valid.npz").write_bytes(b"tampered")
    with pytest.raises(BabiError, match="hash"):
        _ = load_train_valid(first)
