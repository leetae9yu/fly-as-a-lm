# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2,<3", "pydantic>=2.10,<3", "tokenizers==0.22.2", "typer>=0.15,<1"
# ]
# ///
"""Prepare the frozen Task 1 corpus from the authenticated local Meta archive."""

import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Final

import typer

from flyrl.babi_data import (
    BabiError,
    Question,
    encode_question,
    load_tokenizer,
    parse_episodes,
    split_training,
)
from flyrl.babi_storage import (
    ARCHIVE_HASH,
    ARCHIVE_SIZE,
    MEMBERS,
    Metadata,
    Provenance,
    Source,
    Split,
    TrainValid,
    save_payload,
)

app: Final = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
QUESTIONS_PER_EPISODE: Final = 5
TEST_SEEN: Final = 16


def read_members(archive: tarfile.TarFile) -> Source:
    """Read only allowlisted regular files, rejecting aliases and duplicate names."""
    seen: set[str] = set()
    found: dict[str, bytes] = {}
    pins = {member.name: member for member in MEMBERS}
    for member in archive:
        name = member.name
        path = PurePosixPath(name)
        if (
            name in seen
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or str(path) != name.rstrip("/")
        ):
            raise BabiError(reason="Duplicate or unsafe archive member path")
        seen.add(name)
        if name not in pins:
            continue
        pin = pins[name]
        if (
            member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
            or member.size != pin.size
        ):
            raise BabiError(reason="Required member is not regular or has wrong size")
        stream = archive.extractfile(member)
        if stream is None:
            raise BabiError(reason="Missing regular member stream")
        with stream:
            payload = stream.read(pin.size + 1)
        if (
            len(payload) != pin.size
            or hashlib.sha256(payload).hexdigest() != pin.sha256
        ):
            raise BabiError(reason="Archive member hash mismatch")
        found[name] = payload
    if set(found) != set(pins):
        raise BabiError(reason="Missing required archive members")
    return Source(
        train=found[MEMBERS[0].name],
        test=found[MEMBERS[1].name],
        license_bytes=found[MEMBERS[2].name],
    )


def read_source(path: Path) -> Source:
    """Authenticate a bounded local archive and read only three named regular files."""
    with path.open("rb") as stream:
        payload = stream.read(ARCHIVE_SIZE + 1)
    if (
        len(payload) != ARCHIVE_SIZE
        or hashlib.sha256(payload).hexdigest() != ARCHIVE_HASH
    ):
        raise BabiError(reason="Official archive size/hash mismatch")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        return read_members(archive)


def parse_official(
    source: bytes, episodes: int, questions: int
) -> tuple[tuple[Question, ...], ...]:
    """Validate official episode and question totals before splitting or encoding."""
    parsed = parse_episodes(source)
    if (
        len(parsed) != episodes
        or sum(map(len, parsed)) != questions
        or any(len(e) != QUESTIONS_PER_EPISODE for e in parsed)
    ):
        raise BabiError(reason="Official source episode/question counts mismatch")
    return parsed


@app.command()
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """Create an exclusive output directory and emit one JSON provenance summary."""

    archive: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
    output: Annotated[Path, typer.Option(file_okay=False)]
    tokenizer: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/tinystories_quality/tokenizer.json"
    )

    def __post_init__(self) -> None:
        """Validate all source, split, oracle, and lexical invariants before writing."""
        if self.output.exists():
            raise FileExistsError(self.output)
        source = read_source(self.archive)
        tokenizer = load_tokenizer(self.tokenizer)
        train, valid = split_training(parse_official(source.train, 2000, 10000))
        corpus = TrainValid(
            train=Split(questions=tuple(encode_question(q, tokenizer) for q in train)),
            valid=Split(questions=tuple(encode_question(q, tokenizer) for q in valid)),
        )
        # Training membership is finalized before even parsing the official test.
        test = Split(
            questions=tuple(
                encode_question(q, tokenizer)
                for episode in parse_official(source.test, 200, 1000)
                for q in episode
            )
        )
        splits = (corpus.train, corpus.valid, test)
        counts = tuple((len(s.questions), s.unique_prompts) for s in splits)
        train_prompts = {q.prompt for q in train}
        if counts != (
            (8983, 8909),
            (1000, 999),
            (1000, 999),
        ) or train_prompts.intersection(q.prompt for q in valid):
            raise BabiError(
                reason="Frozen split counts or validation decontamination mismatch"
            )
        if sum(q.prompt in train_prompts for q in test.questions) != TEST_SEEN:
            raise BabiError(reason="Untouched test overlap diagnostics mismatch")
        self.output.mkdir(parents=True, exist_ok=False)
        metadata = Metadata(
            provenance=Provenance(),
            train=corpus.train.report,
            valid=corpus.valid.report,
            test=test.report,
            train_valid_sha256=save_payload(corpus, self.output / "train_valid.npz"),
            test_sha256=save_payload(test, self.output / "test.npz"),
            split_fingerprint=corpus.fingerprint,
        )
        for name, payload in (
            ("tokenizer.json", self.tokenizer.read_bytes()),
            ("SOURCE_LICENSE.txt", source.license_bytes),
            ("provenance.json", metadata.model_dump_json(indent=2).encode() + b"\n"),
        ):
            with (self.output / name).open("xb") as stream:
                _ = stream.write(payload)
        typer.echo(metadata.model_dump_json())


if __name__ == "__main__":
    app()
