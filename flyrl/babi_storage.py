"""Pinned local tar ingestion and separate, pickle-free immutable bAbI artifacts."""

import hashlib
import io
from pathlib import Path
from typing import Annotated, Final, Literal, Self
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from pydantic import Field, model_validator

from flyrl.babi_data import (
    ANSWER_IDS,
    CONTEXT,
    TOKENIZER_HASH,
    BabiError,
    Question,
    Record,
    encode_question,
    load_tokenizer,
)

SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ARCHIVE_HASH: Final = "f7f0bee187efca0d81c3daac1b162cda4eb7f9505dee5ad6846eabbed3dbf92e"
ARCHIVE_SIZE: Final = 19_212_062
URL: Final = "https://dl.fbaipublicfiles.com/parlai/babi/babi.tar.gz"
COMMIT: Final = "ea366da91c93f2cec8fe29b6d93b37ed96ed45bd"


class Member(Record):
    """An exact allowlisted regular archive member."""

    name: str
    size: Annotated[int, Field(gt=0)]
    sha256: SHA


MEMBERS: Final = (
    Member(
        name="tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt",
        size=944248,
        sha256="749ea9f7c99070feb2d88c975a254417a0dcc8274add4435ae5ae24c7afc7e9d",
    ),
    Member(
        name="tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt",
        size=94477,
        sha256="55acf66cef2f6d798e2aa1d056e1ec8f24e910ea9215b639450574321132959b",
    ),
    Member(
        name="tasks_1-20_v1-2/LICENSE.txt",
        size=19561,
        sha256="d12f09a636365a040fd581ebab6bf018d6fe61973c6d4862f841a6aca2f53efa",
    ),
)


class Source(Record):
    """Only the official train, test, and unchanged embedded license bytes."""

    train: bytes
    test: bytes
    license_bytes: bytes


class AnswerCount(Record):
    """One lexically ordered answer histogram entry."""

    answer: str
    count: Annotated[int, Field(ge=0)]


class Split(Record):
    """Source-ordered immutable question records with derived split diagnostics."""

    questions: Annotated[tuple[Question, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        """Reject duplicate or reordered source questions and empty encodings."""
        ids = tuple((q.episode_id, q.question_line) for q in self.questions)
        if ids != tuple(sorted(set(ids))) or any(
            not q.prompt_ids or not q.answer_ids or not q.removed_prompt_ids
            for q in self.questions
        ):
            raise BabiError(reason="Invalid split ordering or empty encoded segment")
        return self

    @property
    def report(self) -> "SplitReport":
        """Derive statistics from stored questions, never sidecar claims."""
        return SplitReport(
            questions=len(self.questions),
            unique_prompts=self.unique_prompts,
            episodes=len({q.episode_id for q in self.questions}),
            max_supervised_input=max(
                len(q.prompt_ids) + len(q.answer_ids) - 1 for q in self.questions
            ),
            answer_histogram=self.answer_histogram,
            fingerprint=self.fingerprint,
        )

    @property
    def unique_prompts(self) -> int:
        """Count exact prompts while preserving official question multiplicity."""
        return len({q.prompt_hash for q in self.questions})

    @property
    def answer_histogram(self) -> tuple[AnswerCount, ...]:
        """Return all six answer frequencies in lexical order."""
        return tuple(
            AnswerCount(answer=a, count=sum(q.answer == a for q in self.questions))
            for a in ANSWER_IDS
        )

    @property
    def fingerprint(self) -> str:
        """Bind all source facts and both encoded input conditions."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class TrainValid(Record):
    """Selection data; constructing or loading this never opens official test."""

    train: Split
    valid: Split

    @model_validator(mode="after")
    def isolated(self) -> Self:
        """Require episode and exact-prompt disjointness at the selection boundary."""
        episodes = {q.episode_id for q in self.train.questions}
        prompts = {q.prompt_hash for q in self.train.questions}
        if any(
            q.episode_id in episodes or q.prompt_hash in prompts
            for q in self.valid.questions
        ):
            raise BabiError(reason="Train/validation episode or prompt overlap")
        return self

    @property
    def fingerprint(self) -> str:
        """Bind selection splits without consulting official test."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class SplitReport(Record):
    """Typed, reproducible count and content evidence for one split."""

    questions: int
    unique_prompts: int
    episodes: int
    max_supervised_input: int
    answer_histogram: tuple[AnswerCount, ...]
    fingerprint: SHA


class Provenance(Record):
    """Authenticated source identity and the exact deterministic transformation."""

    format: Literal["flyrl-babi-task1-v1"] = "flyrl-babi-task1-v1"
    official_url: str = URL
    parlai_manifest_commit: str = COMMIT
    archive_sha256: SHA = ARCHIVE_HASH
    archive_bytes: int = ARCHIVE_SIZE
    members: tuple[Member, ...] = MEMBERS
    tokenizer_sha256: SHA = TOKENIZER_HASH
    tokenizers_version: Literal["0.22.2"] = "0.22.2"
    vocabulary: Literal[4096] = 4096
    terminator: Literal[199] = 199
    license: Literal["CC BY 3.0"] = "CC BY 3.0"
    citation: str = (
        "Weston et al., Towards AI-Complete Question Answering: "
        "A Set of Prerequisite Toy Tasks, arXiv:1502.05698v10"
    )
    transformation: str = (
        "Source line 1 resets episodes; preceding declarative facts only; LF-joined "
        "facts + Question: <question> + Answer:; separate prompt and leading-space "
        "answer+LF encoding; unchanged TinyStories tokenizer; no truncation; "
        "PCG64(0).permutation(2000), first 1800 train episodes, last 200 validation; "
        "exclude train questions only for exact validation prompt overlap; retain "
        "source order and multiplicity; untouched official test; remove all "
        "queried-person facts only in the separate input-removal condition"
    )


class Metadata(Record):
    """Separate file hashes permit selection without opening test bytes."""

    provenance: Provenance
    train: SplitReport
    valid: SplitReport
    test: SplitReport
    excluded_for_validation: Literal[17] = 17
    test_train_seen: Literal[16] = 16
    test_train_novel: Literal[984] = 984
    train_valid_sha256: SHA
    test_sha256: SHA
    split_fingerprint: SHA

    @model_validator(mode="after")
    def protocol(self) -> Self:
        """Reject changed provenance, counts, and impossible context evidence."""
        reports = (self.train, self.valid, self.test)
        counts = tuple((r.questions, r.unique_prompts, r.episodes) for r in reports)
        if (
            self.provenance != Provenance()
            or counts != ((8983, 8909, 1800), (1000, 999, 200), (1000, 999, 200))
            or any(not 0 < r.max_supervised_input <= CONTEXT for r in reports)
        ):
            raise BabiError(reason="Changed protocol provenance or counts")
        return self


def save_payload(payload: Record, path: Path) -> str:
    """Write canonical JSON as a uint8 NPY in a fixed-date exclusive NPZ container."""
    data = np.frombuffer(payload.model_dump_json().encode(), dtype=np.uint8)
    with (
        ZipFile(path, "x", compression=ZIP_DEFLATED) as archive,
        archive.open("records.npy", "w") as stream,
    ):
        write_array(stream, data, allow_pickle=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(path: Path, expected: str) -> bytes:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise BabiError(reason="Prepared artifact hash mismatch")
    with NpzFile[np.generic](io.BytesIO(payload), allow_pickle=False) as archive:
        if archive.files != ["records"]:
            raise BabiError(reason="Unexpected NPZ members")
        array = archive["records"]
        if array.dtype != np.uint8 or array.ndim != 1:
            raise BabiError(reason="NPZ records must be a uint8 vector")
        return array.tobytes()


def _metadata(path: Path) -> Metadata:
    metadata = Metadata.model_validate_json((path / "provenance.json").read_bytes())
    if (
        hashlib.sha256((path / "SOURCE_LICENSE.txt").read_bytes()).hexdigest()
        != MEMBERS[2].sha256
    ):
        raise BabiError(reason="Source license hash mismatch")
    return metadata


def _validate(split: Split, report: SplitReport, tokenizer_path: Path) -> None:
    tokenizer = load_tokenizer(tokenizer_path)
    if split.report != report or any(
        encode_question(q, tokenizer) != q for q in split.questions
    ):
        raise BabiError(reason="Split fingerprint or token encoding mismatch")


def load_train_valid(path: Path) -> TrainValid:
    """Open only provenance, license, tokenizer, and train/validation NPZ."""
    metadata = _metadata(path)
    corpus = TrainValid.model_validate_json(
        _payload(path / "train_valid.npz", metadata.train_valid_sha256)
    )
    for split, report in (
        (corpus.train, metadata.train),
        (corpus.valid, metadata.valid),
    ):
        _validate(split, report, path / "tokenizer.json")
    if corpus.fingerprint != metadata.split_fingerprint:
        raise BabiError(reason="Selection split fingerprint mismatch")
    return corpus


def load_test(path: Path) -> Split:
    """Explicitly open official test only when the caller has frozen selection."""
    metadata = _metadata(path)
    split = Split.model_validate_json(_payload(path / "test.npz", metadata.test_sha256))
    _validate(split, metadata.test, path / "tokenizer.json")
    return split
