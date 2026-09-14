"""Read-only transaction inspection and literal per-question evidence validation."""

from dataclasses import dataclass
from math import fsum
from pathlib import Path
from typing import TypeVar
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from pydantic import BaseModel, TypeAdapter

from flyrl.babi_data import BabiError, Question
from flyrl.babi_generation import MAX_GENERATED
from flyrl.babi_learning import TERMINATOR, VOCABULARY
from flyrl.babi_metrics import Decode
from flyrl.babi_pilot_schema import Evaluation, Manifest, Progress, Validation
from flyrl.babi_protocol import BabiProtocol
from scripts.connectome_source import file_digest

Model = TypeVar("Model", bound=BaseModel)


def require(condition: bool, reason: str) -> None:
    """Fail closed on untrusted evidence, with a typed error."""
    if not condition:
        raise BabiError(reason=reason)


def read_model(model: type[Model], path: Path) -> Model:
    """Parse nested evidence with strict types and forbidden extra fields."""
    return model.model_validate_json(path.read_bytes(), strict=True)


def checkpoint(folder: Path, protocol: BabiProtocol, ids: tuple[str, ...]) -> Manifest:
    """Inspect all arrays and replay CPU sampling, never mutate or create a learner."""
    require(
        not folder.is_symlink()
        and {p.name for p in folder.iterdir()} == {"checkpoint.npz", "manifest.json"}
        and all(p.is_file() and not p.is_symlink() for p in folder.iterdir()),
        "Checkpoint transaction membership mismatch",
    )
    manifest = read_model(Manifest, folder / "manifest.json")
    progress = manifest.progress
    progress.verify(protocol, ids)
    require(
        len(ids) == len(set(ids)) == protocol.corpus.counts[0]
        and folder.name == f"update-{progress.updates:06d}"
        and all(row.nll >= 0 and row.gradient_norm >= 0 for row in progress.trace),
        "Checkpoint identity or trace mismatch",
    )
    path = folder / "checkpoint.npz"
    require(file_digest(path) == manifest.checkpoint_sha256, "Checkpoint hash mismatch")
    sensory = min(protocol.graph.nodes // 2, 192)
    shapes = {
        "codes": (4096, sensory),
        "weight": (protocol.graph.edges,),
        "bias": (protocol.graph.nodes,),
        "readout": (
            min(protocol.config.readout_neurons, protocol.graph.nodes - sensory),
            4096,
        ),
        "output_bias": (4096,),
    }
    expected = {"progress", "rng"} | {f"model.{name}" for name in shapes}
    if progress.updates:
        expected.update(
            f"adam.{name}.{key}"
            for name in shapes
            for key in ("step", "exp_avg", "exp_avg_sq")
        )
    try:
        with ZipFile(path) as archive:
            require(
                len(archive.namelist()) == len(expected)
                and set(archive.namelist()) == {key + ".npy" for key in expected}
                and archive.testzip() is None,
                "Checkpoint array membership or CRC mismatch",
            )
        with (
            path.open("rb") as stream,
            NpzFile[np.generic](stream, allow_pickle=False) as data,
        ):
            require(
                data["progress"].shape == () and data["progress"].dtype.kind == "U",
                "Checkpoint progress array mismatch",
            )
            embedded = Progress.model_validate_json(
                TypeAdapter(str).validate_python(data["progress"].item(), strict=True),
                strict=True,
            )
            require(embedded == progress, "Checkpoint embedded progress mismatch")
            rng = torch.Generator(device="cpu").manual_seed(protocol.config.seed + 37)
            for row in progress.trace:
                indices = torch.randint(
                    len(ids), (protocol.config.batch_size,), generator=rng
                )
                require(
                    tuple(ids[int(index)] for index in indices) == row.sampled_ids,
                    "Checkpoint sampled ledger mismatch",
                )
            state = data["rng"]
            require(
                state.dtype == np.uint8 and state.shape == tuple(rng.get_state().shape),
                "Checkpoint RNG shape/dtype mismatch",
            )
            restored = torch.Generator(device="cpu").set_state(torch.tensor(state))
            require(
                torch.equal(restored.get_state(), rng.get_state()),
                "Checkpoint RNG state mismatch",
            )
            for name, shape in shapes.items():
                keys = [f"model.{name}"]
                if progress.updates:
                    keys.extend(
                        f"adam.{name}.{key}"
                        for key in ("step", "exp_avg", "exp_avg_sq")
                    )
                for key in keys:
                    array = data[key]
                    require(
                        array.dtype == np.float32
                        and array.shape == (() if key.endswith(".step") else shape)
                        and bool(np.isfinite(array).all()),
                        "Checkpoint tensor shape/dtype/finiteness mismatch",
                    )
                    require(
                        not key.endswith(".step") or array.item() == progress.updates,
                        "Checkpoint Adam step mismatch",
                    )
                    require(
                        not key.endswith(".exp_avg_sq")
                        or bool(np.greater_equal(array, 0).all()),
                        "Checkpoint negative Adam variance",
                    )
    except (BadZipFile, KeyError, EOFError, RuntimeError, OSError) as error:
        raise BabiError(reason=f"Invalid checkpoint transaction: {error}") from error
    return manifest


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """The selected checkpoint, input condition and unchanged lexical decoder."""

    checkpoint_sha256: str
    removed: bool
    decode: Decode


@dataclass(frozen=True, slots=True)
class Totals:
    """Independently summed primary answer evidence, without cached ratios."""

    correct: int
    questions: int
    loss_sum: float
    tokens: int
    nll: float


def evaluation(
    item: Evaluation, questions: tuple[Question, ...], context: EvaluationContext
) -> Totals:
    """Bind source order, gold, removal, token decoding and answer-loss denominators."""
    require(
        len(item.predictions)
        == len(item.answer_nll.per_question)
        == len(questions)
        > 0,
        "Evaluation question count mismatch",
    )
    identities = tuple(p.example_id for p in item.predictions)
    require(len(set(identities)) == len(identities), "Evaluation duplicate identity")
    correct = 0
    for prediction, loss, question in zip(
        item.predictions, item.answer_nll.per_question, questions, strict=True
    ):
        require(
            (prediction.episode_id, prediction.question_line)
            == (question.episode_id, question.question_line)
            and prediction.example_id
            == f"{question.episode_id:04d}:{question.question_line:02d}"
            and prediction.gold_answer == question.answer
            and prediction.checkpoint_sha256 == context.checkpoint_sha256
            and prediction.removed_line_ids
            == (question.removed_line_ids if context.removed else ())
            and loss.example_id == prediction.example_id
            and loss.count == len(question.answer_ids),
            "Evaluation source/gold/checkpoint/removal/NLL identity mismatch",
        )
        tokens = prediction.generated_ids
        require(
            0 < len(tokens) <= MAX_GENERATED
            and TERMINATOR not in tokens[:-1]
            and all(0 <= token < VOCABULARY for token in tokens),
            "Generation token or stopping mismatch",
        )
        terminated = tokens[-1] == TERMINATOR
        require(
            prediction.terminated == terminated
            and (terminated or len(tokens) == MAX_GENERATED),
            "Generation terminator mismatch",
        )
        text = context.decode(tokens[:-1] if terminated else tokens)
        require(prediction.decoded_answer == text, "Generation decoded text mismatch")
        correct += terminated and text == " " + question.answer
    losses = item.answer_nll.per_question
    return Totals(
        correct,
        len(questions),
        fsum(q.loss_sum for q in losses),
        sum(q.count for q in losses),
        fsum(q.loss_sum / q.count for q in losses) / len(questions),
    )


def validation(update: int, totals: Totals) -> Validation:
    """Construct selection inputs from independently checked question evidence."""
    return Validation(
        update=update,
        correct=totals.correct,
        questions=totals.questions,
        nll=totals.nll,
    )
