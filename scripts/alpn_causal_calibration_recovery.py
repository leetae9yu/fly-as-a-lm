"""Strict CPU recovery of calibration evidence, with no fresh corpus access."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from pydantic import BaseModel, TypeAdapter

from flyrl.ar_checkpoint import Metadata
from flyrl.ar_learning import ARLearner
from flyrl.language_runtime import graph_fingerprint
from scripts.alpn_causal_calibration import TrainingActivity, paired_coordinates
from scripts.alpn_causal_calibration_support import (
    Context,
    check_files,
    load_context,
    selected_indices,
    selection,
    snapshot,
)
from scripts.alpn_causal_calibration_types import (
    CalibrationResult,
    CalibrationSeal,
    MatchingRecord,
    SeedCalibration,
    SourceEvidence,
)
from scripts.alpn_causal_groups import COORDINATES, match_controls
from scripts.alpn_causal_source_archive import read_source_payloads

STATE_BOUND = 1.000001


class SourceRuntime(BaseModel):
    """Read only library identity from the hash-authenticated source runtime."""

    torch: str
    cuda: str


def recover_model(context: Context, evidence: SourceEvidence) -> ARLearner:
    """Reconstruct topology and saved tensors on CPU, without parsing report metrics."""
    source = evidence.source
    members = {
        "checkpoint.npz": source.checkpoint_sha256,
        "report.json": source.report_sha256,
        "runtime.json": source.runtime_sha256,
    }
    payloads = read_source_payloads(context.root / "sources" / source.filename)
    if any(
        sha256(payloads[name]).hexdigest() != digest for name, digest in members.items()
    ):
        message = "Calibration source member hash differs"
        raise ValueError(message)
    runtime = SourceRuntime.model_validate_json(payloads["runtime.json"])
    if runtime.torch != evidence.runtime.torch or runtime.cuda != evidence.runtime.cuda:
        message = "Calibration source runtime differs"
        raise ValueError(message)
    with NpzFile(BytesIO(payloads["checkpoint.npz"]), allow_pickle=False) as data:
        metadata = Metadata.model_validate_json(
            TypeAdapter(str).validate_python(data["metadata"].item(), strict=True)
        )
        config = metadata.config
        plan = next(
            plan for plan in context.inputs.groups.seeds if plan.seed == source.seed
        )
        if (
            config.seed != source.seed
            or config.control != source.wiring
            or config.sensory_indices != plan.sensory_indices
            or config.readout_indices != plan.trained_readout_indices
            or metadata.corpus != context.inputs.protocol.corpus_fingerprint
            or metadata.graph != graph_fingerprint(context.inputs.graph)
        ):
            message = "Calibration checkpoint identity differs"
            raise ValueError(message)
        learner = ARLearner(
            context.inputs.graph, config.model_copy(update={"device": "cpu"})
        )
        learner.config = config
        with torch.no_grad():
            for name, parameter in learner.model.named_parameters():
                values = data[f"model.{name}"]
                if (
                    values.shape != tuple(parameter.shape)
                    or values.dtype != parameter.detach().numpy().dtype
                ):
                    message = "Calibration checkpoint tensor shape or dtype differs"
                    raise ValueError(message)
                _ = parameter.copy_(torch.tensor(values))
                moments = {
                    key: torch.tensor(np.asarray(data[f"adam.{name}.{key}"]))
                    for key in ("step", "exp_avg", "exp_avg_sq")
                    if f"adam.{name}.{key}" in data.files
                }
                if moments:
                    learner.optimizer.state[parameter] = moments
        _ = learner.window_rng.set_state(torch.tensor(data["rng"]))
        learner.updates, learner.trace = metadata.updates, list(metadata.trace)
        _ = learner.model.eval()
        _ = learner.model.requires_grad_(requires_grad=False)
    if snapshot(learner) != evidence.before:
        message = "Calibration source snapshot differs from checkpoint"
        raise ValueError(message)
    return learner


def validate_seed(context: Context, seed: SeedCalibration) -> None:
    """Recompute every coordinate and matching statistic, including overlaps/status."""
    indices = selected_indices(context, seed.seed)
    sources = tuple(
        source
        for source in context.inputs.protocol.checkpoints
        if source.seed == seed.seed
    )
    if seed.indices != indices or tuple(
        item.source.model_dump() for item in seed.sources
    ) != tuple(source.model_dump() for source in sources):
        message = "Calibration seed selection or source pair differs"
        raise ValueError(message)
    activity = tuple(item.array(len(indices), 3) for item in seed.activity)
    if any(
        bool((np.abs(values[:, :2]) > STATE_BOUND).any())
        or bool((values[:, 1:] < 0).any())
        or bool((values[:, 2] > 1).any())
        for values in activity
    ):
        message = "Calibration activity bounds differ"
        raise ValueError(message)
    real, shuffled = (recover_model(context, item) for item in seed.sources)
    coordinates = paired_coordinates(
        context.inputs.graph,
        real.model,
        shuffled.model,
        (
            TrainingActivity(indices, seed.positions, activity[0]),
            TrainingActivity(indices, seed.positions, activity[1]),
        ),
    )
    stored = seed.coordinates.array(len(indices), len(COORDINATES))
    if not np.array_equal(coordinates, stored):
        message = "Calibration coordinate evidence differs"
        raise ValueError(message)
    full = np.zeros(
        (len(context.inputs.graph.node_ids), len(COORDINATES)), dtype=np.float64
    )
    full[list(indices)] = stored
    plan = next(plan for plan in context.inputs.groups.seeds if plan.seed == seed.seed)
    draws = tuple(group.indices for group in plan.groups if group.name == "alpn")
    matching = MatchingRecord.capture(
        match_controls(
            context.inputs.graph.node_ids, selection(context, seed.seed), draws, full
        )
    )
    if seed.matching != matching:
        message = "Calibration controls, costs, balance or status differ"
        raise ValueError(message)


def validate_result(result: CalibrationResult, context: Context) -> None:
    """Bind six results to trusted code, data, protocol and source hashes."""
    result = CalibrationResult.model_validate(result)
    check_files(context.root, context.seal)
    if (
        result.seal != context.seal
        or sha256(context.tokens.tobytes()).hexdigest()
        != context.seal.training_tokens_sha256
        or sha256(
            np.asarray(context.split.offsets, dtype=np.int64).tobytes()
        ).hexdigest()
        != context.seal.training_offsets_sha256
    ):
        message = "Calibration trusted seal or old-training hash differs"
        raise ValueError(message)
    for seed in result.seeds:
        validate_seed(context, seed)
    check_files(context.root, context.seal)


def recover_package(path: Path, context: Context) -> CalibrationResult:
    """Reject missing, extra, duplicate, nested, corrupt or inconsistent payloads."""
    with ZipFile(path) as archive:
        if archive.namelist() != ["calibration.json"] or archive.testzip() is not None:
            message = "Calibration result ZIP membership or CRC differs"
            raise ValueError(message)
        payload = archive.read("calibration.json")
    result = CalibrationResult.model_validate_json(payload)
    if payload != result.model_dump_json().encode():
        message = "Calibration JSON must use its canonical unique-key encoding"
        raise ValueError(message)
    validate_result(result, context)
    return result


def package_result(
    destination: Path, result: CalibrationResult, context: Context
) -> None:
    """Exclusively publish deterministic bytes, after CRC testing and full recovery."""
    result = CalibrationResult.model_validate(result)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "result.zip"
        with ZipFile(staged, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            info = ZipInfo("calibration.json", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                result.model_dump_json().encode(),
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )
        if recover_package(staged, context) != result:
            message = "Calibration deterministic recovery differs"
            raise ValueError(message)
        destination.hardlink_to(staged)


def recover(path: Path, root: Path, seal: CalibrationSeal) -> CalibrationResult:
    """Local public recovery: authenticate inputs independently, without any GPU."""
    return recover_package(path, load_context(root, seal))
