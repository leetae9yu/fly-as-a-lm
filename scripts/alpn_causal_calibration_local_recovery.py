"""Independent cross-runtime CPU recovery for sealed ALPN calibration results."""

from hashlib import sha256
from pathlib import Path
from typing import Final
from zipfile import ZipFile

import numpy as np

from scripts.alpn_causal_calibration import TrainingActivity, paired_coordinates
from scripts.alpn_causal_calibration_recovery import STATE_BOUND, recover_model
from scripts.alpn_causal_calibration_support import (
    Context,
    check_files,
    load_context,
    selected_indices,
    selection,
)
from scripts.alpn_causal_calibration_types import (
    CalibrationResult,
    CalibrationSeal,
    MatchingRecord,
    SeedCalibration,
)
from scripts.alpn_causal_groups import COORDINATES, match_controls

COORDINATE_ATOL: Final = 1e-12


def validate_coordinates(actual: np.ndarray, stored: np.ndarray) -> None:
    """Allow only cross-runtime libm rounding while rejecting material drift."""
    if not np.allclose(actual, stored, rtol=0.0, atol=COORDINATE_ATOL):
        message = "Calibration coordinate evidence differs"
        raise ValueError(message)


def validate_seed(context: Context, seed: SeedCalibration) -> None:
    """Recompute coordinates and exact matching after authenticating both sources."""
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
    actual = paired_coordinates(
        context.inputs.graph,
        real.model,
        shuffled.model,
        (
            TrainingActivity(indices, seed.positions, activity[0]),
            TrainingActivity(indices, seed.positions, activity[1]),
        ),
    )
    stored = seed.coordinates.array(len(indices), len(COORDINATES))
    validate_coordinates(actual, stored)
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


def recover(path: Path, root: Path, seal: CalibrationSeal) -> CalibrationResult:
    """Authenticate the sealed project and independently replay every decision."""
    context = load_context(root, seal)
    check_files(root, seal)
    with ZipFile(path) as archive:
        if archive.namelist() != ["calibration.json"] or archive.testzip() is not None:
            message = "Calibration result ZIP membership or CRC differs"
            raise ValueError(message)
        payload = archive.read("calibration.json")
    result = CalibrationResult.model_validate_json(payload)
    if (
        payload != result.model_dump_json().encode()
        or result.seal != seal
        or sha256(context.tokens.tobytes()).hexdigest() != seal.training_tokens_sha256
        or sha256(
            np.asarray(context.split.offsets, dtype=np.int64).tobytes()
        ).hexdigest()
        != seal.training_offsets_sha256
    ):
        message = "Calibration canonical result or trusted seal differs"
        raise ValueError(message)
    for seed in result.seeds:
        validate_seed(context, seed)
    check_files(root, seal)
    return result
