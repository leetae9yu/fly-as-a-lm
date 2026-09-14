"""Sealed Tesla-T4 calibration only; never load fresh text, fit heads or score."""

import argparse
import platform
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

import numpy as np
import torch
from pydantic import TypeAdapter

from flyrl.ar_learning import ARLearner
from scripts.alpn_causal_calibration import (
    CalibrationConfig,
    paired_coordinates,
    training_activity,
)
from scripts.alpn_causal_calibration_recovery import package_result
from scripts.alpn_causal_calibration_support import (
    Context,
    check_files,
    load_context,
    selected_indices,
    selection,
    snapshot,
)
from scripts.alpn_causal_calibration_types import (
    SEEDS,
    CalibrationResult,
    CalibrationSeal,
    MatchingRecord,
    Matrix,
    Record,
    Runtime,
    SeedCalibration,
    SourceEvidence,
    SourceIdentity,
)
from scripts.alpn_causal_groups import COORDINATES, match_controls
from scripts.alpn_causal_source_archive import read_source_payloads
from scripts.regional_probe_protocol import ProbeSourceCheckpoint
from scripts.regional_probe_worker_support import restore_source


def restore_checkpoint(
    context: Context, source: ProbeSourceCheckpoint
) -> tuple[ARLearner, Runtime]:
    """The sole device/checkpoint boundary; restoration enforces the actual Tesla T4."""
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != "Tesla T4":
        message = "Calibration requires an actual Tesla T4"
        raise ValueError(message)
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        payloads = read_source_payloads(context.root / "sources" / source.filename)
        for name, payload in payloads.items():
            _ = (directory / name).write_bytes(payload)
        learner, _ = restore_source(directory, source, context.inputs)
    runtime = Runtime.model_validate(
        {
            "gpu": torch.cuda.get_device_name(learner.model.weight.device),
            "device": str(learner.model.weight.device),
            "dtype": str(learner.model.weight.dtype),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        }
    )
    return learner, runtime


def calibrate_seed(context: Context, seed: int) -> SeedCalibration:
    """Extract only ALPN plus eligible nodes, and match the five unchanged draws."""
    started = time.perf_counter()
    inputs = context.inputs
    plan = next(plan for plan in inputs.groups.seeds if plan.seed == seed)
    sources = tuple(
        source for source in inputs.protocol.checkpoints if source.seed == seed
    )
    if tuple(source.wiring for source in sources) != ("real", "shuffled"):
        message = "Calibration requires an exact real/shuffled source pair"
        raise ValueError(message)
    indices = selected_indices(context, seed)
    restored = tuple(restore_checkpoint(context, source) for source in sources)
    learners = tuple(item[0] for item in restored)
    before = tuple(snapshot(learner) for learner in learners)
    for learner in learners:
        if (
            TypeAdapter(tuple[int, ...]).validate_python(
                learner.model.sensory.cpu().numpy().tolist()
            )
            != plan.sensory_indices
            or TypeAdapter(tuple[int, ...]).validate_python(
                learner.model.ports.cpu().numpy().tolist()
            )
            != plan.trained_readout_indices
        ):
            message = "Calibration paired ports differ"
            raise ValueError(message)
    config = CalibrationConfig.model_validate(inputs.protocol.extraction.model_dump())
    real, shuffled = (
        training_activity(
            learner.model, context.tokens, context.split, indices, config=config
        )
        for learner in learners
    )
    coordinates = paired_coordinates(
        inputs.graph, learners[0].model, learners[1].model, (real, shuffled)
    )
    full = np.zeros((len(inputs.graph.node_ids), len(COORDINATES)), dtype=np.float64)
    full[list(indices)] = coordinates
    draws = tuple(group.indices for group in plan.groups if group.name == "alpn")
    if tuple(group.draw for group in plan.groups if group.name == "alpn") != tuple(
        range(5)
    ):
        message = "Calibration frozen ALPN draws differ"
        raise ValueError(message)
    matching = MatchingRecord.capture(
        match_controls(inputs.graph.node_ids, selection(context, seed), draws, full)
    )
    evidence = tuple(
        SourceEvidence(
            source=SourceIdentity.model_validate(source.model_dump()),
            runtime=runtime,
            before=saved,
            after=snapshot(learner),
        )
        for source, (learner, runtime), saved in zip(
            sources, restored, before, strict=True
        )
    )
    return SeedCalibration.model_validate(
        {
            "seed": seed,
            "indices": indices,
            "positions": real.positions,
            "activity": (Matrix.capture(real.values), Matrix.capture(shuffled.values)),
            "coordinates": Matrix.capture(coordinates),
            "matching": matching,
            "sources": evidence,
            "seconds": time.perf_counter() - started,
        }
    )


def run_calibration(context: Context, destination: Path) -> CalibrationResult:
    """Publish one CRC-tested result only after all seeds and independent recovery."""
    progress = sys.stdout
    seeds: list[SeedCalibration] = []
    with (
        (context.root / "calibration-worker.log").open("x") as log,
        redirect_stdout(log),
        redirect_stderr(log),
    ):
        check_files(context.root, context.seal)
        for seed in SEEDS:
            seeds.append(calibrate_seed(context, seed))
            check_files(context.root, context.seal)
            _ = progress.write(f"ALPN_CALIBRATION_SEED_READY seed={seed}\n")
            _ = progress.flush()
        result = CalibrationResult(
            seal=context.seal,
            seeds=tuple(seeds),
            status="balanced"
            if all(seed.matching.status == "balanced" for seed in seeds)
            else "insufficient_common_support",
        )
        package_result(destination, result, context)
    _ = progress.write(f"ALPN_CALIBRATION_STATUS status={result.status}\n")
    _ = progress.flush()
    return result


def main() -> None:
    """Run against a separately sealed project; failure details stay off stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("root", type=Path)
    _ = parser.add_argument("seal", type=Path)
    _ = parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    paths = TypePaths.model_validate(vars(arguments))
    try:
        seal = CalibrationSeal.model_validate_json(paths.seal.read_bytes())
        _ = run_calibration(load_context(paths.root, seal), paths.destination)
    except (ValueError, RuntimeError, OSError, KeyError, BadZipFile):
        _ = sys.stderr.write(traceback.format_exc())
        _ = sys.stdout.write("ALPN_CALIBRATION_STATUS status=failed\n")
        raise SystemExit(1) from None


class TypePaths(Record):
    """Typed command-line paths."""

    root: Path
    seal: Path
    destination: Path


if __name__ == "__main__":
    main()
