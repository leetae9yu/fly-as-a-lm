"""Local replay rejects tampering and never opens fresh or heldout corpus arrays."""

from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_STORED, BadZipFile, ZipFile

import pytest

from scripts import alpn_causal_calibration_worker as worker
from scripts.alpn_causal_calibration_recovery import (
    package_result,
    recover_package,
    validate_result,
)
from scripts.alpn_causal_calibration_support import Context, check_files, create_seal
from scripts.alpn_causal_calibration_types import (
    CalibrationResult,
    FileIdentity,
    Matrix,
)
from scripts.connectome_source import file_digest
from tests.test_alpn_causal_calibration_worker import restoration, synthetic_context


@pytest.fixture(scope="module")
def completed(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Context, CalibrationResult, Path]:
    root = tmp_path_factory.mktemp("calibration")
    context, models = synthetic_context(root)
    archive = root / "result.zip"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(worker, "restore_checkpoint", restoration(models))
        result = worker.run_calibration(context, archive)
    return context, result, archive


def test_deterministic_package_and_independent_cpu_recovery(
    completed: tuple[Context, CalibrationResult, Path],
    tmp_path: Path,
) -> None:
    context, result, archive = completed
    assert recover_package(archive, context) == result
    second = tmp_path / "same.zip"
    package_result(second, result, context)
    assert second.read_bytes() == archive.read_bytes()
    with pytest.raises(FileExistsError):
        package_result(second, result, context)
    assert second.read_bytes() == archive.read_bytes()


@pytest.mark.parametrize("fault", ["missing", "duplicate", "reordered", "status"])
def test_exact_six_seeds_and_aggregate_status(
    completed: tuple[Context, CalibrationResult, Path],
    fault: str,
) -> None:
    _, result, _ = completed
    seeds = result.seeds
    if fault == "missing":
        seeds = seeds[:-1]
    elif fault == "duplicate":
        seeds = (*seeds[:-1], seeds[0])
    elif fault == "reordered":
        seeds = tuple(reversed(seeds))
    changed = result.model_copy(
        update={
            "seeds": seeds,
            "status": "balanced" if fault == "status" else result.status,
        }
    )
    with pytest.raises(ValueError, match="completeness"):
        _ = CalibrationResult.model_validate_json(changed.model_dump_json())


@pytest.mark.parametrize(
    "fault",
    [
        "indices",
        "hash",
        "coordinate",
        "activity",
        "controls",
        "cost",
        "balance",
        "overlap",
        "source",
        "runtime",
        "immutability",
    ],
)
def test_all_seed_evidence_is_recomputed(
    completed: tuple[Context, CalibrationResult, Path],
    fault: str,
) -> None:
    context, result, _ = completed
    seed = result.seeds[0]
    if fault == "indices":
        seed = seed.model_copy(update={"indices": (1, 2, 4)})
    elif fault in {"hash", "coordinate"}:
        values = seed.coordinates.array(3, 16)
        values[0, 4] += 0.1
        matrix = (
            Matrix.capture(values)
            if fault == "coordinate"
            else seed.coordinates.model_copy(update={"sha256": "a" * 64})
        )
        seed = seed.model_copy(update={"coordinates": matrix})
    elif fault == "activity":
        values = seed.activity[0].array(3, 3)
        values[0, 0] = 2.0
        seed = seed.model_copy(
            update={"activity": (Matrix.capture(values), seed.activity[1])}
        )
    elif fault in {"source", "runtime", "immutability"}:
        source = seed.sources[0]
        if fault == "source":
            source = source.model_copy(update={"source": seed.sources[1].source})
        elif fault == "runtime":
            source = source.model_copy(
                update={
                    "runtime": source.runtime.model_copy(update={"torch": "changed"})
                }
            )
        else:
            source = source.model_copy(
                update={"after": source.after.model_copy(update={"rng": "a" * 64})}
            )
        seed = seed.model_copy(update={"sources": (source, seed.sources[1])})
    else:
        matching = seed.matching
        control = matching.controls[0]
        if fault == "controls":
            control = control.model_copy(update={"control_indices": (4,)})
        elif fault == "cost":
            control = control.model_copy(update={"pair_costs": (0.0,)})
        elif fault == "balance":
            balance = control.balance[0].model_copy(update={"ks": 0.123})
            control = control.model_copy(
                update={"balance": (balance, *control.balance[1:])}
            )
        matching = matching.model_copy(
            update={
                "controls": (control, *matching.controls[1:]),
                "overlaps": () if fault == "overlap" else matching.overlaps,
            }
        )
        seed = seed.model_copy(update={"matching": matching})
    changed = result.model_copy(update={"seeds": (seed, *result.seeds[1:])})
    with pytest.raises(ValueError, match="Calibration"):
        validate_result(changed, context)


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "duplicate", "nested", "crc", "json_duplicate"]
)
def test_archive_membership_crc_and_duplicate_json(
    completed: tuple[Context, CalibrationResult, Path],
    tmp_path: Path,
    fault: str,
) -> None:
    context, result, _ = completed
    path = tmp_path / "invalid.zip"
    payload = result.model_dump_json().encode()
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        if fault != "missing":
            if fault == "json_duplicate":
                payload = b'{"status":"balanced",' + payload[1:]
            archive.writestr(
                "nested/calibration.json" if fault == "nested" else "calibration.json",
                payload,
            )
        if fault == "extra":
            archive.writestr("extra", b"extra")
        if fault == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("calibration.json", payload)
    if fault == "crc":
        raw = bytearray(path.read_bytes())
        raw[30 + len("calibration.json") + 10] ^= 1
        _ = path.write_bytes(raw)
    with pytest.raises((ValueError, BadZipFile)):
        _ = recover_package(path, context)


@pytest.mark.parametrize(
    "name",
    ["protocol.json", "scripts/worker.py", "corpus.npz", "sources/checkpoint.zip"],
)
def test_local_hash_revalidation_covers_every_input_domain(
    completed: tuple[Context, CalibrationResult, Path],
    tmp_path: Path,
    name: str,
) -> None:
    context, _, _ = completed
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(b"original")
    identity = FileIdentity(
        path=name, sha256=file_digest(path), size=path.stat().st_size
    )
    seal = context.seal.model_copy(update={"files": (identity,)})
    check_files(tmp_path, seal)
    _ = path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash"):
        check_files(tmp_path, seal)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        check_files(tmp_path, seal)


def test_group_membership_and_training_identity_cannot_be_replaced(
    completed: tuple[Context, CalibrationResult, Path],
) -> None:
    context, result, _ = completed
    changed = replace(context, anatomy=replace(context.anatomy, alpn=(1,)))
    with pytest.raises(ValueError, match="controls"):
        validate_result(result, changed)
    changed = replace(context, tokens=context.tokens[:-1])
    with pytest.raises(ValueError, match="training"):
        validate_result(result, changed)


def test_protocol_pin_fails_before_any_data_or_checkpoint_load(tmp_path: Path) -> None:
    _ = (tmp_path / "protocol.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="protocol hash"):
        _ = create_seal(tmp_path)
