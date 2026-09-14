"""Real quality output recovery and adversarial archive transactions."""

import hashlib
import io
import json
import stat
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np
import pytest
from numpy.lib.format import write_array
from numpy.lib.npyio import NpzFile
from pydantic import TypeAdapter
from typer.testing import CliRunner

from flyrl.ar_config import parameter_identity_json
from flyrl.quality_pilot_schema import QualityProtocol, SuccessLabel
from flyrl.quality_recovery import decide
from scripts.connectome_source import file_digest
from scripts.recover_quality_pilot import app, recover
from scripts.run_quality_pilot import run
from tests.test_quality_pilot import protocol as protocol_fixture

if TYPE_CHECKING:
    from numpy import generic


protocol = protocol_fixture


@pytest.fixture
def output(protocol: QualityProtocol, tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "run"
    _ = run(protocol, root)
    trusted = tmp_path / "QUALITY_PILOT.json"
    _ = trusted.write_text(protocol.model_dump_json())
    # Recovery must depend on the prepared artifact, not the ignored raw sources.
    (tmp_path / "train.txt").unlink()
    (tmp_path / "heldout.txt").unlink()
    return root, trusted


def package(root: Path, destination: Path) -> None:
    with ZipFile(destination, "w", compression=ZIP_STORED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                member = ZipInfo(path.relative_to(root).as_posix())
                member.external_attr = (stat.S_IFREG | 0o600) << 16
                archive.writestr(member, path.read_bytes())


def alter_json(path: Path, change: Callable[[dict[str, object]], None]) -> None:
    data = cast("dict[str, object]", json.loads(path.read_text()))
    change(data)
    _ = path.write_text(json.dumps(data))


def alter_report(root: Path, change: Callable[[dict[str, object]], None]) -> None:
    for name in ("progress.json", "report.json"):
        alter_json(root / name, change)


def test_real_recovery_and_cli(output: tuple[Path, Path], tmp_path: Path) -> None:
    root, trusted = output
    archive = tmp_path / "result.zip"
    package(root, archive)
    duplicate = tmp_path / "again.zip"
    package(root, duplicate)
    assert archive.read_bytes() == duplicate.read_bytes()
    destination = tmp_path / "recovered"
    result = recover(archive, trusted, destination)
    assert result.recovery_gates_complete
    assert not result.checkpoint_continuation_verified
    assert result.generation_reproduction == "verified locally (CPU)"
    assert result.success_label != SuccessLabel.IMPROVED
    assert (destination / "recovery.json").is_file()
    assert (
        result.model_dump_json()
        == type(result)
        .model_validate_json((destination / "recovery.json").read_text(), strict=True)
        .model_dump_json()
    )
    cli = CliRunner().invoke(
        app,
        [
            "--archive",
            str(archive),
            "--protocol",
            str(trusted),
            "--destination",
            str(tmp_path / "cli"),
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert "QUALITY_RECOVERY_VERIFIED" in cli.stdout
    with pytest.raises(FileExistsError):
        _ = recover(archive, trusted, destination)


@pytest.mark.parametrize(
    "member",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "a\\\\b",
        "a/./b",
        "a//b",
        "symlink",
        "extra.json",
        "report.json",
    ],
)
def test_unsafe_and_extra_members(
    output: tuple[Path, Path], tmp_path: Path, member: str
) -> None:
    root, trusted = output
    archive = tmp_path / "bad.zip"
    package(root, archive)
    with ZipFile(archive, "a") as zipped:
        info = ZipInfo(member)
        info.external_attr = (
            stat.S_IFLNK if member == "symlink" else stat.S_IFREG
        ) << 16
        if member == "report.json":
            with pytest.warns(UserWarning, match="Duplicate"):
                zipped.writestr(info, (root / "report.json").read_bytes())
        else:
            zipped.writestr(info, b"bad")
    with pytest.raises(ValueError, match=r"archive|Archive|Duplicate|Invalid JSON"):
        _ = recover(archive, trusted, tmp_path / "destination")
    assert not (tmp_path / "destination").exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "torn",
        "updates",
        "selected",
        "milestones",
        "summaries",
        "repeat",
        "success",
        "checkpoint",
        "generation",
        "artifact",
        "protocol",
        "nonfinite",
        "story",
    ],
)
def test_corrupt_evidence(output: tuple[Path, Path], tmp_path: Path, kind: str) -> None:
    root, trusted = output
    changes: dict[str, object] = {
        "updates": 3,
        "selected_update": 0,
        "validation": [],
        "summaries": [],
        "repeat_collapse_count": 99,
        "success_label": SuccessLabel.IMPROVED,
        "protocol_sha256": "0" * 64,
    }
    fields = {
        "selected": "selected_update",
        "milestones": "validation",
        "repeat": "repeat_collapse_count",
        "success": "success_label",
        "protocol": "protocol_sha256",
    }
    if kind in {"missing", "checkpoint"}:
        path = root / "checkpoint-000000.npz"
        if kind == "missing":
            path.unlink()
        else:
            _ = path.write_bytes(b"broken")
    elif kind == "torn":
        alter_json(root / "progress.json", lambda data: data.update(updates=3))
    elif kind == "artifact":
        _ = next(root.rglob("*.png")).write_bytes(b"broken")
    elif kind in {"generation", "nonfinite", "story"}:

        def change(data: dict[str, object]) -> None:
            results = cast("list[dict[str, object]]", data["completed"])
            if kind == "generation":
                records = cast("list[dict[str, object]]", results[0]["generation"])
                records[0]["seed"] = 123
            else:
                metrics = cast("dict[str, object]", results[0]["test"])
                stories = cast("list[dict[str, object]]", metrics["per_story"])
                stories[0]["loss_sum" if kind == "nonfinite" else "count"] = (
                    float("nan") if kind == "nonfinite" else 999
                )
            _ = (root / f"result-{results[0]['update']:06d}.json").write_text(
                json.dumps(results[0])
            )

        alter_report(root, change)
    else:
        field = fields.get(kind, kind)
        alter_report(root, lambda data: data.update({field: changes[field]}))
    archive = tmp_path / "bad.zip"
    package(root, archive)
    with pytest.raises(ValueError, match=r"mismatch|milestones|missing|Torn|finite"):
        _ = recover(archive, trusted, tmp_path / "destination")
    assert not (tmp_path / "destination").exists()


def replace_array(path: Path, key: str, value: object) -> None:
    with path.open("rb") as stream:
        arrays: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with arrays:
            values = {name: arrays[name] for name in arrays.files}
    values[key] = np.asarray(value)
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, array in values.items():
            with archive.open(f"{name}.npy", "w") as member:
                write_array(member, array, allow_pickle=False)
    _ = path.write_bytes(buffer.getvalue())


def rehash_artifact(root: Path, path: Path) -> None:
    def change(data: dict[str, object]) -> None:
        results = cast("list[dict[str, object]]", data["completed"])
        for result in results:
            artifacts = cast("list[dict[str, object]]", result["artifacts"])
            for artifact in artifacts:
                if artifact["path"] == path.relative_to(root).as_posix():
                    artifact["sha256"] = file_digest(path)
            update = cast("int", result["update"])
            _ = (root / f"result-{update:06d}.json").write_text(json.dumps(result))

    alter_report(root, change)


@pytest.mark.parametrize(
    "kind",
    [
        "states",
        "probabilities",
        "selected_indices",
        "generated_ids",
        "metadata",
        "png",
        "svg",
    ],
)
def test_rehashed_activation_corruption(
    output: tuple[Path, Path], tmp_path: Path, kind: str
) -> None:
    root, trusted = output
    if kind in {"png", "svg"}:
        path = next(root.rglob(f"*.{kind}"))
        _ = path.write_bytes(b"broken")
    elif kind == "metadata":
        path = next(root.rglob("activations.json"))
        alter_json(path, lambda data: data.update(parameter_fingerprint="0" * 64))
    else:
        path = next(root.rglob("activations.npz"))
        replace_array(path, kind, np.asarray([999], dtype=np.int64))
    rehash_artifact(root, path)
    archive = tmp_path / "bad.zip"
    package(root, archive)
    with pytest.raises(ValueError, match=r"Activation.*mismatch|figure.*invalid"):
        _ = recover(archive, trusted, tmp_path / "destination")
    assert not (tmp_path / "destination").exists()


@pytest.mark.parametrize(
    ("profile", "gates", "gain", "repeats", "expected"),
    [
        ("quality", True, 0.5, 3, SuccessLabel.IMPROVED),
        ("quality", True, 0.5, 4, SuccessLabel.UNRELIABLE),
        ("quality", False, 0.5, 0, SuccessLabel.UNRELIABLE),
        ("smoke", True, 0.5, 0, SuccessLabel.UNRELIABLE),
        ("quality", True, 0.49, 0, SuccessLabel.UNRELIABLE),
        ("quality", True, 0.0, 0, SuccessLabel.NOT_DEMONSTRATED),
    ],
)
def test_success_decision(
    profile: str, gates: bool, gain: float, repeats: int, expected: SuccessLabel
) -> None:
    assert decide(profile, gates, gain, repeats) == expected


def rehash_checkpoint(root: Path, update: int) -> None:
    path = root / f"checkpoint-{update:06d}.npz"
    digest = file_digest(path)
    alter_json(
        root / f"evaluation-{update:06d}.json",
        lambda data: data.update(checkpoint_sha256=digest),
    )

    def change(data: dict[str, object]) -> None:
        for field in ("validation", "completed"):
            for item in cast("list[dict[str, object]]", data[field]):
                if item["update"] == update:
                    item["checkpoint_sha256"] = digest
                    if field == "completed":
                        _ = (root / f"result-{update:06d}.json").write_text(
                            json.dumps(item)
                        )
        if update == data["updates"]:
            data["checkpoint_sha256"] = digest
            _ = (root / "checkpoint-latest.npz").write_bytes(path.read_bytes())

    alter_report(root, change)


@pytest.mark.parametrize(
    "kind",
    [
        "shape",
        "dtype",
        "nan",
        "adam-step",
        "adam-negative",
        "rng",
        "trace",
        "metadata",
        "extra",
        "duplicate",
        "removed",
    ],
)
def test_rehashed_checkpoint_corruption(
    output: tuple[Path, Path], tmp_path: Path, kind: str
) -> None:
    root, trusted = output
    path = root / "checkpoint-000001.npz"
    with path.open("rb") as stream:
        data: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with data:
            key = next(name for name in data.files if name.startswith("model."))
            array = data[key].copy()
            meta = cast(
                "dict[str, object]",
                json.loads(TypeAdapter(str).validate_python(data["metadata"].item())),
            )
    if kind == "trace":
        meta["trace"] = []
        replace_array(path, "metadata", json.dumps(meta))
    elif kind == "metadata":
        meta["corpus"] = "0" * 64
        replace_array(path, "metadata", json.dumps(meta))
    elif kind == "adam-step":
        replace_array(
            path,
            key.replace("model.", "adam.") + ".step",
            np.asarray(0, dtype=np.float32),
        )
    elif kind == "adam-negative":
        replace_array(
            path,
            key.replace("model.", "adam.") + ".exp_avg_sq",
            np.full(array.shape, -1, dtype=np.float32),
        )
    elif kind == "rng":
        replace_array(path, "rng", np.zeros(5056, dtype=np.uint8))
    elif kind == "duplicate":
        with (
            ZipFile(path, "a") as archive,
            pytest.warns(UserWarning, match="Duplicate"),
        ):
            archive.writestr(key + ".npy", archive.read(key + ".npy"))
    elif kind == "removed":
        with ZipFile(path) as archive:
            members = {
                name: archive.read(name)
                for name in archive.namelist()
                if name != key + ".npy"
            }
        with ZipFile(path, "w") as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
    else:
        values = {
            "shape": np.zeros(1, dtype=np.float32),
            "dtype": array.astype(np.float64),
            "nan": np.full(array.shape, np.nan, dtype=np.float32),
            "extra": np.zeros(1, dtype=np.float32),
        }
        replace_array(path, "extra" if kind == "extra" else key, values[kind])
    rehash_checkpoint(root, 1)
    archive = tmp_path / "bad.zip"
    package(root, archive)
    with pytest.raises(ValueError, match=r"Checkpoint|Invalid quality archive"):
        _ = recover(archive, trusted, tmp_path / "destination")
    assert not (tmp_path / "destination").exists()


@pytest.mark.parametrize(
    "field",
    [
        "contexts",
        "config",
        "prompt_ids",
        "token_labels",
        "readout_node_ids",
        "graph_fingerprint",
        "figures",
    ],
)
def test_activation_metadata_contract(
    output: tuple[Path, Path], tmp_path: Path, field: str
) -> None:
    root, trusted = output
    path = next(root.rglob("activations.json"))

    def change(data: dict[str, object]) -> None:
        if field == "config":
            cast("dict[str, object]", data[field])["seed"] = 123
        else:
            data[field] = "0" * 64 if field == "graph_fingerprint" else []

    alter_json(path, change)
    rehash_artifact(root, path)
    archive = tmp_path / "bad.zip"
    package(root, archive)
    with pytest.raises(ValueError, match="Activation metadata"):
        _ = recover(archive, trusted, tmp_path / "destination")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("states", 1.1),
        ("states", float("nan")),
        ("probabilities", -0.1),
        ("probabilities", float("inf")),
    ],
)
def test_activation_numeric_bounds(
    output: tuple[Path, Path], tmp_path: Path, field: str, value: float
) -> None:
    root, trusted = output
    path = next(root.rglob("activations.npz"))
    with path.open("rb") as stream:
        data: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
        with data:
            shape = data[field].shape
    replace_array(path, field, np.full(shape, value, dtype=np.float32))
    rehash_artifact(root, path)
    archive = tmp_path / "bad.zip"
    package(root, archive)
    with pytest.raises(ValueError, match="Activation nonfinite or bounds"):
        _ = recover(archive, trusted, tmp_path / "destination")


@pytest.mark.parametrize(
    "field",
    ["mode", "draw", "seed", "prompt", "prompt_ids", "token_ids", "text", "missing"],
)
def test_generation_contract(
    output: tuple[Path, Path], tmp_path: Path, field: str
) -> None:
    root, trusted = output

    def change(data: dict[str, object]) -> None:
        results = cast("list[dict[str, object]]", data["completed"])
        item = results[0]
        records = cast("list[dict[str, object]]", item["generation"])
        if field == "missing":
            _ = records.pop()
        else:
            changes: dict[str, object] = {
                "mode": "greedy",
                "draw": 99,
                "seed": 99,
                "prompt": "Wrong prompt",
                "prompt_ids": [1],
                "token_ids": [-1, -1],
                "text": "Wrong text",
            }
            records[1][field] = changes[field]
        update = cast("int", item["update"])
        _ = (root / f"result-{update:06d}.json").write_text(json.dumps(item))

    alter_report(root, change)
    archive = tmp_path / "bad.zip"
    package(root, archive)
    with pytest.raises(ValueError, match="Generation"):
        _ = recover(archive, trusted, tmp_path / "destination")


def test_crc_and_allowed_directories(output: tuple[Path, Path], tmp_path: Path) -> None:
    root, trusted = output
    archive = tmp_path / "result.zip"
    package(root, archive)
    with ZipFile(archive, "a") as zipped:
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                member = ZipInfo(path.relative_to(root).as_posix() + "/")
                member.external_attr = (stat.S_IFDIR | 0o700) << 16
                zipped.writestr(member, b"")
        info = zipped.getinfo("checkpoint-000000.npz")
    _ = recover(archive, trusted, tmp_path / "good")
    damaged = bytearray(archive.read_bytes())
    # Stored member: local header is 30 bytes, then filename and extra fields.
    offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    damaged[offset + 10] ^= 1
    _ = archive.write_bytes(damaged)
    with pytest.raises(ValueError, match="CRC"):
        _ = recover(archive, trusted, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_portable_cuda_metadata_does_not_claim_exact_cpu_resume(
    output: tuple[Path, Path], tmp_path: Path
) -> None:
    root, trusted = output
    original = QualityProtocol.model_validate_json(trusted.read_text(), strict=True)
    portable = original.model_copy(
        update={"config": original.config.model_copy(update={"device": "cuda"})}
    )
    _ = trusted.write_text(portable.model_dump_json())
    _ = (root / "protocol.json").write_text(portable.model_dump_json())
    report = cast("dict[str, object]", json.loads((root / "report.json").read_text()))
    runtime = cast("dict[str, object]", json.loads(cast("str", report["runtime"])))
    runtime.update(device="cuda:0", name="Tesla T4")
    identity = json.dumps(runtime, sort_keys=True)
    fingerprints: dict[int, str] = {}
    # Only portable metadata is synthetic: real smoke tensors/evidence remain intact.
    # This exercises local CUDA-array inspection, not actual remote GPU execution.
    for update in portable.evaluation_updates:
        path = root / f"checkpoint-{update:06d}.npz"
        with path.open("rb") as stream:
            data: NpzFile[generic] = NpzFile(stream, allow_pickle=False)
            with data:
                metadata = cast(
                    "dict[str, object]",
                    json.loads(
                        TypeAdapter(str).validate_python(data["metadata"].item())
                    ),
                )
                digest = hashlib.sha256(
                    parameter_identity_json(portable.config).encode()
                )
                for key in data.files:
                    if key.startswith("model."):
                        digest.update(key.removeprefix("model.").encode())
                        digest.update(data[key].tobytes())
        fingerprints[update] = digest.hexdigest()
        metadata.update(config=portable.config.model_dump(), runtime=identity)
        replace_array(path, "metadata", json.dumps(metadata))
        rehash_checkpoint(root, update)
    alter_report(
        root,
        lambda data: data.update(
            protocol=portable.model_dump(mode="json"),
            protocol_sha256=portable.sha256,
            runtime=identity,
        ),
    )
    for path in root.rglob("activations.json"):

        def change(data: dict[str, object]) -> None:
            data.update(
                config=portable.config.model_dump(),
                parameter_fingerprint=fingerprints[cast("int", data["updates"])],
            )

        alter_json(path, change)
        rehash_artifact(root, path)
    archive = tmp_path / "portable.zip"
    package(root, archive)
    recovered = recover(archive, trusted, tmp_path / "portable")
    assert recovered.generation_reproduction == "remote-worker gate"
    assert not recovered.checkpoint_continuation_verified
    assert recovered.recovery_gates_complete


@pytest.mark.parametrize(
    "kind", ["graph-hash", "corpus-hash", "graph-count", "corpus-count"]
)
def test_trusted_input_binding(
    output: tuple[Path, Path], tmp_path: Path, kind: str
) -> None:
    root, trusted = output
    protocol = QualityProtocol.model_validate_json(trusted.read_text(), strict=True)
    if kind.endswith("hash"):
        path = protocol.graph.path if kind.startswith("graph") else protocol.corpus.path
        _ = path.write_bytes(b"bad")
    else:

        def change(data: dict[str, object]) -> None:
            identity = cast(
                "dict[str, object]",
                data["graph" if kind.startswith("graph") else "corpus"],
            )
            identity["nodes" if kind.startswith("graph") else "tokens"] = (
                99 if kind.startswith("graph") else [99, 99, 99]
            )

        alter_json(trusted, change)
    archive = tmp_path / "result.zip"
    package(root, archive)
    with pytest.raises(ValueError, match=r"Protocol.*mismatch"):
        _ = recover(archive, trusted, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
