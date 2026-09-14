"""Filesystem and process contracts, with real sealing but no calibration execution."""

import os
import select
import subprocess
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast
from zipfile import BadZipFile, ZipFile

import pytest

from scripts import alpn_causal_calibration_bundle as bundle
from scripts import alpn_causal_calibration_execute as execute
from scripts import alpn_causal_calibration_setup as setup
from scripts import alpn_causal_calibration_support as support
from scripts.alpn_causal_calibration_types import CalibrationSeal
from scripts.connectome_source import file_digest
from tests.test_alpn_causal_calibration_support import TOKENIZER_HASH, training_corpus

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@dataclass(frozen=True, slots=True)
class Inputs:
    repository: Path
    source: Path
    uploads: Path

    build: Callable[[Path], str]


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Inputs:
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("pyproject.toml", "uv.lock", "scripts/worker.py", "flyrl/model.py"):
        path = repo / name
        path.parent.mkdir(exist_ok=True)
        _ = path.write_bytes(b"# synthetic source\n")
    source = tmp_path / "source.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        archive.add(repo, arcname="flyrl-0.1.0")
    training = tmp_path / "training"
    training.mkdir()
    protocol = training_corpus(training)
    payloads = {"corpus.npz": (training / "corpus.npz").read_bytes()}
    updates: dict[str, str] = {}
    for field in ("graph", "port_manifest", "port_arrays", "group_manifest"):
        name = cast("str", getattr(protocol, field))
        payloads[name] = field.encode()
        path = training / name
        _ = path.write_bytes(payloads[name])
        updates[f"{field}_sha256"] = file_digest(path)
    payloads["protocol.json"] = (
        protocol.model_copy(update=updates).model_dump_json().encode()
    )
    protocol_path = training / "protocol.json"
    _ = protocol_path.write_bytes(payloads["protocol.json"])
    for module in (setup, support):
        monkeypatch.setattr(module, "PROTOCOL_SHA256", file_digest(protocol_path))
    monkeypatch.setattr(
        support,
        "old_training",
        partial(support.old_training, tokenizer_sha256=TOKENIZER_HASH),
    )
    base = tmp_path / "base.zip"
    with ZipFile(base, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    uploads = training / "sources"
    return Inputs(
        repository=repo,
        source=source,
        uploads=uploads,
        build=partial(bundle.build_bundle, repo, source, base, uploads),
    )


def test_deterministic_bundle_and_remote_layout(inputs: Inputs, tmp_path: Path) -> None:
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    digest = inputs.build(first)
    assert inputs.build(second) == digest
    assert first.read_bytes() == second.read_bytes()
    project = setup.setup(first, inputs.uploads, tmp_path / "remote", digest)
    seal = CalibrationSeal.model_validate_json((project / "seal.json").read_bytes())
    assert support.create_seal(project) == seal
    with ZipFile(first) as archive:
        assert len(archive.namelist()) == 9
        assert all(
            "/" not in name and not name.endswith(".zip") for name in archive.namelist()
        )


@pytest.mark.parametrize("fault", ["changed", "added", "removed", "lock"])
def test_sdist_freshness(inputs: Inputs, fault: str) -> None:
    path = inputs.repository / ("uv.lock" if fault == "lock" else "scripts/worker.py")
    if fault == "added":
        path = path.with_name("new.py")
    if fault == "removed":
        path.unlink()
    else:
        _ = path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="stale"):
        bundle.fresh_source(inputs.repository, inputs.source)


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "nested", "hash", "size", "crc", "duplicate"]
)
def test_input_rejection(inputs: Inputs, tmp_path: Path, fault: str) -> None:
    original = tmp_path / "original.zip"
    _ = inputs.build(original)
    with ZipFile(original) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    if fault == "missing":
        del payloads["seal.json"]
    elif fault in {"extra", "nested"}:
        payloads["extra" if fault == "extra" else "../escape"] = b"extra"
    elif fault == "hash":
        payloads["seal.json"] += b" "
    elif fault == "size":
        payloads["manifest.json"] = payloads["manifest.json"].replace(
            b'"size":', b'"size":1', 1
        )
    damaged = tmp_path / "damaged.zip"
    with ZipFile(damaged, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        if fault == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr("seal.json", payloads["seal.json"])
    if fault == "crc":
        raw = damaged.read_bytes()
        _ = damaged.write_bytes(raw.replace(b'"format"', b'"Format"', 1))
    with pytest.raises(
        (ValueError, BadZipFile), match=r"membership|hash|size|Nested|CRC"
    ):
        _ = setup.unpack(damaged, file_digest(damaged))
    with pytest.raises(ValueError, match="bundle hash"):
        _ = setup.unpack(original, "0" * 64)


@pytest.mark.parametrize("fault", ["missing", "bytes", "length"])
def test_uploaded_archive_rejection(inputs: Inputs, tmp_path: Path, fault: str) -> None:
    bundle = tmp_path / "input.zip"
    digest = inputs.build(bundle)
    source = next(inputs.uploads.glob("*.zip"))
    if fault == "missing":
        source.unlink()
    else:
        payload = source.read_bytes()
        _ = source.write_bytes(
            payload + b"x" if fault == "length" else b"x" + payload[1:]
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        _ = setup.setup(bundle, inputs.uploads, tmp_path / "remote", digest)


@pytest.mark.parametrize("name", ["../escape", "/escape", "flyrl-0.1.0/../../escape"])
def test_unsafe_source_tar(tmp_path: Path, name: str) -> None:
    source = tmp_path / "unsafe.tar"
    with tarfile.open(source, "w") as archive:
        archive.addfile(tarfile.TarInfo(name))
    with pytest.raises(ValueError, match="Unsafe"):
        _ = setup.extract_source(source, tmp_path / "extract")


def test_execute_command_and_event_hold(
    tmp_path: Path,
) -> None:
    destination, ack = tmp_path / "result.zip", tmp_path / "download.complete"
    assert execute.command(tmp_path, destination) == [
        sys.executable,
        "-u",
        "-m",
        "scripts.alpn_causal_calibration_worker",
        str(tmp_path),
        str(tmp_path / "seal.json"),
        str(destination),
    ]
    with pytest.raises(TimeoutError):
        execute.hold(ack, lambda: None, timeout=0)
    execute.hold(ack, ack.touch, timeout=1)
    with pytest.raises(FileExistsError):
        _ = execute.execute(tmp_path, destination, ack)


@pytest.mark.parametrize("returncode", [0, 3])
def test_real_execute_surface_filters_and_holds(
    tmp_path: Path, returncode: int
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    _ = (scripts / "__init__.py").write_text("")
    _ = (scripts / "alpn_causal_calibration_worker.py").write_text(
        "\n".join(
            (
                "import sys",
                'print("PRIVATE_DIAGNOSTIC")',
                'print("ALPN_CALIBRATION_SEED_READY seed=7")',
                'print("ALPN_CALIBRATION_STATUS status=failed")',
                f"sys.exit({returncode})",
            )
        )
    )
    ack = tmp_path / "download.complete"
    with subprocess.Popen[str](
        [
            sys.executable,
            str(SCRIPTS / "alpn_causal_calibration_execute.py"),
            str(tmp_path),
            str(tmp_path / "result.zip"),
            str(ack),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            assert process.stdout is not None
            descriptor = process.stdout.fileno()
            captured = b""
            while b"ALPN_CALIBRATION_HOLD_READY\n" not in captured:
                assert select.select([descriptor], [], [], 10)[0]
                chunk = os.read(descriptor, 4096)
                assert chunk
                captured += chunk
            assert captured.decode().splitlines() == [
                "ALPN_CALIBRATION_SEED_READY seed=7",
                "ALPN_CALIBRATION_STATUS status=failed",
                "ALPN_CALIBRATION_HOLD_READY",
            ]
            assert process.poll() is None
            ack.touch()
            output, error = process.communicate(timeout=10)
            assert output == f"ALPN_CALIBRATION_EXEC_EXIT {returncode}\n"
            assert not error
            assert process.returncode == returncode
        finally:
            if process.poll() is None:
                process.kill()
    assert "PRIVATE_DIAGNOSTIC" in (tmp_path / "calibration-execute.log").read_text()
