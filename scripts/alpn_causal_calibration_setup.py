"""Standalone stdlib bootstrap for the sealed calibration project."""

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import TypeAlias, cast
from zipfile import ZipFile

Json: TypeAlias = str | int | float | bool | list["Json"] | dict[str, "Json"] | None
PROTOCOL_SHA256 = "6b134d0e928caacf13900ccf793a65954e02dadef6e40eee014fa7feb9bb0dba"
FIELDS = ("graph", "corpus", "port_manifest", "port_arrays", "group_manifest")
PROJECT = "flyrl-0.1.0"
SOURCE_COUNT = 12
SOURCE_MEMBERS = {
    "activations-001.png",
    "activations-001.svg",
    "activations-002.png",
    "activations-002.svg",
    "activations.json",
    "activations.npz",
    "checkpoint.npz",
    "initial.json",
    "report.json",
    "runtime.json",
}


def mapping(value: Json) -> dict[str, Json]:
    """Require a JSON object at the untrusted boundary."""
    if not isinstance(value, dict):
        message = "Expected JSON object"
        raise TypeError(message)
    return value


def document(payload: bytes) -> dict[str, Json]:
    """Decode JSON without permitting duplicate keys."""

    def unique(pairs: list[tuple[str, Json]]) -> dict[str, Json]:
        result = dict(pairs)
        if len(result) != len(pairs):
            message = "Duplicate JSON key"
            raise ValueError(message)
        return result

    return mapping(cast("Json", json.loads(payload, object_pairs_hook=unique)))


def flat(value: Json) -> str:
    """Accept only a single ordinary portable filename."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(c in value for c in ("/", "\\", "\x00", ":"))
    ):
        message = "Nested or invalid filename"
        raise ValueError(message)
    return value


def identity(payload: bytes) -> dict[str, Json]:
    """Bind exact payload bytes and length, independently of ZIP metadata."""
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def protocol_files(payloads: dict[str, bytes]) -> dict[str, Json]:
    """Authenticate the frozen protocol and every base artifact."""
    protocol_bytes = payloads["protocol.json"]
    if hashlib.sha256(protocol_bytes).hexdigest() != PROTOCOL_SHA256:
        message = "Frozen protocol hash differs"
        raise ValueError(message)
    protocol = document(protocol_bytes)
    for field in FIELDS:
        name = flat(protocol[field])
        if identity(payloads[name])["sha256"] != protocol[f"{field}_sha256"]:
            message = "Base artifact hash differs"
            raise ValueError(message)
    return protocol


def source_files(protocol: dict[str, Json], uploads: Path) -> dict[str, Path]:
    """Verify twelve original uploaded archives without parsing their metrics."""
    records = protocol["checkpoints"]
    if not isinstance(records, list) or len(records) != SOURCE_COUNT:
        message = "Expected twelve source archives"
        raise ValueError(message)
    result: dict[str, Path] = {}
    for index, record in enumerate(records):
        source = mapping(record)
        seed, wiring = 7 + index // 2, ("real", "shuffled")[index % 2]
        name = flat(source["filename"])
        if (source["seed"], source["wiring"], name) != (
            seed,
            wiring,
            f"seed-{seed}-{wiring}-random_random.zip",
        ):
            message = "Source archive order or membership differs"
            raise ValueError(message)
        path = uploads / name
        if identity(path.read_bytes()) != {
            "sha256": source["archive_sha256"],
            "size": source["archive_bytes"],
        }:
            message = "Source archive hash or size differs"
            raise ValueError(message)
        with ZipFile(path) as archive:
            members = archive.namelist()
            if (
                len(members) != len(set(members))
                or set(members) != SOURCE_MEMBERS
                or any(flat(name) != name for name in members)
                or archive.testzip() is not None
            ):
                message = "Source archive membership or CRC differs"
                raise ValueError(message)
        result[name] = path
    return result


def extract_source(source: Path, root: Path) -> Path:
    """Reject links, traversal, duplicate files and special tar members."""
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        seen: set[str] = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or not path.parts
                or path.parts[0] != PROJECT
                or str(path) in seen
                or not (member.isfile() or member.isdir())
            ):
                message = "Unsafe source tar membership"
                raise ValueError(message)
            seen.add(str(path))
        archive.extractall(root, members=members, filter="data")
    return root / PROJECT


def stage(payloads: dict[str, bytes], uploads: Path, root: Path) -> Path:
    """Construct the identical isolated project for local sealing and remote setup."""
    protocol = protocol_files(payloads)
    sources = source_files(protocol, uploads)
    root.mkdir(exist_ok=False)
    tar = root / "source.tar.gz"
    _ = tar.write_bytes(payloads[tar.name])
    project = extract_source(tar, root)
    data = project / "data/central_connectome"
    data.mkdir(parents=True, exist_ok=True)
    for field in FIELDS:
        name = flat(protocol[field])
        target = project / name if field == "corpus" else data / name
        _ = target.write_bytes(payloads[name])
    _ = (project / "protocol.json").write_bytes(payloads["protocol.json"])
    if "seal.json" in payloads:
        _ = (project / "seal.json").write_bytes(payloads["seal.json"])
    (project / "sources").mkdir()
    for name, path in sources.items():
        _ = shutil.copyfile(path, project / "sources" / name)
    return project


def unpack(bundle: Path, expected_sha256: str) -> dict[str, bytes]:
    """Verify transport identity, exact flat membership, CRCs, sizes and hashes."""
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != expected_sha256:
        message = "Input bundle hash differs"
        raise ValueError(message)
    with ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or archive.testzip() is not None:
            message = "Input bundle duplicate membership or CRC differs"
            raise ValueError(message)
        payloads = {flat(name): archive.read(name) for name in names}
    manifest = document(payloads.pop("manifest.json"))
    if manifest != {name: identity(payload) for name, payload in payloads.items()}:
        message = "Input payload membership, hash or size differs"
        raise ValueError(message)
    protocol = protocol_files(payloads)
    expected = {"source.tar.gz", "protocol.json", "seal.json"} | {
        flat(protocol[field]) for field in FIELDS
    }
    if set(payloads) != expected:
        message = "Input bundle membership differs"
        raise ValueError(message)
    return payloads


def setup(bundle: Path, uploads: Path, root: Path, expected_sha256: str) -> Path:
    """Materialize authenticated inputs without installation or hardcoded roots."""
    return stage(unpack(bundle, expected_sha256), uploads, root)


def main() -> None:
    """Install extracted code, then invoke its CPU-only seal/context boundary."""
    bundle, uploads, root, digest = sys.argv[1:]
    project = setup(Path(bundle), Path(uploads), Path(root), digest)
    with (project / "calibration-setup.log").open("x") as log:
        _ = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[language]"],
            cwd=project,
            stdout=log,
            stderr=log,
            check=True,
        )
        _ = subprocess.run(
            [
                sys.executable,
                "-c",
                """from pathlib import Path
from scripts.alpn_causal_calibration_types import CalibrationSeal
from scripts.alpn_causal_calibration_support import load_context
load_context(Path.cwd(), CalibrationSeal.model_validate_json(
    Path('seal.json').read_bytes()))
""",
            ],
            cwd=project,
            stdout=log,
            stderr=log,
            check=True,
        )
    _ = sys.stdout.write("ALPN_CALIBRATION_STATUS status=setup_ready\n")


if __name__ == "__main__":
    main()
