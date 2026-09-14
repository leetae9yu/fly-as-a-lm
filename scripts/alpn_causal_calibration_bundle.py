"""Create a deterministic sealed calibration input ZIP; source ZIPs stay separate."""

import importlib
import json
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, TypeAlias, runtime_checkable
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.alpn_causal_calibration_support import create_seal

Json: TypeAlias = str | int | float | bool | list["Json"] | dict[str, "Json"] | None


@runtime_checkable
class Bootstrap(Protocol):
    """Typed API of the adjacent independently executable stdlib bootstrap."""

    FIELDS: tuple[str, ...]
    PROJECT: str

    def document(self, payload: bytes) -> dict[str, Json]:
        """Decode strict JSON."""
        ...

    def flat(self, value: Json) -> str:
        """Validate a portable filename."""
        ...

    def identity(self, payload: bytes) -> dict[str, Json]:
        """Hash exact bytes and size."""
        ...

    def stage(self, payloads: dict[str, bytes], uploads: Path, root: Path) -> Path:
        """Construct the shared project layout."""
        ...


def load_bootstrap() -> Bootstrap:
    """Load the adjacent standalone bootstrap with a runtime-checked API."""
    module = importlib.import_module("scripts.alpn_causal_calibration_setup")
    if not isinstance(module, Bootstrap):
        message = "Bootstrap API differs"
        raise TypeError(message)
    return module


bootstrap = load_bootstrap()


def fresh_source(repository: Path, source: Path) -> None:
    """Prove all shipped package sources and build inputs equal the working tree."""
    paths = {
        str(path.relative_to(repository))
        for package in ("flyrl", "scripts", "tests")
        for path in (repository / package).rglob("*.py")
    } | {"pyproject.toml", "uv.lock"}
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        shipped = {
            member.name.removeprefix(f"{bootstrap.PROJECT}/")
            for member in members
            if member.isfile()
            and member.name.startswith(
                tuple(
                    f"{bootstrap.PROJECT}/{p}/" for p in ("flyrl", "scripts", "tests")
                )
            )
            and member.name.endswith(".py")
        } | {"pyproject.toml", "uv.lock"}
        if (
            shipped != paths
            or len({m.name for m in members}) != len(members)
            or not {
                f"{bootstrap.PROJECT}/{name}" for name in ("pyproject.toml", "uv.lock")
            }.issubset(member.name for member in members if member.isfile())
        ):
            message = "Source distribution is stale: source membership differs"
            raise ValueError(message)
        for entry in members:
            if not entry.isfile() or entry.name == f"{bootstrap.PROJECT}/PKG-INFO":
                continue
            name = entry.name.removeprefix(f"{bootstrap.PROJECT}/")
            if (
                not entry.name.startswith(f"{bootstrap.PROJECT}/")
                or ".." in Path(name).parts
                or not (repository / name).is_file()
            ):
                message = "Source distribution is stale: unexpected source file"
                raise ValueError(message)
            member = archive.extractfile(f"{bootstrap.PROJECT}/{name}")
            if member is None or member.read() != (repository / name).read_bytes():
                message = f"Source distribution is stale: {name}"
                raise ValueError(message)


def write_bundle(destination: Path, payloads: dict[str, bytes]) -> str:
    """Use fixed metadata, sorted entries and CRC verification for reproducibility."""
    manifest = json.dumps(
        {name: bootstrap.identity(payload) for name, payload in payloads.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=9) as out:
        for name, payload in sorted({**payloads, "manifest.json": manifest}.items()):
            entry = ZipInfo(bootstrap.flat(name), date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = ZIP_DEFLATED
            out.writestr(entry, payload, compresslevel=9)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            message = "Written input bundle failed CRC validation"
            raise ValueError(message)
    return str(bootstrap.identity(destination.read_bytes())["sha256"])


def build_bundle(
    repository: Path, source: Path, base: Path, uploads: Path, destination: Path
) -> str:
    """Take the exact original regional input ZIP plus a current rebuilt sdist."""
    fresh_source(repository, source)
    with ZipFile(base) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            message = "Regional input ZIP duplicate membership or CRC differs"
            raise ValueError(message)
        protocol_bytes = archive.read("protocol.json")
        protocol = bootstrap.document(protocol_bytes)
        payloads = {
            bootstrap.flat(protocol[field]): archive.read(
                bootstrap.flat(protocol[field])
            )
            for field in bootstrap.FIELDS
        }
    payloads.update(
        {"protocol.json": protocol_bytes, "source.tar.gz": source.read_bytes()}
    )
    with TemporaryDirectory() as temporary:
        project = bootstrap.stage(payloads, uploads, Path(temporary) / "setup")
        payloads["seal.json"] = create_seal(project).model_dump_json(indent=2).encode()
    return write_bundle(destination, payloads)


def main() -> None:
    """Accept repository, current sdist, regional base ZIP, uploads and output paths."""
    repository, source, base, uploads, destination = map(Path, sys.argv[1:])
    digest = build_bundle(repository, source, base, uploads, destination)
    _ = sys.stdout.write(f"ALPN_CALIBRATION_INPUT_READY sha256={digest}\n")


if __name__ == "__main__":
    main()
