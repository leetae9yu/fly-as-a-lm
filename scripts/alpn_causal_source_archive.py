"""Exact authenticated anatomy-factorial source archive boundary."""

from pathlib import Path
from typing import Final
from zipfile import ZipFile

SOURCE_ARCHIVE_MEMBERS: Final = (
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
)
SOURCE_REQUIRED_MEMBERS: Final = ("checkpoint.npz", "report.json", "runtime.json")


def read_source_payloads(path: Path) -> dict[str, bytes]:
    """Authenticate the exact original archive shape and return metric-free inputs."""
    with ZipFile(path) as archive:
        names = archive.namelist()
        if (
            len(names) != len(set(names))
            or set(names) != set(SOURCE_ARCHIVE_MEMBERS)
            or any(Path(name).name != name for name in names)
            or archive.testzip() is not None
        ):
            message = "Calibration source archive membership or CRC differs"
            raise ValueError(message)
        return {name: archive.read(name) for name in SOURCE_REQUIRED_MEMBERS}
