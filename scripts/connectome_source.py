"""Checksum-pinned MaleCNS source boundary."""

import hashlib
import shutil
from contextlib import closing
from http import HTTPStatus
from http.client import HTTPSConnection
from pathlib import Path
from typing import ClassVar, Final, Literal, TypeAlias
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from flyrl.connectome import GraphError

COMMIT: Final = "71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33"
BASE: Final = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
SourceName: TypeAlias = Literal["annotations.feather", "edges.feather"]


class SourcePin(BaseModel):
    """Trusted source identity, checked before parsing rather than learned on import."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    url: str
    bytes: int
    sha256: str


PINS: Final = {
    "annotations.feather": SourcePin(
        url=BASE + "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        bytes=14483314,
        sha256="2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    ),
    "edges.feather": SourcePin(
        url=BASE + "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        bytes=1051241946,
        sha256="e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    ),
}


def file_digest(path: Path) -> str:
    """Hash large files in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: Path, pin: SourcePin) -> None:
    """Reject truncated or changed sources before they reach the parser."""
    if path.stat().st_size != pin.bytes or file_digest(path) != pin.sha256:
        message = f"Source integrity mismatch: {path}"
        raise GraphError(message)


def obtain_source(raw: Path, name: SourceName, *, download: bool) -> None:
    """Stream one missing public file, then enforce its built-in SHA256 pin."""
    raw.mkdir(parents=True, exist_ok=True)
    pin = PINS[name]
    path = raw / name
    if not path.exists() and download:
        temporary = path.with_suffix(".partial")
        url = urlsplit(pin.url)
        with closing(HTTPSConnection(url.netloc, timeout=60)) as connection:
            connection.request("GET", url.path)
            with connection.getresponse() as response:
                if response.status != HTTPStatus.OK:
                    message = (
                        f"Source download failed: HTTP {response.status} {pin.url}"
                    )
                    raise GraphError(message)
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
        verify_source(temporary, pin)
        _ = temporary.replace(path)
    verify_source(path, pin)


def obtain_sources(raw: Path, *, download: bool) -> None:
    """Obtain both pinned public files for full-connectome preparation."""
    for name in ("annotations.feather", "edges.feather"):
        obtain_source(raw, name, download=download)
