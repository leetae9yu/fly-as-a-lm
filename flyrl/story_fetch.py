"""Opt-in bounded HTTPS acquisition of immutable original TinyStories prefixes."""

import hashlib
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from http import HTTPStatus
from math import gcd
from pathlib import Path
from typing import Annotated, Final, Literal

import httpx2
from pydantic import Field

from flyrl.language_data import CorpusError
from flyrl.language_models import Settings

REVISION: Final = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
SOURCE: Final = "https://huggingface.co/datasets/roneneldan/TinyStories"
LICENSE: Final = "cdla-sharing-1.0"
DELIMITER: Final = b"<|endoftext|>"
MAX_REDIRECTS: Final = 5
CARD_BYTES: Final = 128 * 1024


class PrefixBudget(Settings):
    """Limits refer to complete source stories and maximum response bytes read."""

    stories: Annotated[int, Field(ge=1)] = 1032
    max_bytes: Annotated[int, Field(ge=1)] = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PrefixData:
    """Selected complete bytes and consumed response bytes including chunk lookahead."""

    payload: bytes
    bytes_read: int


def read_prefix(chunks: Iterator[bytes], budget: PrefixBudget) -> PrefixData:
    """Select complete stories with bounded buffering and chunk-split delimiters."""
    prefix = bytearray()
    count, cursor = 0, 0
    content = False
    for chunk in chunks:
        if len(prefix) + len(chunk) > budget.max_bytes:
            raise CorpusError(
                reason="Source byte budget exhausted before complete stories"
            )
        prefix.extend(chunk)
        while (end := prefix.find(b"\n", cursor)) >= 0:
            line = bytes(prefix[cursor:end]).strip()
            cursor = end + 1
            if line == DELIMITER:
                count += int(content)
                content = False
                if count == budget.stories:
                    return PrefixData(bytes(prefix[:cursor]), len(prefix))
            else:
                content = content or bool(line)
        if len(prefix) == budget.max_bytes:
            raise CorpusError(
                reason="Source byte budget exhausted before complete stories"
            )
    raise CorpusError(reason="Source ended before requested complete stories")


def _chunks(response: httpx2.Response, max_bytes: int) -> Iterator[bytes]:
    """Bound application reads even if Range is ignored, without decompression."""
    _ = response.raise_for_status()
    if response.status_code == HTTPStatus.PARTIAL_CONTENT and not response.headers.get(
        "Content-Range",
        "",
    ).startswith("bytes 0-"):
        raise CorpusError(reason="Source did not return a prefix byte range")
    if response.headers.get("Content-Encoding", "identity") != "identity":
        raise CorpusError(reason="Source ignored identity content encoding")
    consumed = 0
    # A divisor of the budget prevents application-level chunk overread at the cap.
    for chunk in response.iter_raw(chunk_size=gcd(4096, max_bytes)):
        consumed += len(chunk)
        yield chunk
        if consumed == max_bytes:
            return


class FetchReport(Settings):
    """Retained pinned source/license identity and downloaded prefix hashes."""

    source: str = SOURCE
    revision: str = REVISION
    license: str = LICENSE
    filename: str
    stories: int
    bytes_downloaded: int
    prefix_bytes: int
    prefix_sha256: str
    card_sha256: str
    selection: str = "First N complete source stories; nonrepresentative prefix subset"


def fetch_prefix(
    output: Path,
    split: Literal["train", "valid"],
    budget: PrefixBudget,
) -> FetchReport:
    """Fetch only on explicit invocation; never replace an existing directory."""
    if output.exists():
        message = f"Fetch output already exists: {output}"
        raise FileExistsError(message)
    filename = f"TinyStories-{split}.txt"
    transport = httpx2.HTTPTransport(
        http2=True,
        retries=3,
        limits=httpx2.Limits(
            max_connections=200,
            max_keepalive_connections=40,
            keepalive_expiry=30,
        ),
        socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
    )
    with httpx2.Client(
        transport=transport,
        timeout=httpx2.Timeout(connect=5, read=30, write=10, pool=10),
        follow_redirects=True,
        max_redirects=MAX_REDIRECTS,
        headers={"Accept-Encoding": "identity"},
    ) as client:
        with client.stream(
            "GET",
            f"{SOURCE}/resolve/{REVISION}/{filename}",
            headers={"Range": f"bytes=0-{budget.max_bytes - 1}"},
        ) as response:
            prefix = read_prefix(_chunks(response, budget.max_bytes), budget)
        with client.stream(
            "GET",
            f"{SOURCE}/raw/{REVISION}/README.md",
            headers={"Range": f"bytes=0-{CARD_BYTES - 1}"},
        ) as response:
            card = b"".join(_chunks(response, CARD_BYTES))
    if len(card) == CARD_BYTES:
        raise CorpusError(reason="Source card exceeds metadata byte budget")
    report = FetchReport(
        filename=filename,
        stories=budget.stories,
        bytes_downloaded=prefix.bytes_read,
        prefix_bytes=len(prefix.payload),
        prefix_sha256=hashlib.sha256(prefix.payload).hexdigest(),
        card_sha256=hashlib.sha256(card).hexdigest(),
    )
    output.mkdir(parents=True, exist_ok=False)
    _ = (output / filename).write_bytes(prefix.payload)
    _ = (output / "README.md").write_bytes(card)
    _ = (output / "SOURCE_LICENSE.txt").write_text(
        "\n".join(
            (
                f"Dataset license: {LICENSE}",
                "https://cdla.dev/sharing-1-0/",
                f"Pinned source: {SOURCE}/tree/{REVISION}",
                "",
                card.decode("utf-8"),
            )
        ),
        encoding="utf-8",
    )
    _ = (output / "source.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    return report
