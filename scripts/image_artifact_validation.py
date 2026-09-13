"""Decode activation figures instead of trusting filename signatures."""

import zlib
from pathlib import Path
from typing import Final
from xml.parsers import expat

import numpy as np
from matplotlib import image as matplotlib_image

PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
PNG_CHUNK_OVERHEAD: Final = 12
PNG_IHDR_LENGTH: Final = 13
FINAL_XML_CHUNK: Final = True


def _validate_svg(path: Path) -> bool:
    root: list[str] = []

    def start(name: str, attributes: dict[str, str]) -> None:
        del attributes
        if not root:
            root.append(name)

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    _ = parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    try:
        _ = parser.Parse(path.read_bytes(), FINAL_XML_CHUNK)
    except (expat.ExpatError, OSError, ValueError) as error:
        message = f"Activation figure missing or invalid: {path.name}"
        raise ValueError(message) from error
    return root == ["svg"]


def _png_chunks_complete(data: bytes) -> bool:
    if not data.startswith(PNG_SIGNATURE):
        return False
    offset = 8
    first = True
    saw_data = False
    complete = False
    while offset < len(data):
        if len(data) - offset < PNG_CHUNK_OVERHEAD:
            break
        length = int.from_bytes(data[offset : offset + 4], "big")
        stop = offset + PNG_CHUNK_OVERHEAD + length
        if stop > len(data):
            break
        chunk_type = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(data[offset + 8 + length : stop], "big")
        actual_crc = zlib.crc32(payload, zlib.crc32(chunk_type))
        if actual_crc != expected_crc:
            break
        if first and (chunk_type != b"IHDR" or length != PNG_IHDR_LENGTH):
            break
        first = False
        saw_data = saw_data or chunk_type == b"IDAT"
        if chunk_type == b"IEND":
            complete = length == 0 and saw_data and stop == len(data)
            break
        offset = stop
    return complete


def _validate_png(path: Path) -> bool:
    try:
        complete = _png_chunks_complete(path.read_bytes())
    except OSError as error:
        message = f"Activation figure missing or invalid: {path.name}"
        raise ValueError(message) from error
    if not complete:
        return False
    try:
        pixels = matplotlib_image.imread(path)
    except (OSError, SyntaxError, ValueError) as error:
        message = f"Activation figure missing or invalid: {path.name}"
        raise ValueError(message) from error
    return pixels.size > 0 and bool(np.isfinite(pixels).all())


def validate_figure(path: Path) -> None:
    """Require one complete, decodable PNG or structurally complete SVG."""
    if not path.is_file():
        message = f"Activation figure missing or invalid: {path.name}"
        raise ValueError(message)
    validator = {".png": _validate_png, ".svg": _validate_svg}.get(path.suffix)
    if validator is None or not validator(path):
        message = f"Activation figure missing or invalid: {path.name}"
        raise ValueError(message)
