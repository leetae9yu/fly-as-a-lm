"""Tests for complete renderable activation figures."""

from pathlib import Path

import numpy as np
import pytest

from flyrl.activation_plot import render_heatmaps
from scripts.image_artifact_validation import validate_figure


@pytest.mark.parametrize("extension", ["png", "svg"])
def test_figure_validation_accepts_complete_renderable_files(
    tmp_path: Path,
    extension: str,
) -> None:
    figures = render_heatmaps(
        np.asarray([[0.0]], dtype=np.float32),
        ("token",),
        ("neuron",),
        tmp_path,
    )
    path = next(path for path in figures if path.suffix == f".{extension}")

    validate_figure(path)


def test_figure_validation_rejects_truncated_png(tmp_path: Path) -> None:
    path = tmp_path / "activation.png"
    _ = path.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="invalid"):
        validate_figure(path)


def test_figure_validation_rejects_png_without_iend(tmp_path: Path) -> None:
    path = next(
        path
        for path in render_heatmaps(
            np.asarray([[0.0]], dtype=np.float32),
            ("token",),
            ("neuron",),
            tmp_path,
        )
        if path.suffix == ".png"
    )
    _ = path.write_bytes(path.read_bytes()[:-12])

    with pytest.raises(ValueError, match="invalid"):
        validate_figure(path)


@pytest.mark.parametrize(
    "payload",
    [
        "<svg></svg><svg/>",
        "<svg><g></svg>",
        "<svg><!--",
    ],
)
def test_figure_validation_rejects_malformed_svg(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "activation.svg"
    _ = path.write_text(payload)

    with pytest.raises(ValueError, match="invalid"):
        validate_figure(path)


def test_figure_validation_accepts_self_closing_svg(tmp_path: Path) -> None:
    path = tmp_path / "activation.svg"
    _ = path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')

    validate_figure(path)
