"""Token-synchronized brain-space playback must preserve spatial identity."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flyrl.brain_activity import (
    BrainPlayback,
    RenderOptions,
    activity_changes,
    activity_emphasis,
    align_soma_positions,
    render_playback,
)
from flyrl.brain_activity_scene import view_azimuths


def _small_playback() -> BrainPlayback:
    return BrainPlayback(
        states=np.asarray([[-0.8, 0.1, 0.7], [-0.2, 0.9, 0.4]], dtype=np.float32),
        probabilities=np.asarray([0.4, 0.7], dtype=np.float32),
        token_labels=(".", "Ġfly"),
        prefix_texts=("Once.", "Once. fly"),
        coordinates=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 1.0, 4.0]],
            dtype=np.float64,
        ),
        missing_neurons=1,
    )


def test_align_soma_positions_preserves_activation_order() -> None:
    # Given: annotation rows in a different order with one missing soma.
    node_ids = (
        "malecns:v1.0:30",
        "malecns:v1.0:10",
        "malecns:v1.0:20",
    )
    annotation_ids = np.asarray([10, 20, 30], dtype=np.int64)
    locations = ((1, 2, 3), None, (7, 8, 9))
    # When: anatomical positions are aligned to activation columns.
    positions = align_soma_positions(node_ids, annotation_ids, locations)
    # Then: plotted indices retain activation order and missing somas are explicit.
    assert positions.activation_indices.tolist() == [0, 1]
    assert positions.coordinates.tolist() == [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]
    assert positions.missing_node_ids == ("malecns:v1.0:20",)


def test_activity_emphasis_highlights_token_to_token_change() -> None:
    # Given: one stable neuron and one neuron that changes on the second token.
    states = np.asarray([[0.5, 0.0], [0.5, 1.0]], dtype=np.float32)
    # When: visual emphasis is derived from state change.
    emphasis = activity_emphasis(states)
    # Then: the changed neuron dominates while the stable neuron fades.
    assert emphasis.shape == states.shape
    assert emphasis[1, 1] == 1.0
    assert emphasis[1, 0] == 0.0


def test_activity_changes_preserve_framewise_direction() -> None:
    # Given: stable signs whose token-to-token movement changes direction.
    states = np.asarray([[0.2, -0.1], [0.5, -0.4], [0.3, -0.2]], dtype=np.float32)
    # When: the state change used for color is computed.
    changes = activity_changes(states)
    # Then: frame 1 uses zero and later frames preserve signed movement.
    np.testing.assert_allclose(
        changes,
        np.asarray([[0.2, -0.1], [0.3, -0.3], [-0.2, 0.2]], dtype=np.float32),
    )


def test_render_playback_writes_one_frame_per_token(tmp_path: Path) -> None:
    # Given: a small spatial recording with two generated tokens.
    playback = _small_playback()
    output = tmp_path / "brain-activity.gif"
    # When: the scientific playback is rendered.
    rendered = render_playback(
        playback,
        RenderOptions(output=output, frames_per_second=2, dpi=50),
    )
    # Then: the animation contains exactly one synchronized frame per token.
    with Image.open(rendered) as image:
        assert image.size == (640, 360)
        image.seek(1)
        assert image.tell() == 1
        with pytest.raises(EOFError):
            image.seek(2)


def test_render_playback_accepts_rotating_three_dimensional_view(
    tmp_path: Path,
) -> None:
    # Given: a spatial recording and an explicit rotating 3D view.
    playback = _small_playback()
    output = tmp_path / "brain-activity-3d.gif"
    # When: the rotating anatomical view is rendered.
    rendered = render_playback(
        playback,
        RenderOptions(
            output=output,
            frames_per_second=2,
            dpi=50,
            view="rotating_3d",
        ),
    )
    # Then: a complete 16:9 animation is available for README embedding.
    with Image.open(rendered) as image:
        assert image.size == (640, 360)
        image.seek(1)
        assert image.tell() == 1


def test_render_playback_accepts_fixed_three_dimensional_activity_view(
    tmp_path: Path,
) -> None:
    # Given: a spatial recording and the fixed-camera activity view.
    output = tmp_path / "brain-activity-fixed-3d.gif"
    # When: token-linked changes are rendered without camera rotation.
    rendered = render_playback(
        _small_playback(),
        RenderOptions(
            output=output,
            frames_per_second=2,
            dpi=50,
            view="activity_3d",
        ),
    )
    # Then: every token still has one 16:9 frame.
    with Image.open(rendered) as image:
        assert image.size == (640, 360)
        image.seek(1)
        assert image.tell() == 1


def test_activity_three_dimensional_camera_does_not_move_between_tokens() -> None:
    # Given: four token frames in each 3D view.
    # When: their camera schedules are constructed.
    fixed = view_azimuths("activity_3d", 4)
    rotating = view_azimuths("rotating_3d", 4)
    # Then: activity comparison is fixed while the anatomy overview still rotates.
    assert fixed == (-35.0, -35.0, -35.0, -35.0)
    assert len(set(rotating)) == 4


def test_render_playback_writes_supported_mp4(tmp_path: Path) -> None:
    # Given: a small recording and the machine's available FFmpeg installation.
    output = tmp_path / "brain-activity.mp4"
    # When: the MP4 path is rendered.
    rendered = render_playback(
        _small_playback(),
        RenderOptions(output=output, frames_per_second=2, dpi=50),
    )
    # Then: FFmpeg produced a nonempty MP4 container.
    assert rendered.stat().st_size > 1_000
