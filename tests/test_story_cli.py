"""Tiny CPU preparation, training, checkpoint continuation and exported generation."""

import importlib.util
from math import exp
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flyrl.activations import ActivationMetadata
from flyrl.story_pilot import PilotReport, app
from scripts.prepare_tinystories import app as prepare_app


def test_pilot_entrypoints_exist() -> None:
    # Given/When: resolving explicit preparation and training entry points.
    modules = ("scripts.prepare_tinystories", "flyrl.story_pilot")
    # Then: neither task requires the unrelated flat-text experiment command.
    assert all(importlib.util.find_spec(name) is not None for name in modules)


@pytest.mark.parametrize(
    ("sample_length", "control"),
    [(0, "real"), (2, "real"), (0, "shuffled")],
)
def test_tiny_cpu_cli_prepares_trains_and_resumes(
    tmp_path: Path, sample_length: int, control: str
) -> None:
    # Given: small delimiter-separated local stories with explicit source identity.
    raw = tmp_path / "stories.txt"
    _ = raw.write_text(
        "\n<|endoftext|>\n".join(f"The bird sang {i}." for i in range(8))
    )
    license_file = tmp_path / "license.txt"
    _ = license_file.write_text("Synthetic test fixture: CC0")
    data, output = tmp_path / "data", tmp_path / "run"
    runner = CliRunner()
    prepared = runner.invoke(
        prepare_app,
        [
            "--raw",
            str(raw),
            "--source",
            "synthetic",
            "--revision",
            "fixture-v1",
            "--license-file",
            str(license_file),
            "--output",
            str(data),
            "--train-stories",
            "4",
            "--valid-stories",
            "2",
            "--test-stories",
            "2",
            "--vocab-size",
            "257",
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    command = [
        "--corpus",
        str(data / "corpus.npz"),
        "--output",
        str(output),
        "--graph",
        "data/large_connectome/malecns_v1_n256.npz",
        "--context",
        "3",
        "--batch-size",
        "2",
        "--sample-length",
        str(sample_length),
        "--checkpoint-steps",
        "1",
        "--control",
        control,
        "--prompt",
        "The bird",
    ]
    # When: training through the real CLI and resuming to the same total as a fresh run.
    first = runner.invoke(app, [*command, "--updates", "1"])
    assert first.exit_code == 0, first.output
    before = PilotReport.model_validate_json(first.stdout)
    resumed = runner.invoke(app, [*command, "--updates", "2", "--resume"])
    assert resumed.exit_code == 0, resumed.output
    after = PilotReport.model_validate_json(resumed.stdout)
    fresh_command = command.copy()
    fresh_command[fresh_command.index("--output") + 1] = str(tmp_path / "full")
    fresh = runner.invoke(app, [*fresh_command, "--updates", "2"])
    assert fresh.exit_code == 0, fresh.output
    full = PilotReport.model_validate_json(fresh.stdout)
    # Then: original initial metrics persist and continuation is deterministic.
    assert before.initial == after.initial == full.initial
    assert after.final == full.final
    assert after.trace == full.trace
    assert after.generated_token_ids == full.generated_token_ids
    assert after.config.trainable_codes is True
    assert after.config.control == control
    assert after.config.device == "cpu"
    assert after.updates == 2
    assert after.final.test.tokens > 0
    assert after.final.test.perplexity == pytest.approx(exp(after.final.test.nll))
    assert (output / "checkpoint.npz").is_file()
    if sample_length:
        assert after.activations is not None
        activation = ActivationMetadata.model_validate_json(
            (output / after.activations).read_text()
        )
        assert activation.generated_ids == after.generated_token_ids["greedy"]
    else:
        assert after.activations is None
        assert after.generated_token_ids == {"greedy": (), "sampled": ()}
    # Given/When/Then: an existing run cannot be silently overwritten.
    collision = runner.invoke(app, [*command, "--updates", "2"])
    assert collision.exit_code != 0
