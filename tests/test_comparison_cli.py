"""Run dense references through the same training, reporting and resume CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from flyrl.ar_reporting import RunReport
from flyrl.autoregressive import app


@pytest.mark.parametrize("architecture", ["gru", "transformer"])
def test_dense_cli_reports_and_resumes(architecture: str, tmp_path: Path) -> None:
    # Given: the same BPE artifact and CLI used by anatomical conditions.
    arguments = [
        "--graph",
        "data/large_connectome/malecns_v1_n256.npz",
        "--corpus",
        "data/bpe_corpus/corpus.npz",
        "--output",
        str(tmp_path),
        "--architecture",
        architecture,
        "--generation-context",
        "windowed",
        "--controls",
        "real",
        "--seeds",
        "0",
        "--context",
        "3",
        "--batch-size",
        "2",
        "--eval-windows",
        "2",
        "--sample-length",
        "5",
        "--checkpoint-steps",
        "1",
    ]
    first = CliRunner().invoke(app, [*arguments, "--updates", "1"])
    assert first.exit_code == 0, first.output
    before = RunReport.model_validate_json(first.stdout)
    # When: resuming the same model to the shared total target.
    result = CliRunner().invoke(app, [*arguments, "--updates", "2", "--resume"])
    assert result.exit_code == 0, result.output
    report = RunReport.model_validate_json(result.stdout)
    # Then: model identity, held-out units and exact progress survive the real surface.
    assert report.config.architecture == architecture
    assert report.initial == before.initial
    assert report.updates == 2
    assert report.zero_recurrent == {}
    assert report.final["test"].bits_per_token is not None
    assert len(report.generated_token_ids["sampled"]) == 5
    assert (tmp_path / "seed-0" / architecture / "checkpoint.npz").is_file()
