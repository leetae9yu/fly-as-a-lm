"""Exercise automatic BPE selection, token decoding and resumed runs through CLI."""

from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from flyrl.ar_config import ARConfig
from flyrl.ar_learning import ARLearner
from flyrl.ar_reporting import RunReport, continuations
from flyrl.autoregressive import app
from flyrl.bpe_data import load_bpe_corpus, make_bpe_corpus, save_bpe_corpus
from flyrl.connectome import load_graph
from flyrl.language_data import TextSplits


@pytest.fixture
def bpe_command(tmp_path: Path) -> list[str]:
    text = "the fly can learn. the fly can move. " * 40
    corpus = make_bpe_corpus(TextSplits(text, text, text, "synthetic"), 300)
    path = tmp_path / "bpe.npz"
    save_bpe_corpus(corpus, path)
    return [
        "--graph",
        "data/large_connectome/malecns_v1_n256.npz",
        "--corpus",
        str(path),
        "--output",
        str(tmp_path / "runs"),
        "--device",
        "cpu",
        "--seeds",
        "0",
        "--controls",
        "real,shuffled,frozen",
        "--batch-size",
        "2",
        "--context",
        "3",
        "--eval-windows",
        "4",
        "--sample-length",
        "8",
        "--checkpoint-steps",
        "1",
    ]


def test_bpe_cli_uses_token_metrics_and_decoded_generation(
    bpe_command: list[str],
) -> None:
    # Given: a byte-level corpus supplied to the existing autoregressive CLI.
    # When: training all three conditions through the actual entry point.
    result = CliRunner().invoke(app, [*bpe_command, "--updates", "2"])
    assert result.exit_code == 0, result.output
    reports = [
        RunReport.model_validate_json(line) for line in result.stdout.splitlines()
    ]
    corpus = load_bpe_corpus(Path(bpe_command[bpe_command.index("--corpus") + 1]))
    # Then: BPE is inferred and every neural/baseline score uses token units.
    assert len(reports) == 3
    for report in reports:
        assert report.config.tokenization == "bpe"
        assert report.final["test"].bits_per_character is None
        assert report.final["test"].perplexity is not None
        assert len(report.generated_token_ids["sampled"]) == 8
        for name, ids in report.generated_token_ids.items():
            assert report.generated[name] == corpus.decode(list(ids))
        for score in report.baselines["test"].values():
            assert score.bits_per_character is None
            assert score.bits_per_token is not None
            assert score.windows == report.final["test"].windows


def test_bpe_cli_resumes_all_controls(bpe_command: list[str]) -> None:
    # Given: one persisted update of all three controls.
    first = CliRunner().invoke(app, [*bpe_command, "--updates", "1"])
    assert first.exit_code == 0, first.output
    original = [
        RunReport.model_validate_json(line) for line in first.stdout.splitlines()
    ]
    # When: resuming to a fixed total budget.
    second = CliRunner().invoke(app, [*bpe_command, "--updates", "2", "--resume"])
    assert second.exit_code == 0, second.output
    resumed = [
        RunReport.model_validate_json(line) for line in second.stdout.splitlines()
    ]
    # Then: initial scores survive and progress is continued rather than reset.
    for before, after in zip(original, resumed, strict=True):
        assert after.updates == 2
        assert after.initial == before.initial
        assert (
            after.identities["tokenizer_sha256"]
            == before.identities["tokenizer_sha256"]
        )


def test_generation_decodes_across_prompt_byte_boundary() -> None:
    # Given: a prompt ending halfway through the UTF-8 encoding of e-acute.
    text = "\u00e9" + " a" * 40
    corpus = make_bpe_corpus(TextSplits(text, text, text, "synthetic"), 257)
    graph = load_graph(Path("data/large_connectome/malecns_v1_n256.npz"))
    config = ARConfig(alphabet_size=257, tokenization="bpe", context=1, sample_length=1)
    learner = ARLearner(graph, config)
    with torch.no_grad():
        _ = learner.model.readout.zero_()
        _ = learner.model.output_bias.zero_()
        learner.model.output_bias[int(corpus.train.item(1))] = 100
    # When: decoding the generated continuation together with its prompt.
    result = continuations(learner, corpus)
    # Then: joining raw IDs before decoding preserves the completed Unicode character.
    assert result.complete_text["greedy"] == "\u00e9"
