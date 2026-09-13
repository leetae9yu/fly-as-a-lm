"""Prospective regional probe decision gates."""

from typing import Final

from scripts.regional_probe_groups import ProbeGroupName
from scripts.regional_probe_schema import (
    ProbeBaselines,
    ProbeResult,
    ProbeScore,
    evaluate_regional_probes,
)

GROUP_DRAWS: Final[dict[ProbeGroupName, int]] = {
    "alpn": 5,
    "kenyon": 5,
    "mbon": 1,
    "degree_matched": 1,
    "centrality": 1,
    "trained_readout": 1,
}


def score(nll: float) -> ProbeScore:
    return ProbeScore(tokens=10, nll=nll, perplexity=1.0, accuracy=0.5)


def fixture(
    *,
    weak_seed: int | None = None,
    invalid: bool = False,
) -> tuple[tuple[ProbeResult, ...], tuple[ProbeBaselines, ...]]:
    results: list[ProbeResult] = []
    baselines: list[ProbeBaselines] = []
    for seed in range(7, 13):
        for wiring in ("real", "shuffled"):
            baselines.append(
                ProbeBaselines(
                    seed=seed,
                    wiring=wiring,
                    unigram=score(5.0),
                    bigram=score(4.8),
                    original_head_nll=4.0,
                )
            )
            values = {
                "alpn": 4.6 if seed != weak_seed else 5.1,
                "kenyon": 4.7,
                "mbon": 5.2,
                "degree_matched": 4.9,
                "centrality": 4.5,
                "trained_readout": 5.1 if invalid else 4.2,
            }
            for group, draws in GROUP_DRAWS.items():
                for draw in range(draws):
                    nll = values[group] + (0.1 if wiring == "shuffled" else 0.0)
                    results.append(
                        ProbeResult(
                            seed=seed,
                            wiring=wiring,
                            group=group,
                            draw=draw,
                            valid=score(nll),
                            test=score(nll),
                        )
                    )
    return tuple(results), tuple(baselines)


def test_regional_probe_gate_nominates_all_passing_candidates() -> None:
    # Given: every real candidate beats matched and unigram controls in six seeds.
    results, baselines = fixture()
    # When: the prospective familywise decision is applied.
    summary = evaluate_regional_probes(results, baselines)
    # Then: all candidates survive practical and Holm gates.
    assert summary.analysis.verdict == "multiple_candidates"
    assert summary.analysis.nominated == ("alpn", "kenyon", "centrality")
    assert all(
        candidate.intersection_p == 0.015625
        for candidate in summary.analysis.candidates
    )
    assert all(candidate.holm_rejected for candidate in summary.analysis.candidates)


def test_regional_probe_gate_counts_one_reversal_against_advancement() -> None:
    # Given: ALPN loses to controls in one of six real checkpoints.
    results, baselines = fixture(weak_seed=9)
    # When/Then: ALPN cannot advance despite positive median differences.
    summary = evaluate_regional_probes(results, baselines)
    assert "alpn" not in summary.analysis.nominated
    alpn = next(
        candidate
        for candidate in summary.analysis.candidates
        if candidate.candidate == "alpn"
    )
    assert not alpn.practical_pass


def test_regional_probe_gate_rejects_invalid_positive_control() -> None:
    # Given: refitting the source model's trained readout cannot beat unigram.
    results, baselines = fixture(invalid=True)
    # When/Then: no anatomical nomination is interpreted.
    summary = evaluate_regional_probes(results, baselines)
    assert summary.analysis.verdict == "probe_invalid"
    assert summary.analysis.nominated == ()
