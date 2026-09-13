"""Prospective aggregate decisions for frozen regional probes."""

import statistics
from typing import ClassVar, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from scripts.regional_probe_decision import (
    CANDIDATES,
    Candidate,
    CandidateEvidence,
    ProbeVerdict,
    holm_rejections,
    probe_verdict,
    sign_p,
)
from scripts.regional_probe_groups import ProbeGroupName

Wiring: TypeAlias = Literal["real", "shuffled"]
SEEDS: Final = tuple(range(7, 13))
WIRINGS: Final[tuple[Wiring, ...]] = ("real", "shuffled")
EXPECTED_DRAWS: Final[dict[ProbeGroupName, int]] = {
    "alpn": 5,
    "kenyon": 5,
    "mbon": 1,
    "degree_matched": 1,
    "centrality": 1,
    "trained_readout": 1,
}


class ProbeScore(BaseModel):
    """One fixed-head split score."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    tokens: int
    nll: float
    perplexity: float
    accuracy: float


class ProbeResult(BaseModel):
    """One source checkpoint, group draw and fitted head."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    seed: int
    wiring: Wiring
    group: ProbeGroupName
    draw: int
    valid: ProbeScore
    test: ProbeScore


class ProbeBaselines(BaseModel):
    """Train-only frequency references and original source-head score."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    seed: int
    wiring: Wiring
    unigram: ProbeScore
    bigram: ProbeScore
    original_head_nll: float


class CandidateDecision(BaseModel):
    """Six-seed practical and Holm-corrected advancement evidence."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    candidate: Candidate
    matched_differences: tuple[float, ...]
    unigram_differences: tuple[float, ...]
    matched_sign_p: float
    unigram_sign_p: float
    intersection_p: float
    holm_rejected: bool
    practical_pass: bool


class RegionalProbeAnalysis(BaseModel):
    """Validity gate, nominated candidates and MBON diagnostic."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    verdict: ProbeVerdict
    nominated: tuple[Candidate, ...]
    validity_differences: tuple[float, ...]
    mbon_deficits: tuple[float, ...]
    candidates: tuple[CandidateDecision, ...]


class RegionalProbeSummary(BaseModel):
    """Complete sealed diagnostic decision."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    format: Literal["flyrl-regional-probe-results-v1"] = (
        "flyrl-regional-probe-results-v1"
    )
    results: tuple[ProbeResult, ...]
    baselines: tuple[ProbeBaselines, ...]
    analysis: RegionalProbeAnalysis


def _group_nll(
    by_key: dict[tuple[int, Wiring, ProbeGroupName], tuple[ProbeResult, ...]],
    seed: int,
    wiring: Wiring,
    group: ProbeGroupName,
) -> float:
    draws = by_key[(seed, wiring, group)]
    return statistics.mean(result.test.nll for result in draws)


def _index_results(
    results: tuple[ProbeResult, ...],
    baselines: tuple[ProbeBaselines, ...],
    practical_threshold: float,
) -> tuple[
    dict[tuple[int, Wiring, ProbeGroupName], tuple[ProbeResult, ...]],
    dict[tuple[int, Wiring], ProbeBaselines],
]:
    by_key: dict[
        tuple[int, Wiring, ProbeGroupName],
        tuple[ProbeResult, ...],
    ] = {}
    for seed in SEEDS:
        for wiring in WIRINGS:
            for group, draw_count in EXPECTED_DRAWS.items():
                draws = tuple(
                    result
                    for result in results
                    if (
                        result.seed == seed
                        and result.wiring == wiring
                        and result.group == group
                    )
                )
                if {result.draw for result in draws} != set(range(draw_count)):
                    message = (
                        f"Regional probe draw matrix differs: {seed}/{wiring}/{group}"
                    )
                    raise ValueError(message)
                by_key[(seed, wiring, group)] = draws
    expected_results = 2 * len(SEEDS) * sum(EXPECTED_DRAWS.values())
    baseline_by_key: dict[tuple[int, Wiring], ProbeBaselines] = {
        (baseline.seed, baseline.wiring): baseline for baseline in baselines
    }
    if (
        len(results) != expected_results
        or len(baseline_by_key) != 2 * len(SEEDS)
        or set(baseline_by_key)
        != {(seed, wiring) for seed in SEEDS for wiring in WIRINGS}
        or practical_threshold <= 0
    ):
        message = "Regional probe results or baselines are incomplete"
        raise ValueError(message)
    return by_key, baseline_by_key


def evaluate_regional_probes(
    results: tuple[ProbeResult, ...],
    baselines: tuple[ProbeBaselines, ...],
    practical_threshold: float = 0.10,
) -> RegionalProbeSummary:
    """Apply validity, regional superiority and Holm advancement gates."""
    by_key, baseline_by_key = _index_results(
        results,
        baselines,
        practical_threshold,
    )
    validity = tuple(
        baseline_by_key[(seed, "real")].unigram.nll
        - _group_nll(by_key, seed, "real", "trained_readout")
        for seed in SEEDS
    )
    validity_pass = (
        all(value > 0 for value in validity)
        and statistics.median(validity) >= practical_threshold
    )
    mbon_deficits = tuple(
        _group_nll(by_key, seed, "real", "mbon")
        - _group_nll(by_key, seed, "real", "degree_matched")
        for seed in SEEDS
    )
    evidence: list[CandidateEvidence] = []
    for candidate in CANDIDATES:
        matched = tuple(
            _group_nll(by_key, seed, "real", "degree_matched")
            - _group_nll(by_key, seed, "real", candidate)
            for seed in SEEDS
        )
        unigram = tuple(
            baseline_by_key[(seed, "real")].unigram.nll
            - _group_nll(by_key, seed, "real", candidate)
            for seed in SEEDS
        )
        matched_p, unigram_p = sign_p(matched), sign_p(unigram)
        evidence.append(
            CandidateEvidence(
                candidate,
                matched,
                unigram,
                matched_p,
                unigram_p,
                max(matched_p, unigram_p),
                (
                    all(value > 0 for value in (*matched, *unigram))
                    and statistics.median(matched) >= practical_threshold
                    and statistics.median(unigram) >= practical_threshold
                ),
            )
        )
    rejected = holm_rejections(tuple(evidence))
    decisions = tuple(
        CandidateDecision(
            candidate=item.candidate,
            matched_differences=item.matched,
            unigram_differences=item.unigram,
            matched_sign_p=item.matched_p,
            unigram_sign_p=item.unigram_p,
            intersection_p=item.intersection_p,
            holm_rejected=item.candidate in rejected,
            practical_pass=item.practical,
        )
        for item in evidence
    )
    nominated: tuple[Candidate, ...] = tuple(
        item.candidate
        for item in evidence
        if validity_pass and item.practical and item.candidate in rejected
    )
    return RegionalProbeSummary(
        results=tuple(
            sorted(
                results,
                key=lambda result: (
                    result.seed,
                    result.wiring,
                    result.group,
                    result.draw,
                ),
            )
        ),
        baselines=tuple(sorted(baselines, key=lambda item: (item.seed, item.wiring))),
        analysis=RegionalProbeAnalysis(
            verdict=probe_verdict(validity_pass, nominated),
            nominated=nominated,
            validity_differences=validity,
            mbon_deficits=mbon_deficits,
            candidates=decisions,
        ),
    )
