"""Pure sign, Holm and verdict helpers for regional probes."""

from dataclasses import dataclass
from math import comb
from typing import Final, Literal, TypeAlias

Candidate: TypeAlias = Literal["alpn", "kenyon", "centrality"]
ProbeVerdict: TypeAlias = Literal[
    "probe_invalid",
    "no_candidate",
    "single_candidate",
    "multiple_candidates",
]
CANDIDATES: Final[tuple[Candidate, ...]] = ("alpn", "kenyon", "centrality")


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Decision inputs before the joint Holm step."""

    candidate: Candidate
    matched: tuple[float, ...]
    unigram: tuple[float, ...]
    matched_p: float
    unigram_p: float
    intersection_p: float
    practical: bool


def sign_p(values: tuple[float, ...]) -> float:
    """Return the one-sided exact sign probability, counting ties as failures."""
    successes = sum(1 for value in values if value > 0)
    numerator = sum(
        comb(len(values), count) for count in range(successes, len(values) + 1)
    )
    return float(numerator) / float(1 << len(values))


def holm_rejections(evidence: tuple[CandidateEvidence, ...]) -> set[Candidate]:
    """Apply a step-down Holm familywise correction at alpha .05."""
    ordered = sorted(
        evidence,
        key=lambda item: (item.intersection_p, CANDIDATES.index(item.candidate)),
    )
    rejected: set[Candidate] = set()
    stepdown_open = True
    for rank, item in enumerate(ordered):
        passes = stepdown_open and item.intersection_p <= 0.05 / (len(ordered) - rank)
        if passes:
            rejected.add(item.candidate)
        else:
            stepdown_open = False
    return rejected


def probe_verdict(
    valid: bool,
    nominated: tuple[Candidate, ...],
) -> ProbeVerdict:
    """Name the exact validity and candidate-count outcome."""
    if not valid:
        return "probe_invalid"
    if not nominated:
        return "no_candidate"
    if len(nominated) == 1:
        return "single_candidate"
    return "multiple_candidates"
