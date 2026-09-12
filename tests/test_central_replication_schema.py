"""Prospective central replication decision regressions."""

from math import exp

import pytest

from scripts.central_replication_schema import ReplicationRow, evaluate


def row(seed: int, delta: float) -> ReplicationRow:
    """Build one paired outcome with independently consistent NLL values."""
    real_nll = 4.5
    return ReplicationRow(
        seed=seed,
        real_test_nll=real_nll,
        shuffled_test_nll=real_nll + delta,
        delta=delta,
        real_accuracy=0.24,
        shuffled_accuracy=0.20,
    )


@pytest.mark.parametrize(
    ("deltas", "verdict", "advance", "sign_p"),
    [
        ((0.05, 0.08, 0.10, 0.10, 0.12, 0.15), "advance", True, 0.03125),
        ((0.10, 0.11, 0.12, 0.13, 0.14, 0.15), "advance", True, 0.03125),
        ((0.01, 0.02, 0.03, 0.04, 0.05, 0.06), "inconclusive", False, 0.03125),
        ((0.10, 0.11, -0.01, 0.13, 0.14, 0.15), "inconclusive", False, None),
        ((-0.10, -0.11, -0.12, -0.13, -0.14, -0.15), "reversed", False, 0.03125),
    ],
)
def test_evaluate_applies_frozen_advancement_rule(
    deltas: tuple[float, ...],
    verdict: str,
    advance: bool,
    sign_p: float | None,
) -> None:
    # Given: all six complete fresh-seed differences.
    rows = tuple(row(seed, delta) for seed, delta in enumerate(deltas, start=1))
    # When: the prospective threshold and sign rule are evaluated.
    summary = evaluate(rows, 0.10)
    # Then: direction, practical magnitude and paired PPL ratio stay distinct.
    assert summary.analysis.verdict == verdict
    assert summary.analysis.advance is advance
    assert summary.analysis.two_sided_unanimous_sign_p == sign_p
    expected_mean = sum(deltas) / len(deltas)
    assert summary.analysis.mean_delta == pytest.approx(expected_mean)
    assert summary.analysis.geometric_mean_ppl_ratio_real_over_shuffled == (
        pytest.approx(exp(-expected_mean))
    )
