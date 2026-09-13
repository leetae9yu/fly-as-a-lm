"""Prospective anatomy-port factorial contrast and decision regressions."""

import pytest

import scripts.anatomy_factorial_schema as schema


def condition(
    seed: int,
    wiring: schema.Wiring,
    policy: schema.FactorialPortPolicy,
    nll: float,
) -> schema.FactorialConditionResult:
    return schema.FactorialConditionResult(
        seed=seed,
        wiring=wiring,
        port_policy=policy,
        test_nll=nll,
        test_accuracy=0.2,
    )


def passing_results() -> tuple[schema.FactorialConditionResult, ...]:
    rows: list[schema.FactorialConditionResult] = []
    for seed in range(7, 13):
        rows.extend(
            (
                condition(seed, "real", "alpn_mbon", 4.0),
                condition(seed, "real", "alpn_random", 4.2),
                condition(seed, "real", "random_mbon", 4.3),
                condition(seed, "real", "random_random", 4.5),
                condition(seed, "shuffled", "alpn_mbon", 4.4),
                condition(seed, "shuffled", "random_random", 4.6),
            )
        )
    return tuple(rows)


def test_factorial_decision_requires_port_and_topology_gates() -> None:
    # Given: six seeds with a .5 real port gain and .3 wiring interaction.
    results = passing_results()
    # When: the prospective practical and unanimous-sign gates are applied.
    summary = schema.evaluate_factorial(results, 0.10, 0.05)
    # Then: both gates advance and the 2x2 real-wiring effects are explicit.
    assert summary.analysis.verdict == "joint_success"
    assert summary.analysis.advance is True
    assert summary.analysis.median_primary_gain == 0.5
    assert summary.analysis.median_topology_interaction == pytest.approx(0.3)
    assert summary.rows[0].input_gain == pytest.approx(0.3)
    assert summary.rows[0].output_gain == pytest.approx(0.2)
    assert summary.rows[0].input_output_interaction == pytest.approx(0.0)


def test_factorial_decision_separates_port_only_result() -> None:
    # Given: anatomy ports help equally under real and shuffled wiring.
    results = tuple(
        result.model_copy(update={"test_nll": 4.1})
        if result.wiring == "shuffled" and result.port_policy == "alpn_mbon"
        else result
        for result in passing_results()
    )
    # When: only the primary port gate passes.
    summary = schema.evaluate_factorial(results, 0.10, 0.05)
    # Then: the result is useful engineering evidence but not a wiring-coupled advance.
    assert summary.analysis.verdict == "primary_only"
    assert summary.analysis.advance is False


def test_factorial_decision_retains_interaction_only_result() -> None:
    # Given: a subthreshold primary gain but a practical wiring interaction.
    results = tuple(
        result.model_copy(update={"test_nll": 4.45})
        if result.wiring == "real" and result.port_policy == "alpn_mbon"
        else result.model_copy(update={"test_nll": 4.7})
        if result.wiring == "shuffled" and result.port_policy == "alpn_mbon"
        else result
        for result in passing_results()
    )
    # When: the two gates are evaluated independently.
    summary = schema.evaluate_factorial(results, 0.10, 0.05)
    # Then: interaction evidence is not collapsed into generic inconclusive.
    assert summary.analysis.primary_status == "positive_subthreshold"
    assert summary.analysis.topology_status == "pass"
    assert summary.analysis.verdict == "interaction_only"
