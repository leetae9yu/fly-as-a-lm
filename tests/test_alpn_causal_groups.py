"""Deterministic joint-wiring controls and auditable common-support failures."""

from itertools import permutations
from math import nextafter

import numpy as np
import pytest

from scripts.alpn_causal_groups import (
    COORDINATES,
    Balance,
    FloatMatrix,
    MatchingError,
    assemble_coordinates,
    empirical_ks,
    match_controls,
    minimum_cost_assignment,
    standardized_mean_difference,
)
from scripts.regional_probe_groups import RegionalSelection


def fixture() -> tuple[tuple[str, ...], RegionalSelection, FloatMatrix]:
    selection = RegionalSelection((0, 1, 2), (3,), (4,), (5,), (6,), 7)
    values = np.zeros((12, 16), dtype=np.float64)
    values[0::2] = 1.0
    return tuple(f"body:{index:02}" for index in range(12)), selection, values


@pytest.mark.parametrize(("rows", "columns"), [(1, 1), (1, 4), (2, 3), (3, 3), (4, 5)])
@pytest.mark.parametrize("fractional", [False, True])
def test_assignment_matches_brute_force(
    rows: int, columns: int, fractional: bool
) -> None:
    rng = np.random.default_rng(291)
    for _ in range(12):
        costs = rng.integers(-3, 5, size=(rows, columns)).astype(np.float64)
        if fractional:
            costs += rng.random((rows, columns))
        result = minimum_cost_assignment(costs)
        optimum = min(
            sum(float(costs.item(row, column)) for row, column in enumerate(order))
            for order in permutations(range(columns), rows)
        )
        assert len(set(result)) == rows
        assert (
            sum(float(costs.item(row, col)) for row, col in enumerate(result))
            == optimum
        )


def test_assignment_is_not_greedy_and_does_not_perturb_costs() -> None:
    assert minimum_cost_assignment(np.array([[1.0, 2.0], [1.0, 100.0]])) == (1, 0)
    assert minimum_cost_assignment(np.zeros((3, 5), dtype=np.float64)) == (0, 1, 2)
    assert minimum_cost_assignment(np.array([[1e-15, 0.0], [0.0, 0.0]])) == (1, 0)


@pytest.mark.parametrize(
    "costs",
    [np.zeros((2, 1)), np.zeros((0, 2)), np.array([[np.nan]]), np.array([[np.inf]])],
)
def test_assignment_rejects_invalid_costs(costs: FloatMatrix) -> None:
    with pytest.raises(MatchingError):
        _ = minimum_cost_assignment(costs)


def test_coordinate_assembly_preserves_promised_order_and_transforms() -> None:
    structure = np.array([[1.0, 2.0, 3.0, 4.0]])
    propagation = np.array([[2.0, 3.0, 0.0, np.inf, 4.0, 2.0]])
    activity = np.array([[-0.5, 0.75, 0.0, 0.1, 0.2, 1.0]])
    actual = assemble_coordinates(structure, propagation, activity)
    expected = np.array(
        [
            [
                *np.log1p([1.0, 2.0, 3.0, 4.0]),
                np.log1p(2),
                np.log1p(3),
                0,
                3,
                3,
                2,
                -0.5,
                0.75,
                np.log(0.001),
                np.log(0.1),
                0.2,
                1.0,
            ]
        ]
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64
    assert len(COORDINATES) == actual.shape[1] == 16
    assert propagation.item(0, 3) == float("inf")
    assert activity.item(0, 2) == 0


@pytest.mark.parametrize(("column", "value"), [(0, np.nan), (2, -1.0), (4, 1.01)])
def test_bad_activity_statistics_are_rejected(column: int, value: float) -> None:
    activity = np.zeros((1, 6), dtype=np.float64)
    activity[0, column] = value
    with pytest.raises(MatchingError):
        _ = assemble_coordinates(np.zeros((1, 4)), np.zeros((1, 6)), activity)


def test_balance_statistics_handle_ties_constants_and_shape_differences() -> None:
    left = np.array([0.0, 0.0, 2.0, 2.0])
    right = np.array([1.0, 1.0, 3.0, 3.0])
    assert standardized_mean_difference(left, right) == -1.0
    assert empirical_ks(left, right) == 0.5
    assert empirical_ks(left, left[::-1]) == 0.0
    assert empirical_ks(np.array([0.0]), np.array([0.0, 1.0])) == 0.5
    assert standardized_mean_difference(np.ones(2), np.ones(3)) == 0.0
    assert np.isinf(standardized_mean_difference(np.ones(2), np.zeros(2)))
    assert Balance("x", -0.10, 0.20).passed
    assert not Balance("x", nextafter(0.10, 1.0), 0.20).passed
    assert not Balance("x", 0.10, nextafter(0.20, 1.0)).passed


def test_exact_pairs_costs_exclusions_and_shared_controls() -> None:
    bodies, selection, values = fixture()
    draws = ((1, 0), (2, 1))
    original = values.copy()
    result = match_controls(bodies, selection, draws, values)
    assert result.status == "balanced"
    assert result.required == 2
    assert result.eligible == (7, 8, 9, 10, 11)
    assert result.excluded == tuple(range(7))
    assert result.for_wiring("real") is result.for_wiring("shuffled")
    assert len(result.controls) == 4
    pool = values[[0, 1, 2, 7, 8, 9, 10, 11]]
    mean, scale = np.empty(16), np.empty(16)
    _ = pool.mean(axis=0, out=mean)
    _ = pool.std(axis=0, out=scale)
    np.testing.assert_array_equal(result.mean, mean)
    np.testing.assert_array_equal(result.scale, scale)
    normalized = (values - result.mean) / result.scale
    for control in result.controls:
        width = 10 if control.comparator == "S" else 16
        assert len(control.balance) == width
        assert len(set(control.control_indices)) == 2
        assert set(control.control_indices) <= set(result.eligible)
        squared: FloatMatrix = (
            normalized[:, None, :width] - normalized[None, :, :width]
        ) ** 2
        expected = tuple(
            sum(float(squared.item(row, col, i)) for i in range(width))
            for row, col in zip(
                control.target_indices, control.control_indices, strict=True
            )
        )
        assert control.pair_costs == expected
        assert control.total_cost == sum(expected) == 0.0
        assert all(item.passed for item in control.balance)
    assert len(result.overlaps) == 6
    for overlap in result.overlaps:
        left = set(result.controls[overlap.left].control_indices)
        right = set(result.controls[overlap.right].control_indices)
        assert set(overlap.shared) == left & right
        assert len(overlap.shared) == 2
    assert result == match_controls(bodies, selection, draws, values)
    np.testing.assert_array_equal(values, original)


def test_body_identity_not_input_order_resolves_ties() -> None:
    bodies, selection, values = fixture()
    values[:] = 0
    baseline = match_controls(bodies, selection, ((1, 0),), values)
    order = (11, 9, 7, 10, 8, 6, 5, 4, 3, 2, 1, 0)
    inverse = {old: new for new, old in enumerate(order)}
    permuted_selection = RegionalSelection(
        *(
            tuple(inverse[i] for i in indices)
            for indices in (
                selection.alpn,
                selection.mbon,
                selection.kenyon,
                selection.sensory,
                selection.trained_readout,
            )
        ),
        seed=selection.seed,
    )
    reordered_bodies = tuple(bodies[i] for i in order)
    permuted = match_controls(
        reordered_bodies,
        permuted_selection,
        ((inverse[0], inverse[1]),),
        values[list(order)],
    )
    for original, reordered in zip(baseline.controls, permuted.controls, strict=True):
        assert tuple(bodies[i] for i in original.control_indices) == (
            "body:07",
            "body:08",
        )
        assert tuple(reordered_bodies[i] for i in reordered.control_indices) == (
            "body:07",
            "body:08",
        )
        assert tuple(reordered_bodies[i] for i in reordered.target_indices) == (
            "body:00",
            "body:01",
        )
    assert baseline.scale == (1.0,) * 16


def test_activity_changes_m_but_not_s_and_failing_coordinate_is_retained() -> None:
    bodies, selection, values = fixture()
    values[:] = 0
    values[7:9, 10:] = 10
    result = match_controls(bodies, selection, ((0, 1),), values)
    structure, activity = result.controls
    assert structure.control_indices == (7, 8)
    assert activity.control_indices == (9, 10)
    assert result.status == "balanced"
    values[7:, 15] = 1
    failed = match_controls(bodies, selection, ((0, 1),), values)
    assert failed.status == "insufficient_common_support"
    assert all(item.passed for item in failed.controls[0].balance)
    assert not failed.controls[1].balance[15].passed
    assert failed.controls[1].balance[15].ks == 1.0
    assert len(failed.controls) == 2


@pytest.mark.parametrize("coordinate", range(16))
def test_each_promised_coordinate_can_block_scoring(coordinate: int) -> None:
    bodies, selection, values = fixture()
    values[:] = 0
    values[7:, coordinate] = 1
    result = match_controls(bodies, selection, ((0, 1),), values)
    assert result.status == "insufficient_common_support"
    mean, scale = result.mean[coordinate], result.scale[coordinate]
    delta = (1 - mean) / scale - (0 - mean) / scale
    for control in result.controls:
        included = coordinate < len(control.balance)
        assert control.total_cost == (2 * delta * delta if included else 0)
        assert all(item.passed for item in control.balance) is not included
        if included:
            assert control.balance[coordinate].coordinate == COORDINATES[coordinate]
            assert control.balance[coordinate].ks == 1


def test_capacity_shortage_is_explicit_without_partial_assignment() -> None:
    bodies, selection, values = fixture()
    result = match_controls(bodies[:8], selection, ((0, 1),), values[:8])
    assert result.status == "insufficient_common_support"
    assert result.required == 2
    assert result.eligible == (7,)
    assert result.controls == result.overlaps == ()


@pytest.mark.parametrize("draws", [(), ((0, 0),), ((0, 7),), ((0,), (1, 2))])
def test_invalid_targets_are_not_silently_replaced(
    draws: tuple[tuple[int, ...], ...],
) -> None:
    bodies, selection, values = fixture()
    with pytest.raises(MatchingError):
        _ = match_controls(bodies, selection, draws, values)
