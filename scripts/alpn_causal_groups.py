"""Pure joint-wiring ALPN controls; no extraction, fitting, or fresh-text scores."""

from bisect import bisect_right
from itertools import combinations
from math import copysign, fsum, sqrt
from statistics import fmean, pvariance
from typing import Literal, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scripts.regional_probe_groups import RegionalSelection

FloatMatrix: TypeAlias = NDArray[np.float64]
MATRIX_DIMENSIONS = 2


class MatchingError(ValueError):
    """Invalid matching inputs, distinct from valid insufficient-support evidence."""


ORIGINAL_FEATURES = ("in_degree", "out_degree", "in_strength", "out_strength")
PROPAGATION_FEATURES = ("outgoing_l2", "sensory_distance", "readout_distance")
ACTIVITY_FEATURES = ("state_mean", "log_state_std", "saturation")
COORDINATES = ORIGINAL_FEATURES + tuple(
    f"{feature}_{wiring}"
    for feature in PROPAGATION_FEATURES + ACTIVITY_FEATURES
    for wiring in ("real", "shuffled")
)
STRUCTURAL_WIDTH = 10
SMD_LIMIT = 0.10
KS_LIMIT = 0.20


def _require(valid: bool, message: str) -> None:
    if not valid:
        raise MatchingError(message)


def _finite(values: FloatMatrix) -> bool:
    return values.dtype == np.float64 and bool(np.isfinite(values).all())


def assemble_coordinates(
    structure: FloatMatrix,
    propagation: FloatMatrix,
    activity: FloatMatrix,
) -> FloatMatrix:
    """Transform raw n-by-4/6/6 statistics in COORDINATES order; +inf is unreachable."""
    # Activity summarizes all old training targets, never fresh stories: signed
    # mean, population std, fraction(abs(h)>=.99), paired real before shuffled.
    nodes = structure.size // len(ORIGINAL_FEATURES)
    _require(
        structure.shape == (nodes, 4)
        and propagation.shape == activity.shape == (nodes, 6),
        "Invalid coordinate shapes",
    )
    _require(_finite(structure) and bool((structure >= 0).all()), "Invalid structure")
    _require(
        _finite(propagation[:, :2]) and bool((propagation >= 0).all()),
        "Invalid propagation",
    )
    _require(
        _finite(activity)
        and bool((activity[:, 2:] >= 0).all())
        and bool((activity[:, 4:] <= 1).all()),
        "Invalid activity",
    )
    result = np.empty((nodes, len(COORDINATES)), dtype=np.float64)
    result[:, :4] = np.log1p(structure)
    result[:, 4:6] = np.log1p(propagation[:, :2])
    result[:, 6:10] = np.minimum(propagation[:, 2:], 3)
    result[:, 10:12] = activity[:, :2]
    result[:, 12:14] = np.log(np.maximum(activity[:, 2:4], 0.001))
    result[:, 14:] = activity[:, 4:]
    return result


def minimum_cost_assignment(costs: FloatMatrix) -> tuple[int, ...]:
    """Return rectangular Hungarian column assignments in O(rows**2 * columns)."""
    # Callers supply body-sorted rows/columns. Strict slack updates retain the
    # earliest predecessor; argmin takes the first column. No epsilon perturbation.
    _require(costs.ndim == MATRIX_DIMENSIONS and _finite(costs), "Invalid costs")
    rows, columns = len(costs), len(costs.T)
    _require(0 < rows <= columns, "Assignment requires 1 <= rows <= columns")
    row_potential = np.zeros(rows + 1, dtype=np.float64)
    col_potential = np.zeros(columns + 1, dtype=np.float64)
    owner = np.zeros(columns + 1, dtype=np.int64)
    predecessor = np.zeros(columns + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        owner[0] = row
        column = 0
        slack = np.full(columns + 1, np.inf, dtype=np.float64)
        visited = np.zeros(columns + 1, dtype=np.bool_)
        while True:
            visited[column] = True
            current_row = int(owner.item(column))
            remaining = (~visited[1:]).nonzero()[0] + 1
            reduced: FloatMatrix = (
                costs[current_row - 1 : current_row, remaining - 1].reshape(-1)
                - float(row_potential.item(current_row))
                - col_potential[remaining]
            )
            improved = remaining[reduced < slack[remaining]]
            slack[improved] = reduced[reduced < slack[remaining]]
            predecessor[improved] = column
            column = int(remaining.item(int(slack[remaining].argmin())))
            delta = float(slack.item(column))
            row_potential[owner[visited]] += delta
            col_potential[visited] -= delta
            slack[remaining] -= delta
            if owner.item(column) == 0:
                break
        while column:
            previous = int(predecessor.item(column))
            owner[column] = owner[previous]
            column = previous
    result = np.empty(rows, dtype=np.int64)
    used = owner[1:] > 0
    result[owner[1:][used] - 1] = np.arange(columns)[used]
    return tuple(int(value) for value in result.flat)


def standardized_mean_difference(left: FloatMatrix, right: FloatMatrix) -> float:
    """Signed difference divided by pooled population SD; unequal constants fail."""
    difference = fmean(_floats(left)) - fmean(_floats(right))
    scale = sqrt((pvariance(_floats(left)) + pvariance(_floats(right))) / 2)
    if scale:
        return difference / scale
    return 0.0 if difference == 0 else copysign(float("inf"), difference)


def _floats(values: FloatMatrix) -> tuple[float, ...]:
    return tuple(float(value) for value in values.flat)


def empirical_ks(left: FloatMatrix, right: FloatMatrix) -> float:
    """Exact two-sample empirical-CDF supremum, including ties and unequal sizes."""
    first, second = sorted(_floats(left)), sorted(_floats(right))
    return max(
        abs(bisect_right(first, x) / left.size - bisect_right(second, x) / right.size)
        for x in set(first) | set(second)
    )


class Balance(NamedTuple):
    """Per-coordinate evidence; both inclusive thresholds are mandatory."""

    coordinate: str
    smd: float
    ks: float

    @property
    def passed(self) -> bool:
        """Whether this coordinate has prospective common support."""
        return abs(self.smd) <= SMD_LIMIT and self.ks <= KS_LIMIT


class MatchedControl(NamedTuple):
    """Correspondence follows body-sorted targets, not sorted matched columns."""

    comparator: Literal["S", "M"]
    draw: int
    target_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    pair_costs: tuple[float, ...]
    balance: tuple[Balance, ...]

    @property
    def total_cost(self) -> float:
        """Sum exact selected squared-Euclidean float64 costs."""
        return fsum(self.pair_costs)


class Overlap(NamedTuple):
    """Pairwise overlap across draws and comparators, identified by control index."""

    left: int
    right: int
    shared: tuple[int, ...]


class MatchingEvidence(NamedTuple):
    """Metric-free seed result; required > len(eligible) explicitly records shortage."""

    eligible: tuple[int, ...]
    excluded: tuple[int, ...]
    required: int
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    controls: tuple[MatchedControl, ...]
    overlaps: tuple[Overlap, ...]

    @property
    def status(self) -> Literal["balanced", "insufficient_common_support"]:
        """Reject any capacity or coordinate failure, without retries or relaxation."""
        passed = len(self.eligible) >= self.required and all(
            item.passed for control in self.controls for item in control.balance
        )
        return "balanced" if passed else "insufficient_common_support"

    def for_wiring(
        self, wiring: Literal["real", "shuffled"]
    ) -> tuple[MatchedControl, ...]:
        """Expose the identical realized ordered controls under either wiring."""
        _require(wiring in ("real", "shuffled"), "Unknown wiring")
        return self.controls


def _control(
    target: tuple[int, ...],
    eligible: tuple[int, ...],
    normalized: FloatMatrix,
    comparator: Literal["S", "M"],
    draw: int,
) -> MatchedControl:
    width = STRUCTURAL_WIDTH if comparator == "S" else len(COORDINATES)
    left = normalized[list(target), :width]
    right = normalized[list(eligible), :width]
    difference: FloatMatrix = left[:, None, :] - right[None, :, :]
    costs = np.empty((len(target), len(eligible)), dtype=np.float64)
    difference *= difference
    _ = difference.sum(axis=2, out=costs)
    columns = minimum_cost_assignment(costs)
    balances = tuple(
        Balance(
            name,
            standardized_mean_difference(left[:, index], right[list(columns), index]),
            empirical_ks(left[:, index], right[list(columns), index]),
        )
        for index, name in enumerate(COORDINATES[:width])
    )
    matched = tuple(eligible[column] for column in columns)
    pair_costs = tuple(float(costs.item(r, c)) for r, c in enumerate(columns))
    return MatchedControl(comparator, draw, target, matched, pair_costs, balances)


def match_controls(
    body_ids: tuple[str, ...],
    selection: RegionalSelection,
    draws: tuple[tuple[int, ...], ...],
    coordinates: FloatMatrix,
) -> MatchingEvidence:
    """Match frozen equal-width ALPN draws, using the full ALPN + eligible pool."""
    # Indices refer to input rows; identities sort lexicographically. Graph and
    # source-seed/port pairing are caller provenance obligations. No new draws.
    nodes = len(body_ids)
    groups = (selection.alpn, selection.mbon, selection.kenyon)
    groups += (selection.sensory, selection.trained_readout)
    excluded = {index for group in groups for index in group}
    _require(nodes == len(set(body_ids)) and all(body_ids), "Invalid body identities")
    _require(
        coordinates.shape == (nodes, len(COORDINATES)) and _finite(coordinates),
        "Invalid coordinates",
    )
    _require(all(0 <= i < nodes for i in excluded), "Invalid exclusion indices")
    allowed = (
        set(selection.alpn) - set(selection.sensory) - set(selection.trained_readout)
    )
    required = len(draws[0]) if draws else 0
    valid_draws = all(
        len(draw) == len(set(draw)) == required and set(draw) <= allowed
        for draw in draws
    )
    _require(required > 0 and valid_draws, "Invalid frozen ALPN draws")
    eligible = tuple(sorted(set(range(nodes)) - excluded, key=body_ids.__getitem__))
    pool = coordinates[
        sorted(set(selection.alpn) | set(eligible), key=body_ids.__getitem__)
    ]
    mean, scale = np.empty(len(COORDINATES)), np.empty(len(COORDINATES))
    _ = pool.mean(axis=0, out=mean)
    _ = pool.std(axis=0, out=scale)
    scale[scale == 0] = 1
    normalized = (coordinates - mean) / scale
    controls: list[MatchedControl] = []
    if len(eligible) >= required:
        for draw, indices in enumerate(draws):
            target = tuple(sorted(indices, key=body_ids.__getitem__))
            controls.extend(
                _control(target, eligible, normalized, comparator, draw)
                for comparator in ("S", "M")
            )
    overlaps: list[Overlap] = []
    memberships = [set(control.control_indices) for control in controls]
    for left, right in combinations(range(len(controls)), 2):
        shared = memberships[left] & memberships[right]
        overlaps.append(
            Overlap(left, right, tuple(sorted(shared, key=body_ids.__getitem__)))
        )
    return MatchingEvidence(
        eligible,
        tuple(sorted(excluded, key=body_ids.__getitem__)),
        required,
        _floats(mean),
        _floats(scale),
        tuple(controls),
        tuple(overlaps),
    )
