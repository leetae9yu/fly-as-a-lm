"""Pure pre-gate balance and assay eligibility decisions for ALPN evidence."""

from __future__ import annotations

import statistics

import scripts.alpn_causal_types as types


def balance_passes(checks: tuple[types.BalanceCheck, ...]) -> bool:
    """Validate the promised controls and return their metric-free balance status."""
    if any(
        not check.passed
        or check.maximum_standardized_mean_difference > types.BALANCE_SMD_THRESHOLD
        or check.maximum_ks_distance > types.BALANCE_KS_THRESHOLD
        for check in checks
    ):
        return False
    expected = {
        (seed, wiring, comparator, draw)
        for seed, wiring in types.SOURCE_KEYS
        for comparator in types.COMPARATORS
        for draw in range(types.DRAWS_PER_GROUP)
    }
    actual = {
        (
            item.control.source.seed,
            item.control.source.wiring,
            item.control.comparator,
            item.control.draw,
        )
        for item in checks
    }
    if len(checks) != len(expected) or actual != expected:
        types.raise_invalid(
            "Balance matrix differs from the promised 12-source controls"
        )
    return True


def _medians(items: tuple[types.AssayValidity, ...]) -> tuple[float, float, float]:
    """Return original-head, reused-probe, and replacement-damage medians."""
    return (
        statistics.median(tuple(item.original_head_gain for item in items)),
        statistics.median(tuple(item.trained_readout_probe_gain for item in items)),
        statistics.median(tuple(item.readout_replacement_damage for item in items)),
    )


def assay_decision(assay: tuple[types.AssayValidity, ...]) -> types.AssayDecision:
    """Apply all-source and wiring-specific assay validity requirements."""
    expected = set(types.SOURCE_KEYS)
    by_source = {(item.source.seed, item.source.wiring): item for item in assay}
    if len(assay) != len(expected) or set(by_source) != expected:
        types.raise_invalid("Assay matrix differs from the promised 12 sources")
    ordered = tuple(by_source[key] for key in sorted(expected))
    original = tuple(item.original_head_gain for item in ordered)
    trained = tuple(item.trained_readout_probe_gain for item in ordered)
    replacement = tuple(item.readout_replacement_damage for item in ordered)
    real = tuple(item for item in ordered if item.source.wiring == "real")
    shuffled = tuple(item for item in ordered if item.source.wiring == "shuffled")
    real_original, real_trained, real_replacement = _medians(real)
    shuffled_original, shuffled_trained, shuffled_replacement = _medians(shuffled)
    return types.AssayDecision(
        original_head_gains=original,
        trained_readout_probe_gains=trained,
        readout_replacement_damages=replacement,
        real_median_original_head_gain=real_original,
        shuffled_median_original_head_gain=shuffled_original,
        real_median_trained_readout_probe_gain=real_trained,
        shuffled_median_trained_readout_probe_gain=shuffled_trained,
        real_median_readout_replacement_damage=real_replacement,
        shuffled_median_readout_replacement_damage=shuffled_replacement,
        passed=(
            all(value > 0.0 for value in original)
            and all(value > 0.0 for value in trained)
            and all(value > 0.0 for value in replacement)
            and min(real_original, shuffled_original, real_trained, shuffled_trained)
            >= types.ACCESSIBILITY_THRESHOLD
            and min(real_replacement, shuffled_replacement)
            >= types.READOUT_REPLACEMENT_THRESHOLD
        ),
    )
