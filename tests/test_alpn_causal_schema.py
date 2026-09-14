"""Fixed-sequence decision tests for the frozen ALPN causal experiment."""

import statistics
from dataclasses import dataclass

import pytest

import scripts.alpn_causal_schema as schema
import scripts.alpn_causal_types as types


@dataclass(frozen=True, slots=True)
class _EvidenceOptions:
    balance_passed: bool = True
    assay_valid: bool = True
    accessibility_positive_stories: int = types.STORY_COUNT
    causal_positive: bool = True
    moderation_positive: bool = True
    threshold_value: float = 0.10


def _source(seed: int, wiring: types.Wiring) -> types.SourceIdentity:
    return types.SourceIdentity(seed=seed, wiring=wiring)


def make_evidence(
    options: _EvidenceOptions | None = None,
) -> types.CausalExperimentEvidence:
    options = _EvidenceOptions() if options is None else options
    balance = tuple(
        types.BalanceCheck(
            control=types.ControlIdentity(
                source=_source(seed, wiring), comparator=comparator, draw=draw
            ),
            maximum_standardized_mean_difference=0.10,
            maximum_ks_distance=0.20,
            passed=options.balance_passed,
        )
        for seed in types.SEEDS
        for wiring in types.WIRINGS
        for comparator in types.COMPARATORS
        for draw in range(types.DRAWS_PER_GROUP)
    )
    gain = 0.10 if options.assay_valid else 0.0
    assay = tuple(
        types.AssayValidity(
            source=_source(seed, wiring),
            original_head_gain=gain,
            trained_readout_probe_gain=gain,
            readout_replacement_damage=0.01,
        )
        for seed in types.SEEDS
        for wiring in types.WIRINGS
    )
    contrasts = tuple(
        types.SourceStoryContrasts(
            source=_source(seed, wiring),
            story=f"story-{story:02d}",
            targets=182 if story == types.STORY_COUNT - 1 else 180,
            accessibility_s=(
                options.threshold_value
                if story < options.accessibility_positive_stories
                else -options.threshold_value
            ),
            accessibility_m=(
                options.threshold_value
                if story < options.accessibility_positive_stories
                else -options.threshold_value
            ),
            unigram_advantage=(
                options.threshold_value
                if story < options.accessibility_positive_stories
                else -options.threshold_value
            ),
            alpn_damage=0.01 if options.causal_positive else -0.01,
            control_s_damage=0.0 if wiring == "real" else 0.01,
            control_m_damage=0.0 if wiring == "real" else 0.01,
        )
        for seed in types.SEEDS
        for wiring in types.WIRINGS
        for story in range(types.STORY_COUNT)
    )
    if not options.moderation_positive:
        contrasts = tuple(
            row.model_copy(
                update={"control_s_damage": -0.01, "control_m_damage": -0.01}
                if row.source.wiring == "shuffled"
                else {}
            )
            for row in contrasts
        )
    return types.CausalExperimentEvidence(
        balance=balance, assay=assay, contrasts=contrasts
    )


def test_sign_probability_counts_ties_against_success() -> None:
    assert schema.exact_one_sided_sign_probability((1.0, 0.0, -1.0)) == 0.875


def test_sign_probability_distinguishes_41_and_42_positive_stories() -> None:
    p41 = schema.exact_one_sided_sign_probability((1.0,) * 41 + (-1.0,) * 27)
    p42 = schema.exact_one_sided_sign_probability((1.0,) * 42 + (-1.0,) * 26)
    assert p41 > types.ALPHA
    assert p42 <= types.ALPHA


def test_fixed_sequence_returns_every_documented_verdict() -> None:
    assert (
        schema.evaluate_causal_experiment(
            make_evidence(_EvidenceOptions(balance_passed=False))
        ).verdict
        == "insufficient_common_support"
    )
    assert (
        schema.evaluate_causal_experiment(
            make_evidence(_EvidenceOptions(assay_valid=False))
        ).verdict
        == "assay_invalid"
    )
    assert schema.evaluate_causal_experiment(make_evidence()).verdict == "joint_success"
    assert (
        schema.evaluate_causal_experiment(
            make_evidence(_EvidenceOptions(moderation_positive=False))
        ).verdict
        == "accessibility_causal"
    )
    assert (
        schema.evaluate_causal_experiment(
            make_evidence(_EvidenceOptions(causal_positive=False))
        ).verdict
        == "accessibility_only"
    )
    assert (
        schema.evaluate_causal_experiment(
            make_evidence(_EvidenceOptions(accessibility_positive_stories=41))
        ).verdict
        == "no_accessibility_specificity"
    )


def test_assay_requires_passing_readout_replacement_control() -> None:
    summary = schema.evaluate_causal_experiment(make_evidence())
    assert summary.assay is not None
    assert summary.assay.passed
    assert summary.assay.readout_replacement_damages == (0.01,) * 12
    assert summary.assay.real_median_readout_replacement_damage == 0.01
    assert summary.assay.shuffled_median_readout_replacement_damage == 0.01

    experiment = make_evidence()
    failed = experiment.assay[0].model_copy(update={"readout_replacement_damage": 0.0})
    invalid = experiment.model_copy(update={"assay": (failed, *experiment.assay[1:])})
    assert schema.evaluate_causal_experiment(invalid).verdict == "assay_invalid"


def test_assay_rejects_a_subthreshold_wiring_median_despite_pooled_pass() -> None:
    experiment = make_evidence()
    assay = tuple(
        item.model_copy(
            update={
                "original_head_gain": 0.20 if item.source.wiring == "real" else 0.09
            }
        )
        for item in experiment.assay
    )
    summary = schema.evaluate_causal_experiment(
        experiment.model_copy(update={"assay": assay})
    )
    assert summary.assay is not None
    assert summary.assay.real_median_original_head_gain == 0.20
    assert summary.assay.shuffled_median_original_head_gain == 0.09
    assert (
        statistics.median(summary.assay.original_head_gains)
        >= types.ACCESSIBILITY_THRESHOLD
    )
    assert summary.verdict == "assay_invalid"


def test_threshold_equality_and_sequential_stopping() -> None:
    summary = schema.evaluate_causal_experiment(
        make_evidence(_EvidenceOptions(threshold_value=0.10))
    )
    assert summary.gates[0].passed
    assert all(component.practical_pass for component in summary.gates[0].components)

    stopped = schema.evaluate_causal_experiment(
        make_evidence(_EvidenceOptions(accessibility_positive_stories=41))
    )
    assert not stopped.gates[0].passed
    assert not stopped.gates[1].inferentially_tested
    assert not stopped.gates[2].inferentially_tested
    assert all(not gate.passed for gate in stopped.gates[1:])


def test_balance_observations_cannot_be_marked_passing_above_calipers() -> None:
    experiment = make_evidence()
    failed = experiment.balance[0].model_copy(
        update={"maximum_standardized_mean_difference": 0.100001}
    )
    unbalanced = experiment.model_copy(
        update={"balance": (failed, *experiment.balance[1:])}
    )
    assert (
        schema.evaluate_causal_experiment(unbalanced).verdict
        == "insufficient_common_support"
    )


def test_rejects_malformed_contrast_matrix() -> None:
    experiment = make_evidence()
    malformed = experiment.model_copy(update={"contrasts": experiment.contrasts[:-1]})
    with pytest.raises(ValueError, match="contrast matrix"):
        _ = schema.evaluate_causal_experiment(malformed)


def test_evidence_options_are_slotted() -> None:
    assert hasattr(_EvidenceOptions, "__slots__")
