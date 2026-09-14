"""Target-weighted ALPN contrast aggregation regressions."""

import statistics

import pytest

import scripts.alpn_causal_schema as schema
import scripts.alpn_causal_types as types
from tests.test_alpn_causal_schema import make_evidence


def test_seed_contrasts_are_token_weighted_not_equal_story_averages() -> None:
    experiment = make_evidence()
    contrasts = tuple(
        row.model_copy(
            update={
                "targets": types.FRESH_TARGET_COUNT - (types.STORY_COUNT - 1)
                if row.story == "story-00"
                else 1,
                "accessibility_s": -1.0 if row.story == "story-00" else 0.1,
                "accessibility_m": -1.0 if row.story == "story-00" else 0.1,
                "unigram_advantage": -1.0 if row.story == "story-00" else 0.1,
            }
        )
        for row in experiment.contrasts
    )
    summary = schema.evaluate_causal_experiment(
        experiment.model_copy(update={"contrasts": contrasts})
    )
    accessibility = summary.gates[0].components[0]
    assert statistics.fmean(accessibility.story_contrasts) > 0.0
    assert all(value < 0.0 for value in accessibility.seed_contrasts)
    assert summary.verdict == "no_accessibility_specificity"


def test_rejects_inconsistent_per_story_target_counts() -> None:
    experiment = make_evidence()
    contrasts = tuple(
        row.model_copy(update={"targets": row.targets + 1})
        if row.source.wiring == "shuffled" and row.story == "story-00"
        else row
        for row in experiment.contrasts
    )
    with pytest.raises(ValueError, match="identical per-story target counts"):
        _ = schema.evaluate_causal_experiment(
            experiment.model_copy(update={"contrasts": contrasts})
        )


def test_rejects_wrong_per_source_target_total() -> None:
    experiment = make_evidence()
    contrasts = tuple(
        row.model_copy(update={"targets": row.targets + 1})
        if row.story == "story-00"
        else row
        for row in experiment.contrasts
    )
    with pytest.raises(ValueError, match="12,242 fresh targets"):
        _ = schema.evaluate_causal_experiment(
            experiment.model_copy(update={"contrasts": contrasts})
        )
