"""Pure fixed-sequence ALPN causal decision evaluation.

Workers compute and seal sufficient statistics and source/story contrasts. This module
only validates those matrices and applies the frozen balance, assay, sign, and
ordered intersection-union gates.
"""

import statistics
from math import comb, fsum
from typing import assert_never

import scripts.alpn_causal_types as types
from scripts.alpn_causal_assay import assay_decision, balance_passes


def exact_one_sided_sign_probability(values: tuple[float, ...]) -> float:
    """Return one-sided exact sign probability, counting ties against success."""
    if not values:
        types.raise_invalid("An exact sign probability requires at least one story")
    successes = sum(value > 0.0 for value in values)
    return sum(
        comb(len(values), count) for count in range(successes, len(values) + 1)
    ) / (1 << len(values))


def _contrast_index(
    contrasts: tuple[types.SourceStoryContrasts, ...],
) -> tuple[
    dict[tuple[int, types.Wiring, str], types.SourceStoryContrasts],
    tuple[str, ...],
    dict[str, int],
]:
    index: dict[tuple[int, types.Wiring, str], types.SourceStoryContrasts] = {
        (item.source.seed, item.source.wiring, item.story): item for item in contrasts
    }
    if len(index) != len(contrasts):
        types.raise_invalid("The contrast matrix contains duplicate source/story rows")
    stories_by_source = {
        source: {story for seed, wiring, story in index if (seed, wiring) == source}
        for source in types.SOURCE_KEYS
    }
    if any(len(stories) != types.STORY_COUNT for stories in stories_by_source.values()):
        types.raise_invalid(
            "The contrast matrix must contain 68 stories for every source"
        )
    story_sets = tuple(stories_by_source.values())
    if any(stories != story_sets[0] for stories in story_sets[1:]):
        types.raise_invalid(
            "The contrast matrix must use the same stories for every source"
        )
    if len(contrasts) != len(types.SOURCE_KEYS) * types.STORY_COUNT:
        types.raise_invalid("The contrast matrix has an unexpected number of rows")
    stories = tuple(sorted(story_sets[0]))
    targets = {
        story: index[(types.SEEDS[0], "real", story)].targets for story in stories
    }
    if sum(targets.values()) != types.FRESH_TARGET_COUNT:
        types.raise_invalid("Each source must contain exactly 12,242 fresh targets")
    if any(
        index[(seed, wiring, story)].targets != targets[story]
        for seed, wiring in types.SOURCE_KEYS
        for story in stories
    ):
        types.raise_invalid("Every source must use identical per-story target counts")
    return index, stories, targets


def _damage(row: types.SourceStoryContrasts, comparator: types.Comparator) -> float:
    match comparator:
        case "S":
            return row.alpn_damage - row.control_s_damage
        case "M":
            return row.alpn_damage - row.control_m_damage
        case _:
            assert_never(comparator)


def _components(
    index: dict[tuple[int, types.Wiring, str], types.SourceStoryContrasts],
    stories: tuple[str, ...],
    targets: dict[str, int],
) -> dict[types.ComponentName, types.ComponentValues]:
    def real(seed: int, story: str, name: types.ComponentName) -> float:
        row = index[(seed, "real", story)]
        values: dict[types.ComponentName, float] = {
            "A_S": row.accessibility_s,
            "A_M": row.accessibility_m,
            "Q": row.unigram_advantage,
            "C_ALPN": row.alpn_damage,
            "D_S": _damage(row, "S"),
            "D_M": _damage(row, "M"),
        }
        return values[name]

    values: dict[types.ComponentName, types.ComponentValues] = {}
    for name in ("A_S", "A_M", "Q", "C_ALPN", "D_S", "D_M"):
        values[name] = types.ComponentValues(
            tuple(
                statistics.fmean(real(seed, story, name) for seed in types.SEEDS)
                for story in stories
            ),
            tuple(
                fsum(real(seed, story, name) * targets[story] for story in stories)
                / types.FRESH_TARGET_COUNT
                for seed in types.SEEDS
            ),
        )
    moderation_components: tuple[tuple[types.ComponentName, types.Comparator], ...] = (
        ("K_S", "S"),
        ("K_M", "M"),
    )
    for name, comparator in moderation_components:

        def moderation(
            seed: int, story: str, comparator: types.Comparator = comparator
        ) -> float:
            return _damage(index[(seed, "real", story)], comparator) - _damage(
                index[(seed, "shuffled", story)], comparator
            )

        values[name] = types.ComponentValues(
            tuple(
                statistics.fmean(moderation(seed, story) for seed in types.SEEDS)
                for story in stories
            ),
            tuple(
                fsum(moderation(seed, story) * targets[story] for story in stories)
                / types.FRESH_TARGET_COUNT
                for seed in types.SEEDS
            ),
        )
    return values


def _component(
    name: types.ComponentName,
    values: types.ComponentValues,
    threshold: float,
) -> types.ComponentDecision:
    """Apply the practical and exact-sign requirements to one component."""
    all_positive = all(value > 0.0 for value in values.seed)
    median = statistics.median(values.seed)
    return types.ComponentDecision(
        component=name,
        story_contrasts=values.story,
        seed_contrasts=values.seed,
        exact_one_sided_sign_probability=exact_one_sided_sign_probability(values.story),
        all_seeds_positive=all_positive,
        median_seed_contrast=median,
        practical_threshold=threshold,
        practical_pass=all_positive and median >= threshold,
    )


def _gate(
    name: types.GateName,
    components: tuple[types.ComponentDecision, ...],
    tested: bool,
) -> types.GateDecision:
    """Apply an intersection-union gate if the preceding gate passed."""
    probability = max(item.exact_one_sided_sign_probability for item in components)
    return types.GateDecision(
        gate=name,
        components=components,
        intersection_union_probability=probability,
        inferentially_tested=tested,
        passed=tested
        and probability <= types.ALPHA
        and all(item.practical_pass for item in components),
    )


def evaluate_causal_experiment(
    evidence: types.CausalExperimentEvidence,
) -> types.CausalExperimentSummary:
    """Apply balance, assay, and the documented ordered ALPN gates."""
    if not balance_passes(evidence.balance):
        return types.CausalExperimentSummary(
            verdict="insufficient_common_support",
            balance_passed=False,
            assay=None,
            gates=(),
        )
    assay = assay_decision(evidence.assay)
    if not assay.passed:
        return types.CausalExperimentSummary(
            verdict="assay_invalid", balance_passed=True, assay=assay, gates=()
        )
    index, stories, targets = _contrast_index(evidence.contrasts)
    values = _components(index, stories, targets)
    accessibility = _gate(
        "accessibility",
        tuple(
            _component(name, values[name], types.ACCESSIBILITY_THRESHOLD)
            for name in ("A_S", "A_M", "Q")
        ),
        tested=True,
    )
    causal = _gate(
        "causal_specificity",
        tuple(
            _component(name, values[name], types.CAUSAL_THRESHOLD)
            for name in ("C_ALPN", "D_S", "D_M")
        ),
        accessibility.passed,
    )
    moderation = _gate(
        "wiring_moderation",
        tuple(
            _component(name, values[name], types.CAUSAL_THRESHOLD)
            for name in ("K_S", "K_M")
        ),
        causal.passed,
    )
    verdict = (
        "joint_success"
        if moderation.passed
        else "accessibility_causal"
        if causal.passed
        else "accessibility_only"
        if accessibility.passed
        else "no_accessibility_specificity"
    )
    return types.CausalExperimentSummary(
        verdict=verdict,
        balance_passed=True,
        assay=assay,
        gates=(accessibility, causal, moderation),
    )
