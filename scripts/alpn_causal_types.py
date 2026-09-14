"""Immutable protocol, evidence, and decision records for the ALPN experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NoReturn, TypeAlias, assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator

Wiring: TypeAlias = Literal["real", "shuffled"]
Comparator: TypeAlias = Literal["S", "M"]
GroupName: TypeAlias = Literal["alpn", "S", "M"]
InterventionArm: TypeAlias = Literal["baseline", "alpn", "S", "M"]
CausalVerdict: TypeAlias = Literal[
    "insufficient_common_support",
    "assay_invalid",
    "joint_success",
    "accessibility_causal",
    "accessibility_only",
    "no_accessibility_specificity",
]
GateName: TypeAlias = Literal[
    "accessibility", "causal_specificity", "wiring_moderation"
]
ComponentName: TypeAlias = Literal[
    "A_S",
    "A_M",
    "Q",
    "C_ALPN",
    "D_S",
    "D_M",
    "K_S",
    "K_M",
]

SEEDS: Final[tuple[int, ...]] = (7, 8, 9, 10, 11, 12)
WIRINGS: Final[tuple[Wiring, ...]] = ("real", "shuffled")
SOURCE_KEYS: Final[frozenset[tuple[int, Wiring]]] = frozenset(
    (seed, wiring) for seed in SEEDS for wiring in WIRINGS
)
COMPARATORS: Final[tuple[Comparator, ...]] = ("S", "M")
DRAWS_PER_GROUP: Final = 5
STORY_COUNT: Final = 68
FRESH_TARGET_COUNT: Final = 12_242
ALPHA: Final = 0.05
ACCESSIBILITY_THRESHOLD: Final = 0.10
CAUSAL_THRESHOLD: Final = 0.01
BALANCE_SMD_THRESHOLD: Final = 0.10
BALANCE_KS_THRESHOLD: Final = 0.20
READOUT_REPLACEMENT_THRESHOLD: Final = 0.01


def raise_invalid(message: str) -> NoReturn:
    """Raise a stable schema-validation error."""
    raise ValueError(message)


class FrozenModel(BaseModel):
    """Reject non-finite values and prevent mutation at the data boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, allow_inf_nan=False)


class SourceIdentity(FrozenModel):
    """One frozen source checkpoint."""

    seed: int
    wiring: Wiring


class GroupIdentity(FrozenModel):
    """One realized ALPN or matched-control probe draw."""

    source: SourceIdentity
    group: GroupName
    draw: int = Field(ge=0, lt=DRAWS_PER_GROUP)


class ControlIdentity(FrozenModel):
    """One realized S or M control draw."""

    source: SourceIdentity
    comparator: Comparator
    draw: int = Field(ge=0, lt=DRAWS_PER_GROUP)


class InterventionIdentity(FrozenModel):
    """One baseline or centered-outgoing-state intervention arm."""

    source: SourceIdentity
    arm: InterventionArm
    draw: int | None = None

    @model_validator(mode="after")
    def _validate_draw(self) -> InterventionIdentity:
        match self.arm:
            case "baseline":
                if self.draw is not None:
                    raise_invalid("A baseline intervention cannot have a draw")
            case "alpn" | "S" | "M":
                if self.draw is None or not 0 <= self.draw < DRAWS_PER_GROUP:
                    raise_invalid(
                        "A non-baseline intervention requires draw 0 through 4"
                    )
            case _:
                assert_never(self.arm)
        return self


class AlpnCausalProtocol(FrozenModel):
    """The sealed source, fresh-story, control, and arm identities."""

    format: Literal["flyrl-alpn-causal-v1"] = "flyrl-alpn-causal-v1"
    sources: tuple[SourceIdentity, ...]
    fresh_story_ids: tuple[str, ...]
    controls: tuple[ControlIdentity, ...]
    interventions: tuple[InterventionIdentity, ...]
    alpha: float = ALPHA
    accessibility_threshold: float = ACCESSIBILITY_THRESHOLD
    causal_threshold: float = CAUSAL_THRESHOLD

    @model_validator(mode="after")
    def _validate_completeness(self) -> AlpnCausalProtocol:
        if self.alpha != ALPHA:
            raise_invalid("Protocol alpha must equal the frozen constant")
        if self.accessibility_threshold != ACCESSIBILITY_THRESHOLD:
            raise_invalid(
                "Protocol accessibility_threshold must equal the frozen constant"
            )
        if self.causal_threshold != CAUSAL_THRESHOLD:
            raise_invalid("Protocol causal_threshold must equal the frozen constant")
        sources = {(item.seed, item.wiring) for item in self.sources}
        expected_sources = {(seed, wiring) for seed in SEEDS for wiring in WIRINGS}
        if len(self.sources) != len(expected_sources) or sources != expected_sources:
            raise_invalid(
                "Protocol sources must be the exact twelve frozen checkpoints"
            )
        if (
            len(self.fresh_story_ids) != STORY_COUNT
            or len(set(self.fresh_story_ids)) != STORY_COUNT
        ):
            raise_invalid("Protocol requires 68 unique fresh story identities")
        expected_controls = {
            (seed, wiring, comparator, draw)
            for seed, wiring in expected_sources
            for comparator in COMPARATORS
            for draw in range(DRAWS_PER_GROUP)
        }
        controls = {
            (item.source.seed, item.source.wiring, item.comparator, item.draw)
            for item in self.controls
        }
        if (
            len(self.controls) != len(expected_controls)
            or controls != expected_controls
        ):
            raise_invalid(
                "Protocol controls must contain every S/M draw for each source"
            )
        expected_arms = {
            (seed, wiring, "baseline", None) for seed, wiring in expected_sources
        } | {
            (seed, wiring, arm, draw)
            for seed, wiring in expected_sources
            for arm in ("alpn", "S", "M")
            for draw in range(DRAWS_PER_GROUP)
        }
        arms = {
            (item.source.seed, item.source.wiring, item.arm, item.draw)
            for item in self.interventions
        }
        if len(self.interventions) != len(expected_arms) or arms != expected_arms:
            raise_invalid(
                "Protocol interventions must contain baseline plus fifteen arms"
            )
        return self


class StorySufficientStatistic(FrozenModel):
    """Token-count and NLL-sum sufficient statistics for one fresh story."""

    story: str = Field(min_length=1)
    targets: int = Field(gt=0)
    nll_sum: float = Field(ge=0.0)

    @property
    def nll(self) -> float:
        """Return the story's token-weighted NLL."""
        return self.nll_sum / self.targets


class ProbeStoryMetric(FrozenModel):
    """Fresh selected-state probe sufficient statistics."""

    group: GroupIdentity
    statistic: StorySufficientStatistic


class OriginalReadoutStoryMetric(FrozenModel):
    """Fresh original-readout sufficient statistics for one intervention."""

    intervention: InterventionIdentity
    statistic: StorySufficientStatistic


class UnigramStoryMetric(FrozenModel):
    """Fresh training-unigram sufficient statistics for one source."""

    source: SourceIdentity
    statistic: StorySufficientStatistic


class BalanceCheck(FrozenModel):
    """Metric-free common-support result for one control draw."""

    control: ControlIdentity
    maximum_standardized_mean_difference: float = Field(ge=0.0)
    maximum_ks_distance: float = Field(ge=0.0, le=1.0)
    passed: bool


class AssayValidity(FrozenModel):
    """Fresh source-level unigram gains and readout replacement damage."""

    source: SourceIdentity
    original_head_gain: float
    trained_readout_probe_gain: float
    readout_replacement_damage: float


class SourceStoryContrasts(FrozenModel):
    """Per-source/story A, Q, and causal-damage sufficient contrasts."""

    source: SourceIdentity
    story: str = Field(min_length=1)
    targets: int = Field(gt=0)
    accessibility_s: float
    accessibility_m: float
    unigram_advantage: float
    alpn_damage: float
    control_s_damage: float
    control_m_damage: float


class CausalExperimentEvidence(FrozenModel):
    """Sealed evidence consumed by the pure fixed-sequence evaluator."""

    balance: tuple[BalanceCheck, ...]
    assay: tuple[AssayValidity, ...]
    contrasts: tuple[SourceStoryContrasts, ...]


class AssayDecision(FrozenModel):
    """All-source assay validity decision."""

    original_head_gains: tuple[float, ...]
    trained_readout_probe_gains: tuple[float, ...]
    readout_replacement_damages: tuple[float, ...]
    real_median_original_head_gain: float
    shuffled_median_original_head_gain: float
    real_median_trained_readout_probe_gain: float
    shuffled_median_trained_readout_probe_gain: float
    real_median_readout_replacement_damage: float
    shuffled_median_readout_replacement_damage: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ComponentValues:
    """The story and seed aggregation levels for one contrast component."""

    story: tuple[float, ...]
    seed: tuple[float, ...]


class ComponentDecision(FrozenModel):
    """Story-sign and six-seed practical decision for one contrast."""

    component: ComponentName
    story_contrasts: tuple[float, ...]
    seed_contrasts: tuple[float, ...]
    exact_one_sided_sign_probability: float
    all_seeds_positive: bool
    median_seed_contrast: float
    practical_threshold: float
    practical_pass: bool


class GateDecision(FrozenModel):
    """One ordered intersection-union gate decision."""

    gate: GateName
    components: tuple[ComponentDecision, ...]
    intersection_union_probability: float
    inferentially_tested: bool
    passed: bool


class CausalExperimentSummary(FrozenModel):
    """Terminal verdict and evaluated-or-descriptive gate evidence."""

    verdict: CausalVerdict
    balance_passed: bool
    assay: AssayDecision | None
    gates: tuple[GateDecision, ...]
