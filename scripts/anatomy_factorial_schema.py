"""Immutable outcomes and prospective decisions for anatomy-aware ports."""

import statistics
from typing import ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

Wiring: TypeAlias = Literal["real", "shuffled"]
FactorialPortPolicy: TypeAlias = Literal[
    "alpn_mbon",
    "alpn_random",
    "random_mbon",
    "random_random",
]
ConditionKey: TypeAlias = tuple[Wiring, FactorialPortPolicy]
ConditionName: TypeAlias = Literal[
    "real-alpn_mbon",
    "real-alpn_random",
    "real-random_mbon",
    "real-random_random",
    "shuffled-alpn_mbon",
    "shuffled-random_random",
]
EndpointStatus: TypeAlias = Literal[
    "pass",
    "positive_subthreshold",
    "reversed",
    "inconclusive",
]


class FactorialCondition(BaseModel):
    """One named wiring and port-policy condition."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    name: ConditionName
    wiring: Wiring
    port_policy: FactorialPortPolicy


class FactorialSeedPlan(BaseModel):
    """One fresh seed and its six-condition counterbalanced order."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    seed: int
    order: tuple[ConditionName, ...]


class FactorialProtocol(BaseModel):
    """Complete source, matrix and decision contract sealed before execution."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    format: Literal["flyrl-anatomy-factorial-v1"] = "flyrl-anatomy-factorial-v1"
    graph: str
    graph_sha256: str
    graph_nodes: int
    graph_edges: int
    port_manifest: str
    port_manifest_sha256: str
    port_arrays: str
    port_arrays_sha256: str
    annotations_sha256: str
    alpn_count: int
    mbon_count: int
    kenyon_count: int
    corpus_fingerprint: str
    corpus_sha256: str
    source_sha256: str
    worker_sha256: str
    conditions: tuple[FactorialCondition, ...]
    seeds: tuple[FactorialSeedPlan, ...]
    context: int = 64
    batch_size: int = 8
    updates: int = 1000
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    leak: float = 0.5
    initial_gain: float = 0.9
    sensory_neurons: int = 313
    readout_neurons: int = 97
    trainable_codes: bool = True
    checkpoint_steps: int = 100
    sample_length: int = 64
    activation_neurons: int = 64
    prompt: str = "Once upon a time"
    primary_threshold: float = 0.10
    topology_threshold: float = 0.05
    heldout_policy: str = "sealed until all thirty-six conditions complete"


class FactorialConditionResult(BaseModel):
    """One sealed condition's final held-out result."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    seed: int
    wiring: Wiring
    port_policy: FactorialPortPolicy
    test_nll: float
    test_accuracy: float


class FactorialSeedRow(BaseModel):
    """Six condition metrics and predeclared contrasts for one seed."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    seed: int
    conditions: tuple[FactorialConditionResult, ...]
    primary_gain: float
    shuffled_gain: float
    topology_interaction: float
    input_gain: float
    output_gain: float
    input_output_interaction: float


class FactorialAnalysis(BaseModel):
    """Across-seed practical thresholds and exact sign decisions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    verdict: Literal[
        "joint_success",
        "primary_only",
        "interaction_only",
        "neither_gate_passed",
    ]
    advance: bool
    primary_status: EndpointStatus
    topology_status: EndpointStatus
    primary_unanimous_positive: bool
    topology_unanimous_positive: bool
    median_primary_gain: float
    mean_primary_gain: float
    primary_range: tuple[float, float]
    median_topology_interaction: float
    mean_topology_interaction: float
    topology_range: tuple[float, float]
    primary_two_sided_unanimous_sign_p: float | None
    topology_two_sided_unanimous_sign_p: float | None


class FactorialSummary(BaseModel):
    """Portable rows and decision emitted only after all 36 conditions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    format: Literal["flyrl-anatomy-factorial-results-v1"] = (
        "flyrl-anatomy-factorial-results-v1"
    )
    rows: tuple[FactorialSeedRow, ...]
    analysis: FactorialAnalysis


def _seed_row(
    seed: int,
    conditions: tuple[FactorialConditionResult, ...],
) -> FactorialSeedRow:
    by_key: dict[ConditionKey, FactorialConditionResult] = {
        (condition.wiring, condition.port_policy): condition for condition in conditions
    }
    expected: set[ConditionKey] = {
        ("real", "alpn_mbon"),
        ("real", "alpn_random"),
        ("real", "random_mbon"),
        ("real", "random_random"),
        ("shuffled", "alpn_mbon"),
        ("shuffled", "random_random"),
    }
    if len(conditions) != len(expected) or set(by_key) != expected:
        message = f"Seed {seed} does not contain the exact six-condition matrix"
        raise ValueError(message)
    real_am = by_key[("real", "alpn_mbon")].test_nll
    real_ar = by_key[("real", "alpn_random")].test_nll
    real_rm = by_key[("real", "random_mbon")].test_nll
    real_rr = by_key[("real", "random_random")].test_nll
    shuffled_am = by_key[("shuffled", "alpn_mbon")].test_nll
    shuffled_rr = by_key[("shuffled", "random_random")].test_nll
    primary = real_rr - real_am
    shuffled = shuffled_rr - shuffled_am
    return FactorialSeedRow(
        seed=seed,
        conditions=tuple(
            sorted(
                conditions,
                key=lambda condition: (condition.wiring, condition.port_policy),
            )
        ),
        primary_gain=primary,
        shuffled_gain=shuffled,
        topology_interaction=primary - shuffled,
        input_gain=((real_rm - real_am) + (real_rr - real_ar)) / 2,
        output_gain=((real_ar - real_am) + (real_rr - real_rm)) / 2,
        input_output_interaction=real_ar + real_rm - real_am - real_rr,
    )


def _endpoint_status(
    values: tuple[float, ...],
    threshold: float,
) -> EndpointStatus:
    if all(value > 0 for value in values):
        return (
            "pass"
            if statistics.median(values) >= threshold
            else "positive_subthreshold"
        )
    if all(value < 0 for value in values):
        return "reversed"
    return "inconclusive"


def evaluate_factorial(
    results: tuple[FactorialConditionResult, ...],
    primary_threshold: float,
    topology_threshold: float,
) -> FactorialSummary:
    """Apply the frozen six-seed port and topology-interaction gates."""
    seeds = tuple(sorted({result.seed for result in results}))
    if seeds != tuple(range(7, 13)):
        message = "Factorial decision requires exactly fresh seeds 7-12"
        raise ValueError(message)
    rows = tuple(
        _seed_row(
            seed,
            tuple(result for result in results if result.seed == seed),
        )
        for seed in seeds
    )
    primary = tuple(row.primary_gain for row in rows)
    topology = tuple(row.topology_interaction for row in rows)
    primary_positive = all(value > 0 for value in primary)
    topology_positive = all(value > 0 for value in topology)
    primary_status = _endpoint_status(primary, primary_threshold)
    topology_status = _endpoint_status(topology, topology_threshold)
    primary_pass = primary_status == "pass"
    topology_pass = topology_status == "pass"
    if primary_pass and topology_pass:
        verdict = "joint_success"
    elif primary_pass:
        verdict = "primary_only"
    elif topology_pass:
        verdict = "interaction_only"
    else:
        verdict = "neither_gate_passed"
    return FactorialSummary(
        rows=rows,
        analysis=FactorialAnalysis(
            verdict=verdict,
            advance=primary_pass and topology_pass,
            primary_status=primary_status,
            topology_status=topology_status,
            primary_unanimous_positive=primary_positive,
            topology_unanimous_positive=topology_positive,
            median_primary_gain=statistics.median(primary),
            mean_primary_gain=statistics.mean(primary),
            primary_range=(min(primary), max(primary)),
            median_topology_interaction=statistics.median(topology),
            mean_topology_interaction=statistics.mean(topology),
            topology_range=(min(topology), max(topology)),
            primary_two_sided_unanimous_sign_p=(
                0.03125
                if primary_positive or all(value < 0 for value in primary)
                else None
            ),
            topology_two_sided_unanimous_sign_p=(
                0.03125
                if topology_positive or all(value < 0 for value in topology)
                else None
            ),
        ),
    )
