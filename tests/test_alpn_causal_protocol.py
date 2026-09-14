"""Protocol and data-boundary tests for the frozen ALPN experiment."""

import math

import pytest
from pydantic import ValidationError

import scripts.alpn_causal_types as types


def _source(seed: int, wiring: types.Wiring) -> types.SourceIdentity:
    return types.SourceIdentity(seed=seed, wiring=wiring)


def _protocol(
    sources: tuple[types.SourceIdentity, ...] | None = None,
    stories: tuple[str, ...] | None = None,
    controls: tuple[types.ControlIdentity, ...] | None = None,
    interventions: tuple[types.InterventionIdentity, ...] | None = None,
) -> types.AlpnCausalProtocol:
    sources = sources or tuple(
        _source(seed, wiring) for seed in types.SEEDS for wiring in types.WIRINGS
    )
    stories = stories or tuple(
        f"story-{index:02d}" for index in range(types.STORY_COUNT)
    )
    controls = controls or tuple(
        types.ControlIdentity(source=source, comparator=comparator, draw=draw)
        for source in sources
        for comparator in types.COMPARATORS
        for draw in range(types.DRAWS_PER_GROUP)
    )
    interventions = interventions or tuple(
        types.InterventionIdentity(source=source, arm="baseline") for source in sources
    ) + tuple(
        types.InterventionIdentity(source=source, arm=arm, draw=draw)
        for source in sources
        for arm in ("alpn", "S", "M")
        for draw in range(types.DRAWS_PER_GROUP)
    )
    return types.AlpnCausalProtocol(
        sources=sources,
        fresh_story_ids=stories,
        controls=controls,
        interventions=interventions,
    )


def _protocol_with_thresholds(
    protocol: types.AlpnCausalProtocol,
    alpha: float = types.ALPHA,
    accessibility_threshold: float = types.ACCESSIBILITY_THRESHOLD,
    causal_threshold: float = types.CAUSAL_THRESHOLD,
) -> types.AlpnCausalProtocol:
    return types.AlpnCausalProtocol(
        sources=protocol.sources,
        fresh_story_ids=protocol.fresh_story_ids,
        controls=protocol.controls,
        interventions=protocol.interventions,
        alpha=alpha,
        accessibility_threshold=accessibility_threshold,
        causal_threshold=causal_threshold,
    )


def test_protocol_requires_the_complete_frozen_identity_matrix() -> None:
    protocol = _protocol()
    assert len(protocol.sources) == 12
    assert len(protocol.fresh_story_ids) == 68
    assert len(protocol.controls) == 120
    assert len(protocol.interventions) == 192
    with pytest.raises(ValueError, match="sources"):
        _ = _protocol(sources=protocol.sources[:-1])
    with pytest.raises(ValueError, match="68 unique"):
        _ = _protocol(stories=(*protocol.fresh_story_ids[:-1], "story-00"))
    with pytest.raises(ValueError, match="controls"):
        _ = _protocol(controls=protocol.controls[:-1])
    with pytest.raises(ValueError, match="interventions"):
        _ = _protocol(interventions=protocol.interventions[:-1])
    with pytest.raises(ValueError, match="alpha"):
        _ = _protocol_with_thresholds(protocol, alpha=0.051)
    with pytest.raises(ValueError, match="accessibility_threshold"):
        _ = _protocol_with_thresholds(protocol, accessibility_threshold=0.101)
    with pytest.raises(ValueError, match="causal_threshold"):
        _ = _protocol_with_thresholds(protocol, causal_threshold=0.011)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_boundary_models_reject_nonfinite_values(value: float) -> None:
    with pytest.raises(ValidationError):
        _ = types.AssayValidity(
            source=_source(7, "real"),
            original_head_gain=0.10,
            trained_readout_probe_gain=0.10,
            readout_replacement_damage=value,
        )


def test_typed_metric_and_identity_schemas_reject_invalid_values() -> None:
    source = _source(7, "real")
    with pytest.raises(ValidationError):
        _ = types.StorySufficientStatistic(story="a", targets=0, nll_sum=1.0)
    with pytest.raises(ValidationError):
        _ = types.GroupIdentity(source=source, group="alpn", draw=5)
    metric = types.ProbeStoryMetric(
        group=types.GroupIdentity(source=source, group="alpn", draw=0),
        statistic=types.StorySufficientStatistic(story="a", targets=2, nll_sum=3.0),
    )
    assert math.isclose(metric.statistic.nll, 1.5)
