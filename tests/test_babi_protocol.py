"""Frozen schedules, typed identities and validation-only deterministic selection."""

import pytest
from pydantic import ValidationError

from flyrl.babi_data import BabiError
from flyrl.babi_pilot_schema import Validation, select_update
from flyrl.babi_protocol import BabiSchedule, code_identity


def test_schedule_when_frozen_boundaries() -> None:
    # Given: the protocol's exact one-based schedule.
    schedule = BabiSchedule()
    # When/Then: boundaries and midpoint are machine-defined values.
    assert schedule.rate(1) == 0.003 / 200
    assert schedule.rate(200) == 0.003
    assert schedule.rate(6100) == pytest.approx(0.00165)
    assert schedule.rate(12000) == 0.0003
    assert schedule.rate(201) < schedule.rate(200)
    for update in (0, 12001):
        with pytest.raises(BabiError):
            _ = schedule.rate(update)


def test_schedule_when_mutated_is_rejected() -> None:
    # Given/When/Then: frozen literal settings cannot silently change the benchmark.
    with pytest.raises(ValidationError):
        _ = BabiSchedule.model_validate_json('{"target_updates": 4}')


def test_selection_when_exact_nll_and_earliest_tie() -> None:
    # Given: initialization wins raw score but is ineligible.
    history = tuple(
        Validation(update=u, correct=c, questions=1000, nll=n)
        for u, c, n in (
            (0, 1000, 0.1),
            (500, 900, 2.0),
            (2000, 901, 3.0),
            (4000, 901, 2.0),
            (8000, 901, 2.0),
        )
    )
    # When/Then: exact count precedes NLL, which precedes earliest update.
    assert select_update(history) == 4000
    assert select_update(tuple(reversed(history))) == 4000
    with pytest.raises(BabiError):
        _ = select_update(history[:1])


def test_code_identity_when_unrelated_modules_exist_are_excluded() -> None:
    # Given/When: code sealing uses only explicitly reviewed transitive dependencies.
    paths = tuple(item.path.as_posix() for item in code_identity())
    # Then: another experiment cannot invalidate Task 1 recovery.
    assert "scripts/run_quality_pilot.py" not in paths
    assert "flyrl/cuda_edge_grad.py" in paths
    assert "scripts/run_babi_task1.py" in paths
