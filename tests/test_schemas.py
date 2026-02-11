from __future__ import annotations

import pytest

from agent_economy.schemas import Bid, CommandSpec, TaskSpec, VerifyMode


def test_task_spec_acceptance_required_in_commands_mode() -> None:
    with pytest.raises(ValueError, match="acceptance commands must be non-empty"):
        TaskSpec(
            id="T1",
            title="t1",
            bounty=1,
            verify_mode=VerifyMode.COMMANDS,
            acceptance=[],
        )


def test_task_spec_allows_empty_acceptance_in_manual_mode() -> None:
    t = TaskSpec(
        id="T1",
        title="t1",
        bounty=1,
        verify_mode=VerifyMode.MANUAL,
        acceptance=[],
        hidden_acceptance=[CommandSpec(cmd="true")],
    )
    assert t.verify_mode == VerifyMode.MANUAL


def test_task_spec_allows_empty_acceptance_in_judges_mode() -> None:
    t = TaskSpec(
        id="T1",
        title="t1",
        bounty=1,
        verify_mode=VerifyMode.JUDGES,
        acceptance=[],
        hidden_acceptance=[CommandSpec(cmd="true")],
    )
    assert t.verify_mode == VerifyMode.JUDGES


def test_bid_coerces_float_ask_to_int() -> None:
    b = Bid(task_id="T1", ask=0.5, self_assessed_p_success=0.5, eta_minutes=10)
    assert b.ask == 1
