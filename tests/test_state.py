from __future__ import annotations

import pytest

from institution_service.ledger import HashChainedLedger
from institution_service.schemas import EventType, VerifyStatus
from institution_service.state import replay_ledger


def test_replay_updates_balance_and_reputation(tmp_path) -> None:
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    run_id = "run-1"

    ledger.append(EventType.RUN_CREATED, run_id=run_id, round_id=0, payload={"payment_rule": "ask"})
    ledger.append(
        EventType.WORKER_REGISTERED,
        run_id=run_id,
        round_id=0,
        payload={"worker_id": "w1", "balance": 0, "reputation": 1.0},
    )
    ledger.append(
        EventType.TASK_CREATED,
        run_id=run_id,
        round_id=0,
        payload={"task_id": "T1", "bounty": 50},
    )
    ledger.append(
        EventType.TASK_ASSIGNED,
        run_id=run_id,
        round_id=0,
        payload={"task_id": "T1", "worker_id": "w1"},
    )
    ledger.append(
        EventType.TASK_COMPLETED,
        run_id=run_id,
        round_id=0,
        payload={"task_id": "T1", "worker_id": "w1", "success": True},
    )
    ledger.append(
        EventType.PAYMENT_MADE,
        run_id=run_id,
        round_id=0,
        payload={"task_id": "T1", "worker_id": "w1", "amount": 40},
    )

    state = replay_ledger(events=list(ledger.iter_events()))
    assert state.tasks["T1"].status == "DONE"
    assert state.workers["w1"].completions == 1
    assert state.workers["w1"].balance == 40.0
    assert state.workers["w1"].reputation > 1.0


def test_replay_manual_review_sets_task_to_review(tmp_path) -> None:
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    run_id = "run-1"

    ledger.append(EventType.RUN_CREATED, run_id=run_id, round_id=0, payload={"payment_rule": "ask"})
    ledger.append(
        EventType.WORKER_REGISTERED,
        run_id=run_id,
        round_id=0,
        payload={"worker_id": "w1", "balance": 0, "reputation": 1.0},
    )
    ledger.append(
        EventType.TASK_CREATED,
        run_id=run_id,
        round_id=0,
        payload={"task_id": "T1", "bounty": 50},
    )
    ledger.append(
        EventType.TASK_ASSIGNED,
        run_id=run_id,
        round_id=0,
        payload={"task_id": "T1", "worker_id": "w1"},
    )
    ledger.append(
        EventType.TASK_COMPLETED,
        run_id=run_id,
        round_id=0,
        payload={
            "task_id": "T1",
            "worker_id": "w1",
            "success": False,
            "verify_status": VerifyStatus.MANUAL_REVIEW.value,
        },
    )

    state = replay_ledger(events=list(ledger.iter_events()))
    assert state.tasks["T1"].status == "REVIEW"
    assert state.tasks["T1"].assigned_worker == "w1"
    assert state.tasks["T1"].fail_count == 0
    assert state.workers["w1"].assigned_task is None
    assert state.workers["w1"].failures == 0
    assert state.workers["w1"].reputation == pytest.approx(1.0)


@pytest.mark.parametrize(
    "status",
    [
        VerifyStatus.INFRA,
        VerifyStatus.TIMEOUT,
        VerifyStatus.FLAKE_SUSPECTED,
    ],
)
def test_replay_no_fault_status_does_not_count_as_failure(tmp_path, status: VerifyStatus) -> None:
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    run_id = "run-1"

    ledger.append(EventType.RUN_CREATED, run_id=run_id, round_id=0, payload={"payment_rule": "ask"})
    ledger.append(
        EventType.WORKER_REGISTERED,
        run_id=run_id,
        round_id=0,
        payload={"worker_id": "w1", "balance": 0, "reputation": 1.0},
    )
    ledger.append(
        EventType.TASK_CREATED,
        run_id=run_id,
        round_id=0,
        payload={"task_id": "T1", "bounty": 50},
    )
    ledger.append(
        EventType.TASK_ASSIGNED,
        run_id=run_id,
        round_id=0,
        payload={"task_id": "T1", "worker_id": "w1"},
    )
    ledger.append(
        EventType.TASK_COMPLETED,
        run_id=run_id,
        round_id=0,
        payload={
            "task_id": "T1",
            "worker_id": "w1",
            "success": False,
            "verify_status": status.value,
        },
    )

    state = replay_ledger(events=list(ledger.iter_events()))
    assert state.tasks["T1"].status == "TODO"
    assert state.tasks["T1"].assigned_worker is None
    assert state.tasks["T1"].fail_count == 0
    assert state.workers["w1"].assigned_task is None
    assert state.workers["w1"].failures == 0
    assert state.workers["w1"].reputation == pytest.approx(1.0)
