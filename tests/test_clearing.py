from __future__ import annotations

from institution_service.clearing import BidSubmission, choose_assignments
from institution_service.schemas import Bid, TaskRuntime, WorkerRuntime


def test_choose_assignments_deterministic_tiebreak() -> None:
    tasks = [
        TaskRuntime(task_id="T1", bounty_current=100, bounty_original=100),
        TaskRuntime(task_id="T2", bounty_current=100, bounty_original=100),
    ]
    workers = [
        WorkerRuntime(worker_id="w1", reputation=1.0),
        WorkerRuntime(worker_id="w2", reputation=1.0),
    ]
    bid = Bid(task_id="T1", ask=10, self_assessed_p_success=1.0, eta_minutes=10)
    bid2 = Bid(task_id="T2", ask=10, self_assessed_p_success=1.0, eta_minutes=10)

    bids_by_task = {
        "T1": [
            BidSubmission(worker_id="w1", bid=bid),
            BidSubmission(worker_id="w2", bid=bid),
        ],
        "T2": [
            BidSubmission(worker_id="w1", bid=bid2),
            BidSubmission(worker_id="w2", bid=bid2),
        ],
    }

    assignments = choose_assignments(
        ready_tasks=tasks, available_workers=workers, bids_by_task=bids_by_task
    )
    assert [(a.task_id, a.worker_id) for a in assignments] == [("T1", "w2"), ("T2", "w1")]


def test_expected_cost_can_change_winner() -> None:
    tasks = [TaskRuntime(task_id="T1", bounty_current=100, bounty_original=100)]
    workers = [
        WorkerRuntime(worker_id="cheap", reputation=1.0),
        WorkerRuntime(worker_id="expensive", reputation=1.0),
    ]
    bid_cheap = Bid(task_id="T1", ask=10, self_assessed_p_success=1.0, eta_minutes=10)
    bid_exp = Bid(task_id="T1", ask=9, self_assessed_p_success=1.0, eta_minutes=10)

    bids_by_task = {
        "T1": [
            BidSubmission(worker_id="cheap", bid=bid_cheap, expected_cost=0.0),
            BidSubmission(worker_id="expensive", bid=bid_exp, expected_cost=5.0),
        ]
    }

    assignments = choose_assignments(
        ready_tasks=tasks, available_workers=workers, bids_by_task=bids_by_task
    )
    assert len(assignments) == 1
    assert assignments[0].worker_id == "cheap"
