from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from institution_service.engine import BidResult, ExecutionOutcome, ReadyTask
from institution_service.schemas import (
    Bid,
    DiscussionMessage,
    TaskSpec,
    VerifyStatus,
    WorkerRuntime,
)


class FixedBidder:
    """Bidder that bids on up to 2 ready tasks with fixed parameters."""

    def get_bids(
        self,
        *,
        worker: WorkerRuntime,
        ready_tasks: Sequence[ReadyTask],
        round_id: int,
        discussion_history: Sequence[DiscussionMessage] = (),
        **kw: Any,
    ) -> BidResult:
        return BidResult(
            bids=[
                Bid(task_id=rt.spec.id, ask=5, self_assessed_p_success=0.9, eta_minutes=10)
                for rt in ready_tasks
            ][:2]
        )


class AlwaysPassExecutor:
    """Executor that always returns PASS."""

    def execute(
        self,
        *,
        worker: WorkerRuntime,
        task: TaskSpec,
        bid: Bid,
        round_id: int,
        discussion_history: Sequence[DiscussionMessage] = (),
        **kw: Any,
    ) -> ExecutionOutcome:
        return ExecutionOutcome(status=VerifyStatus.PASS, notes="ok")


class AlwaysFailExecutor:
    """Executor that always returns FAIL."""

    def execute(
        self,
        *,
        worker: WorkerRuntime,
        task: TaskSpec,
        bid: Bid,
        round_id: int,
        discussion_history: Sequence[DiscussionMessage] = (),
        **kw: Any,
    ) -> ExecutionOutcome:
        return ExecutionOutcome(status=VerifyStatus.FAIL, notes="fail")
