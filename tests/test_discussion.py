from collections.abc import Sequence
from typing import Any

from agent_economy.engine import (
    BidResult,
    ClearinghouseEngine,
    EngineSettings,
    ExecutionOutcome,
    ReadyTask,
)
from agent_economy.ledger import HashChainedLedger
from agent_economy.schemas import (
    Bid,
    CommandSpec,
    TaskSpec,
    VerifyStatus,
    WorkerRuntime,
)
from agent_economy.state import replay_ledger


class DiscussionBidder:
    def get_bids(
        self,
        *,
        worker: WorkerRuntime,
        ready_tasks: Sequence[ReadyTask],
        round_id: int,
        discussion_history: Sequence[Any],
    ) -> BidResult:
        return BidResult(bids=[], discussion="This is a public message")


class NoOpExecutor:
    def execute(
        self,
        *,
        worker: WorkerRuntime,
        task: TaskSpec,
        bid: Bid,
        round_id: int,
        discussion_history: Sequence[Any],
    ) -> ExecutionOutcome:
        return ExecutionOutcome(status=VerifyStatus.PASS)


def test_discussion_flow(tmp_path):
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(ledger=ledger, settings=EngineSettings(max_concurrency=1))

    tasks = [TaskSpec(id="T1", title="t1", bounty=10, acceptance=[CommandSpec(cmd="true")])]
    workers = [WorkerRuntime(worker_id="w1")]

    engine.create_run(run_id="run-1", workers=workers, tasks=tasks)

    # Run step - bidder submits discussion
    engine.step(bidder=DiscussionBidder(), executor=NoOpExecutor())

    # Check state
    state = replay_ledger(events=list(ledger.iter_events()))
    assert len(state.discussion_history) == 1
    msg = state.discussion_history[0]
    assert msg.sender == "w1"
    assert msg.message == "This is a public message"

    # Check that next step receives the history
    class VerifyingBidder:
        def get_bids(
            self,
            *,
            worker: WorkerRuntime,
            ready_tasks: Sequence[ReadyTask],
            round_id: int,
            discussion_history: Sequence[Any],
        ) -> BidResult:
            assert len(discussion_history) == 1
            assert discussion_history[0].message == "This is a public message"
            return BidResult()

    engine.step(bidder=VerifyingBidder(), executor=NoOpExecutor())
