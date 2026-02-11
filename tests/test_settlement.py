from __future__ import annotations

from collections.abc import Sequence

from agent_economy.engine import (
    BidResult,
    ClearinghouseEngine,
    EngineSettings,
    ExecutionOutcome,
    ReadyTask,
)
from agent_economy.finalize import release_judges_holdbacks
from agent_economy.ledger import HashChainedLedger
from agent_economy.schemas import (
    Bid,
    CommandSpec,
    DiscussionMessage,
    PaymentRule,
    TaskSpec,
    VerifyMode,
    VerifyStatus,
    WorkerRuntime,
)
from agent_economy.state import replay_ledger


class ScriptedBidder:
    def __init__(self, scripted: dict[tuple[int, str], Sequence[Bid]]) -> None:
        self._scripted = scripted

    def get_bids(
        self,
        *,
        worker: WorkerRuntime,
        ready_tasks: Sequence[ReadyTask],
        round_id: int,
        discussion_history: Sequence[DiscussionMessage] = (),
    ) -> BidResult:
        _ = ready_tasks, discussion_history
        return BidResult(bids=list(self._scripted.get((round_id, worker.worker_id), ())))


class UsageExecutor:
    def __init__(self, *, status: VerifyStatus) -> None:
        self._status = status

    def execute(
        self,
        *,
        worker: WorkerRuntime,
        task: TaskSpec,
        bid: Bid,
        round_id: int,
        discussion_history: Sequence[DiscussionMessage] = (),
    ) -> ExecutionOutcome:
        _ = worker, task, bid, round_id, discussion_history
        return ExecutionOutcome(
            status=self._status,
            notes="scripted",
            llm_usage={"calls": 1, "input_tokens": 1000, "output_tokens": 500},
        )


class FixedCostEstimator:
    def __init__(self, *, usage_cost: float) -> None:
        self._usage_cost = float(usage_cost)

    def expected_cost(
        self,
        *,
        worker: WorkerRuntime,
        task: TaskSpec,
        bid: Bid,
        round_id: int,
    ) -> float:
        _ = worker, task, bid, round_id
        return 0.0

    def actual_cost(
        self,
        *,
        worker: WorkerRuntime,
        llm_usage: dict[str, int] | None,
    ) -> float:
        _ = worker
        return self._usage_cost if llm_usage else 0.0


def test_engine_applies_usage_cost_ding_as_penalty(tmp_path) -> None:
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(ledger=ledger, settings=EngineSettings(max_concurrency=1))

    tasks = [
        TaskSpec(id="T1", title="t1", bounty=100, acceptance=[CommandSpec(cmd="true")]),
    ]
    workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)

    bidder = ScriptedBidder(
        {(0, "w1"): [Bid(task_id="T1", ask=10, self_assessed_p_success=1.0, eta_minutes=10)]}
    )
    executor = UsageExecutor(status=VerifyStatus.PASS)
    cost_estimator = FixedCostEstimator(usage_cost=2.5)

    engine.step(bidder=bidder, executor=executor, cost_estimator=cost_estimator)
    engine.step(bidder=bidder, executor=executor, cost_estimator=cost_estimator)
    state = replay_ledger(events=list(ledger.iter_events()))

    # Paid ask (10) minus usage cost (2.5).
    assert state.workers["w1"].balance == 7.5


def test_release_judges_holdbacks_after_run_complete(tmp_path) -> None:
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(ledger=ledger, settings=EngineSettings(max_concurrency=1))

    tasks = [
        TaskSpec(
            id="T1",
            title="t1",
            bounty=100,
            verify_mode=VerifyMode.JUDGES,
            acceptance=[],
        ),
    ]
    workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)

    bidder = ScriptedBidder(
        {(0, "w1"): [Bid(task_id="T1", ask=10, self_assessed_p_success=1.0, eta_minutes=10)]}
    )
    executor = UsageExecutor(status=VerifyStatus.PASS)
    cost_estimator = FixedCostEstimator(usage_cost=0.0)

    engine.step(bidder=bidder, executor=executor, cost_estimator=cost_estimator)
    engine.step(bidder=bidder, executor=executor, cost_estimator=cost_estimator)
    state = replay_ledger(events=list(ledger.iter_events()))
    assert state.tasks["T1"].status == "DONE"

    # Hold back 25% for judges mode; release on completion.
    assert state.workers["w1"].balance == 7.5
    assert release_judges_holdbacks(ledger=ledger) == 1

    state2 = replay_ledger(events=list(ledger.iter_events()))
    assert state2.workers["w1"].balance == 10.0
    assert release_judges_holdbacks(ledger=ledger) == 0
