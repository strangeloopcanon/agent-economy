from __future__ import annotations

import time
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
    EventType,
    PaymentRule,
    TaskSpec,
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
        discussion_history: Sequence[Any],
    ) -> BidResult:
        _ = ready_tasks, discussion_history
        return BidResult(bids=list(self._scripted.get((round_id, worker.worker_id), ())))


class ScriptedExecutor:
    def __init__(self, *, fail_task_ids: set[str] | None = None) -> None:
        self._fail_task_ids = fail_task_ids or set()

    def execute(
        self,
        *,
        worker: WorkerRuntime,
        task: TaskSpec,
        bid: Bid,
        round_id: int,
        discussion_history: Sequence[Any],
    ) -> ExecutionOutcome:
        _ = worker, bid, round_id, discussion_history
        if task.id in self._fail_task_ids:
            return ExecutionOutcome(status=VerifyStatus.FAIL, notes="scripted fail")
        return ExecutionOutcome(status=VerifyStatus.PASS, notes="scripted pass")


class StatusExecutor:
    def __init__(self, *, status_by_task_id: dict[str, VerifyStatus]) -> None:
        self._status_by_task_id = dict(status_by_task_id)

    def execute(
        self,
        *,
        worker: WorkerRuntime,
        task: TaskSpec,
        bid: Bid,
        round_id: int,
        discussion_history: Sequence[Any],
    ) -> ExecutionOutcome:
        _ = worker, bid, round_id, discussion_history
        status = self._status_by_task_id.get(task.id, VerifyStatus.PASS)
        return ExecutionOutcome(status=status, notes=f"scripted {status.value}")


class RaisingBidder:
    def __init__(self, *, exc: Exception) -> None:
        self._exc = exc

    def get_bids(
        self,
        *,
        worker: WorkerRuntime,
        ready_tasks: Sequence[ReadyTask],
        round_id: int,
        discussion_history: Sequence[Any],
    ) -> BidResult:
        _ = worker, ready_tasks, round_id, discussion_history
        raise self._exc


class RaisingExecutor:
    def __init__(self, *, exc: Exception) -> None:
        self._exc = exc

    def execute(
        self,
        *,
        worker: WorkerRuntime,
        task: TaskSpec,
        bid: Bid,
        round_id: int,
        discussion_history: Sequence[Any],
    ) -> ExecutionOutcome:
        _ = worker, task, bid, round_id, discussion_history
        raise self._exc


def test_engine_runs_multiple_assignments_concurrently(tmp_path) -> None:
    import threading

    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(ledger=ledger, settings=EngineSettings(max_concurrency=2))

    tasks = [
        TaskSpec(id="T1", title="t1", bounty=10, deps=[], acceptance=[CommandSpec(cmd="true")]),
        TaskSpec(id="T2", title="t2", bounty=10, deps=[], acceptance=[CommandSpec(cmd="true")]),
    ]
    workers = [
        WorkerRuntime(worker_id="w1", reputation=1.0),
        WorkerRuntime(worker_id="w2", reputation=1.0),
    ]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)

    bidder = ScriptedBidder(
        {
            (0, "w1"): [Bid(task_id="T1", ask=1, self_assessed_p_success=1.0, eta_minutes=10)],
            (0, "w2"): [Bid(task_id="T2", ask=1, self_assessed_p_success=1.0, eta_minutes=10)],
        }
    )

    class BarrierExecutor:
        def __init__(self) -> None:
            self._barrier = threading.Barrier(2)

        def execute(
            self,
            *,
            worker: WorkerRuntime,
            task: TaskSpec,
            bid: Bid,
            round_id: int,
            discussion_history: Sequence[Any],
        ) -> ExecutionOutcome:
            _ = worker, task, bid, round_id, discussion_history
            try:
                self._barrier.wait(timeout=2.0)
            except threading.BrokenBarrierError as e:
                raise RuntimeError("executor did not run concurrently") from e
            return ExecutionOutcome(status=VerifyStatus.PASS, notes="ok")

    # Step 1 assigns + starts work; step 2 records completions.
    engine.step(bidder=bidder, executor=BarrierExecutor())
    engine.step(bidder=bidder, executor=BarrierExecutor())
    state = replay_ledger(events=list(ledger.iter_events()))
    assert state.tasks["T1"].status == "DONE"
    assert state.tasks["T2"].status == "DONE"


def test_engine_rounds_settlement_and_bounty_bumps(tmp_path) -> None:
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(ledger=ledger, settings=EngineSettings(max_concurrency=1))

    tasks = [
        TaskSpec(
            id="T1",
            title="t1",
            bounty=100,
            deps=[],
            acceptance=[CommandSpec(cmd="true")],
        ),
        TaskSpec(
            id="T2",
            title="t2",
            bounty=100,
            deps=["T1"],
            acceptance=[CommandSpec(cmd="true")],
        ),
    ]
    workers = [
        WorkerRuntime(worker_id="w1", reputation=1.0),
        WorkerRuntime(worker_id="w2", reputation=1.0),
    ]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)

    class BiddingPolicy:
        def get_bids(
            self,
            *,
            worker: WorkerRuntime,
            ready_tasks: Sequence[ReadyTask],
            round_id: int,
            discussion_history: Sequence[Any],
        ) -> BidResult:
            _ = round_id, discussion_history
            ready_ids = {t.spec.id for t in ready_tasks}
            if worker.worker_id == "w1":
                if "T2" in ready_ids:
                    return BidResult(
                        bids=[
                            Bid(
                                task_id="T2",
                                ask=10,
                                self_assessed_p_success=1.0,
                                eta_minutes=10,
                            )
                        ]
                    )
                if "T1" in ready_ids:
                    return BidResult(
                        bids=[
                            Bid(
                                task_id="T1",
                                ask=15,
                                self_assessed_p_success=1.0,
                                eta_minutes=10,
                            )
                        ]
                    )
                return BidResult()

            if worker.worker_id == "w2" and "T1" in ready_ids:
                return BidResult(
                    bids=[Bid(task_id="T1", ask=10, self_assessed_p_success=1.0, eta_minutes=10)]
                )
            return BidResult()

    bidder = BiddingPolicy()
    executor = ScriptedExecutor(fail_task_ids={"T2"})

    # T1 clears to w2 (lower ask), passes, paid ask=10.
    for _ in range(5):
        engine.step(bidder=bidder, executor=executor)
        state = replay_ledger(events=list(ledger.iter_events()))
        if state.tasks["T1"].status == "DONE":
            break
    assert state.tasks["T1"].status == "DONE"
    assert state.tasks["T2"].status == "TODO"
    assert state.workers["w2"].balance == 10.0
    assert state.workers["w2"].reputation > 1.0

    # T2 clears to w1, fails, base penalty 10 + confidence penalty 10 (p_success=1.0).
    for _ in range(5):
        engine.step(bidder=bidder, executor=executor)
        state = replay_ledger(events=list(ledger.iter_events()))
        if state.tasks["T2"].fail_count >= 1:
            break
    assert state.tasks["T2"].status == "TODO"
    assert state.tasks["T2"].fail_count == 1
    assert state.workers["w1"].balance == -20.0
    assert state.workers["w1"].reputation < 1.0

    # T2 fails again; after 2nd failure bounty bumps by 10% (100 -> 110).
    for _ in range(10):
        engine.step(bidder=bidder, executor=executor)
        state = replay_ledger(events=list(ledger.iter_events()))
        if state.tasks["T2"].fail_count >= 2:
            break
    assert state.tasks["T2"].fail_count == 2
    assert state.tasks["T2"].bounty_current == 110
    assert state.workers["w1"].balance == -40.0

    penalty_events = [e for e in ledger.iter_events() if e.type == EventType.PENALTY_APPLIED]
    fail_penalties = [e for e in penalty_events if e.payload.get("reason") == "verification_fail"]
    assert fail_penalties
    assert fail_penalties[0].payload.get("base_penalty") == 10
    assert fail_penalties[0].payload.get("confidence_penalty") == 10


def test_engine_manual_review_does_not_settle(tmp_path) -> None:
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(ledger=ledger, settings=EngineSettings(max_concurrency=1))

    tasks = [
        TaskSpec(
            id="T1",
            title="t1",
            bounty=100,
            deps=[],
            acceptance=[CommandSpec(cmd="true")],
        ),
    ]
    workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)

    bidder = ScriptedBidder(
        {(0, "w1"): [Bid(task_id="T1", ask=10, self_assessed_p_success=1.0, eta_minutes=10)]}
    )
    executor = StatusExecutor(status_by_task_id={"T1": VerifyStatus.MANUAL_REVIEW})

    engine.step(bidder=bidder, executor=executor)
    engine.step(bidder=bidder, executor=executor)
    events = list(ledger.iter_events())
    state = replay_ledger(events=events)

    assert state.tasks["T1"].status == "REVIEW"
    assert state.workers["w1"].balance == 0.0
    assert all(e.type not in {EventType.PAYMENT_MADE, EventType.PENALTY_APPLIED} for e in events)


def test_engine_bidder_exception_does_not_crash_or_stall(tmp_path) -> None:
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(ledger=ledger, settings=EngineSettings(max_concurrency=1))

    tasks = [
        TaskSpec(id="T1", title="t1", bounty=100, deps=[], acceptance=[CommandSpec(cmd="true")])
    ]
    workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)

    bidder = RaisingBidder(exc=RuntimeError("boom"))
    executor = ScriptedExecutor()

    engine.step(bidder=bidder, executor=executor)
    state = replay_ledger(events=list(ledger.iter_events()))
    assert state.tasks["T1"].status == "TODO"
    assert state.tasks["T1"].bounty_current > 100
    assert state.workers["w1"].assigned_task is None


def test_engine_executor_exception_is_recorded_and_task_released(tmp_path) -> None:
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(ledger=ledger, settings=EngineSettings(max_concurrency=1))

    tasks = [
        TaskSpec(id="T1", title="t1", bounty=100, deps=[], acceptance=[CommandSpec(cmd="true")])
    ]
    workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)

    bidder = ScriptedBidder(
        {(0, "w1"): [Bid(task_id="T1", ask=10, self_assessed_p_success=1.0, eta_minutes=10)]}
    )
    executor = RaisingExecutor(exc=RuntimeError("boom"))

    engine.step(bidder=bidder, executor=executor)
    engine.step(bidder=bidder, executor=executor)
    events = list(ledger.iter_events())
    state = replay_ledger(events=events)

    assert state.tasks["T1"].status == "TODO"
    assert state.tasks["T1"].fail_count == 0
    assert state.workers["w1"].balance == 0.0
    assert state.workers["w1"].failures == 0
    assert state.workers["w1"].reputation == 1.0
    assert state.workers["w1"].assigned_task is None

    assert any(e.type == EventType.PATCH_SUBMITTED for e in events)
    assert any(e.type == EventType.TASK_COMPLETED for e in events)


def test_engine_create_run_clears_inflight_state(tmp_path) -> None:
    import threading

    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(ledger=ledger, settings=EngineSettings(max_concurrency=2))

    tasks = [
        TaskSpec(id="T1", title="t1", bounty=10, deps=[], acceptance=[CommandSpec(cmd="true")])
    ]
    workers = [
        WorkerRuntime(worker_id="fast", reputation=1.0),
        WorkerRuntime(worker_id="slow", reputation=1.0),
    ]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)

    allow_slow = threading.Event()

    class MixedBidder:
        def get_bids(
            self,
            *,
            worker: WorkerRuntime,
            ready_tasks: Sequence[ReadyTask],
            round_id: int,
            discussion_history: Sequence[Any],
        ) -> BidResult:
            _ = ready_tasks, round_id, discussion_history
            if worker.worker_id == "slow":
                allow_slow.wait(timeout=2.0)
            return BidResult(
                bids=[
                    Bid(task_id="T1", ask=1, self_assessed_p_success=1.0, eta_minutes=10),
                ]
            )

    engine.step(bidder=MixedBidder(), executor=ScriptedExecutor())
    assert "slow" in engine._inflight_bids

    engine.create_run(run_id="run-2", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)
    allow_slow.set()

    assert engine._inflight_bids == {}
    assert engine._inflight_exec == {}
    assert engine._bid_cache == {}


def test_engine_bidder_timeout_records_error_and_releases_worker(tmp_path) -> None:
    import threading

    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(
        ledger=ledger,
        settings=EngineSettings(
            max_concurrency=1, bid_timeout_seconds=0.05, execution_timeout_seconds=1.0
        ),
    )

    tasks = [
        TaskSpec(id="T1", title="t1", bounty=50, deps=[], acceptance=[CommandSpec(cmd="true")]),
    ]
    workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)

    block = threading.Event()

    class SlowBidder:
        def get_bids(
            self,
            *,
            worker: WorkerRuntime,
            ready_tasks: Sequence[ReadyTask],
            round_id: int,
            discussion_history: Sequence[Any],
        ) -> BidResult:
            _ = worker, ready_tasks, round_id, discussion_history
            block.wait(timeout=1.0)
            return BidResult()

    engine.step(bidder=SlowBidder(), executor=ScriptedExecutor())
    block.set()

    events = list(ledger.iter_events())
    bid_events = [e for e in events if e.type == EventType.BID_SUBMITTED]
    assert bid_events
    assert "bidder_timeout_after_s=0.05" in str(bid_events[-1].payload.get("error") or "")

    state = replay_ledger(events=events)
    assert state.tasks["T1"].status == "TODO"
    assert state.workers["w1"].assigned_task is None


def test_engine_executor_timeout_becomes_infra_without_failure_penalty(tmp_path) -> None:
    import threading

    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(
        ledger=ledger,
        settings=EngineSettings(
            max_concurrency=1, bid_timeout_seconds=1.0, execution_timeout_seconds=0.05
        ),
    )

    tasks = [
        TaskSpec(id="T1", title="t1", bounty=100, deps=[], acceptance=[CommandSpec(cmd="true")]),
    ]
    workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)

    bidder = ScriptedBidder(
        {(0, "w1"): [Bid(task_id="T1", ask=10, self_assessed_p_success=1.0, eta_minutes=10)]}
    )
    block = threading.Event()

    class SlowExecutor:
        def execute(
            self,
            *,
            worker: WorkerRuntime,
            task: TaskSpec,
            bid: Bid,
            round_id: int,
            discussion_history: Sequence[Any],
        ) -> ExecutionOutcome:
            _ = worker, task, bid, round_id, discussion_history
            block.wait(timeout=1.0)
            return ExecutionOutcome(status=VerifyStatus.PASS, notes="late pass")

    engine.step(bidder=bidder, executor=SlowExecutor())
    time.sleep(0.08)
    engine.step(bidder=bidder, executor=SlowExecutor())
    block.set()

    events = list(ledger.iter_events())
    completed = [e for e in events if e.type == EventType.TASK_COMPLETED]
    assert completed
    assert str(completed[-1].payload.get("verify_status") or "") == VerifyStatus.INFRA.value

    state = replay_ledger(events=events)
    assert state.tasks["T1"].status == "TODO"
    assert state.tasks["T1"].fail_count == 0
    assert state.workers["w1"].failures == 0
    assert state.workers["w1"].reputation == 1.0


def test_engine_emits_scoring_snapshots_for_market_and_assignment(tmp_path) -> None:
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(ledger=ledger, settings=EngineSettings(max_concurrency=1))

    tasks = [
        TaskSpec(id="T1", title="t1", bounty=100, deps=[], acceptance=[CommandSpec(cmd="true")])
    ]
    workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)

    bidder = ScriptedBidder(
        {(0, "w1"): [Bid(task_id="T1", ask=20, self_assessed_p_success=0.9, eta_minutes=10)]}
    )
    executor = ScriptedExecutor()

    engine.step(bidder=bidder, executor=executor)
    events = list(ledger.iter_events())
    market = [e for e in events if e.type == EventType.MARKET_CLEARED]
    assigned = [e for e in events if e.type == EventType.TASK_ASSIGNED]
    assert market and assigned

    market_snapshot = ((market[-1].payload.get("assignments") or [])[0] or {}).get("score_snapshot")
    assigned_snapshot = assigned[-1].payload.get("score_snapshot")
    assert isinstance(market_snapshot, dict)
    assert isinstance(assigned_snapshot, dict)
    assert market_snapshot.get("components", {}).get("score") is not None
    assert assigned_snapshot.get("components", {}).get("score") is not None
