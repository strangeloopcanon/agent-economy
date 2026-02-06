from __future__ import annotations

from collections.abc import Sequence

from institution_service.engine import (
    BidResult,
    ClearinghouseEngine,
    EngineSettings,
    ExecutionOutcome,
    ReadyTask,
)
from institution_service.learning_trace import extract_attempt_transitions
from institution_service.ledger import HashChainedLedger
from institution_service.schemas import (
    Bid,
    CommandSpec,
    DiscussionMessage,
    PaymentRule,
    TaskSpec,
    VerifyStatus,
    WorkerRuntime,
)


class ScriptedBidder:
    def __init__(self, bid: Bid) -> None:
        self._bid = bid

    def get_bids(
        self,
        *,
        worker: WorkerRuntime,
        ready_tasks: Sequence[ReadyTask],
        round_id: int,
        discussion_history: Sequence[DiscussionMessage] = (),
    ) -> BidResult:
        _ = worker, ready_tasks, round_id, discussion_history
        return BidResult(bids=[self._bid])


class ScriptedExecutor:
    def __init__(self, *, status: VerifyStatus, llm_usage: dict[str, int] | None = None) -> None:
        self._status = status
        self._llm_usage = llm_usage

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
        return ExecutionOutcome(status=self._status, notes="scripted", llm_usage=self._llm_usage)


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


def _single_task_setup(tmp_path) -> tuple[HashChainedLedger, ClearinghouseEngine]:
    ledger = HashChainedLedger(tmp_path / "ledger.jsonl")
    engine = ClearinghouseEngine(ledger=ledger, settings=EngineSettings(max_concurrency=1))
    tasks = [TaskSpec(id="T1", title="t1", bounty=100, acceptance=[CommandSpec(cmd="true")])]
    workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]
    engine.create_run(run_id="run-1", payment_rule=PaymentRule.ASK, workers=workers, tasks=tasks)
    return ledger, engine


def test_extract_attempt_transitions_reward_for_pass_with_usage_cost(tmp_path) -> None:
    ledger, engine = _single_task_setup(tmp_path)
    bidder = ScriptedBidder(Bid(task_id="T1", ask=10, self_assessed_p_success=0.9, eta_minutes=10))
    executor = ScriptedExecutor(
        status=VerifyStatus.PASS,
        llm_usage={"calls": 1, "input_tokens": 1000, "output_tokens": 500},
    )
    estimator = FixedCostEstimator(usage_cost=2.5)

    engine.step(bidder=bidder, executor=executor, cost_estimator=estimator)
    engine.step(bidder=bidder, executor=executor, cost_estimator=estimator)

    transitions = extract_attempt_transitions(events=list(ledger.iter_events()))
    assert len(transitions) == 1
    row = transitions[0]
    assert row["task_id"] == "T1"
    assert row["worker_id"] == "w1"
    assert row["outcome"]["verify_status"] == VerifyStatus.PASS.value
    assert row["payment"] == 10.0
    assert row["reward"] == 7.5
    reasons = {str(p.get("reason") or "") for p in row["penalties"]}
    assert "usage_cost" in reasons
    assert isinstance(row["award"]["score_snapshot"], dict)


def test_extract_attempt_transitions_captures_confidence_fail_penalty(tmp_path) -> None:
    ledger, engine = _single_task_setup(tmp_path)
    bidder = ScriptedBidder(Bid(task_id="T1", ask=10, self_assessed_p_success=1.0, eta_minutes=10))
    executor = ScriptedExecutor(status=VerifyStatus.FAIL)

    engine.step(bidder=bidder, executor=executor)
    engine.step(bidder=bidder, executor=executor)

    transitions = extract_attempt_transitions(events=list(ledger.iter_events()))
    assert len(transitions) == 1
    row = transitions[0]
    assert row["outcome"]["verify_status"] == VerifyStatus.FAIL.value
    assert row["payment"] == 0.0
    assert row["reward"] == -20.0
    fail_penalty = next(p for p in row["penalties"] if p.get("reason") == "verification_fail")
    assert fail_penalty["base_penalty"] == 10.0
    assert fail_penalty["confidence_penalty"] == 10.0
