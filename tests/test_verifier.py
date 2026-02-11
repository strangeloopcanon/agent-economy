"""Tests for the Verifier protocol injection."""

from __future__ import annotations

from tests.helpers import AlwaysFailExecutor, AlwaysPassExecutor, FixedBidder
from agent_economy.engine import (
    ClearinghouseEngine,
    EngineSettings,
)
from agent_economy.ledger import InMemoryLedger
from agent_economy.schemas import (
    CommandSpec,
    EventType,
    PaymentRule,
    TaskSpec,
    VerifyStatus,
    WorkerRuntime,
)
from agent_economy.state import replay_ledger


class _OverrideToFailVerifier:
    """Verifier that always overrides to FAIL."""

    def verify(self, *, task, worker, outcome):
        return VerifyStatus.FAIL


class _OverrideToPassVerifier:
    """Verifier that always overrides to PASS."""

    def verify(self, *, task, worker, outcome):
        return VerifyStatus.PASS


class _ConditionalVerifier:
    """Verifier that fails a specific task, passes everything else."""

    def __init__(self, *, fail_task_ids: set[str]) -> None:
        self._fail_task_ids = fail_task_ids

    def verify(self, *, task, worker, outcome):
        if task.id in self._fail_task_ids:
            return VerifyStatus.FAIL
        return outcome.status


class _ExplodingVerifier:
    """Verifier that raises an exception."""

    def verify(self, *, task, worker, outcome):
        raise RuntimeError("verifier crashed")


class _InvalidStatusVerifier:
    """Verifier that returns an invalid status value."""

    def verify(self, *, task, worker, outcome):
        return "not-a-status"


def _run_scenario(
    *,
    executor,
    verifier=None,
    tasks=None,
    workers=None,
    max_steps=20,
):
    if tasks is None:
        tasks = [
            TaskSpec(
                id="T1",
                title="Task",
                bounty=20,
                deps=[],
                acceptance=[CommandSpec(cmd="true")],
            )
        ]
    if workers is None:
        workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]

    ledger = InMemoryLedger()
    engine = ClearinghouseEngine(
        ledger=ledger,
        settings=EngineSettings(max_concurrency=1, deterministic=True),
        verifier=verifier,
    )
    engine.create_run(
        run_id="verifier-test",
        payment_rule=PaymentRule.ASK,
        workers=workers,
        tasks=tasks,
    )

    bidder = FixedBidder()
    for _ in range(max_steps):
        engine.step(bidder=bidder, executor=executor)
        state = replay_ledger(events=list(ledger.iter_events()))
        if all(t.status == "DONE" for t in state.tasks.values()):
            break

    return ledger, state


class TestVerifierProtocol:
    def test_no_verifier_passes_through(self) -> None:
        """Without a verifier, executor's verdict is used as-is."""
        ledger, state = _run_scenario(executor=AlwaysPassExecutor())
        assert state.tasks["T1"].status == "DONE"
        assert state.workers["w1"].balance > 0

    def test_verifier_overrides_pass_to_fail(self) -> None:
        """Verifier can override executor's PASS to FAIL."""
        ledger, state = _run_scenario(
            executor=AlwaysPassExecutor(),
            verifier=_OverrideToFailVerifier(),
        )
        # Task should NOT be done -- verifier rejected it.
        assert state.tasks["T1"].status != "DONE"
        assert state.tasks["T1"].fail_count > 0
        # Worker should have negative balance (penalties).
        assert state.workers["w1"].balance < 0

    def test_verifier_overrides_fail_to_pass(self) -> None:
        """Verifier can override executor's FAIL to PASS."""
        ledger, state = _run_scenario(
            executor=AlwaysFailExecutor(),
            verifier=_OverrideToPassVerifier(),
        )
        assert state.tasks["T1"].status == "DONE"
        assert state.workers["w1"].balance > 0

    def test_conditional_verifier(self) -> None:
        """Verifier can selectively override specific tasks."""
        tasks = [
            TaskSpec(
                id="T1", title="Good", bounty=20, deps=[], acceptance=[CommandSpec(cmd="true")]
            ),
            TaskSpec(
                id="T2", title="Bad", bounty=20, deps=[], acceptance=[CommandSpec(cmd="true")]
            ),
        ]
        workers = [
            WorkerRuntime(worker_id="w1", reputation=1.0),
            WorkerRuntime(worker_id="w2", reputation=1.0),
        ]

        ledger, state = _run_scenario(
            executor=AlwaysPassExecutor(),
            verifier=_ConditionalVerifier(fail_task_ids={"T2"}),
            tasks=tasks,
            workers=workers,
            max_steps=30,
        )
        # T1 should pass (verifier passes through executor's PASS).
        assert state.tasks["T1"].status == "DONE"
        # T2 should have failed at least once (verifier overrides to FAIL).
        assert state.tasks["T2"].fail_count > 0

    def test_verifier_exception_becomes_infra(self) -> None:
        """If verifier raises, outcome degrades to INFRA (no fault)."""
        ledger, state = _run_scenario(
            executor=AlwaysPassExecutor(),
            verifier=_ExplodingVerifier(),
            max_steps=10,
        )
        # INFRA status doesn't count as pass or fail: no fail_count
        # increment, no reputation loss, task never reaches DONE.
        assert state.tasks["T1"].fail_count == 0
        assert state.tasks["T1"].status != "DONE"
        # Worker reputation unchanged from initial 1.0 (INFRA is no-fault).
        assert state.workers["w1"].reputation == 1.0

    def test_verifier_events_in_ledger(self) -> None:
        """Settlement events reflect the verifier's verdict, not the executor's."""
        ledger, state = _run_scenario(
            executor=AlwaysPassExecutor(),
            verifier=_OverrideToFailVerifier(),
            max_steps=5,
        )
        events = list(ledger.iter_events())
        # Should have VERIFICATION_FAILED events, not VERIFICATION_PASSED.
        verify_events = [e for e in events if e.type == EventType.VERIFICATION_FAILED]
        assert len(verify_events) > 0
        pass_events = [e for e in events if e.type == EventType.VERIFICATION_PASSED]
        assert len(pass_events) == 0

    def test_invalid_verifier_status_becomes_infra(self) -> None:
        """Invalid verifier return values are treated as infra failures."""
        ledger, state = _run_scenario(
            executor=AlwaysPassExecutor(),
            verifier=_InvalidStatusVerifier(),
            max_steps=10,
        )

        assert state.tasks["T1"].status != "DONE"
        assert state.tasks["T1"].fail_count == 0

        verify_events = [e for e in ledger.iter_events() if e.type == EventType.VERIFICATION_FAILED]
        assert len(verify_events) > 0
        assert verify_events[-1].payload["status"] == VerifyStatus.INFRA.value
