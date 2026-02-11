"""Tests for InMemoryLedger and deterministic engine mode."""

from __future__ import annotations

import pytest

from tests.helpers import AlwaysPassExecutor, FixedBidder
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
    WorkerRuntime,
)
from agent_economy.state import replay_ledger


class TestInMemoryLedger:
    def test_append_and_iter(self) -> None:
        ledger = InMemoryLedger()
        ledger.append(
            EventType.RUN_CREATED,
            run_id="r1",
            round_id=0,
            payload={"payment_rule": "ask"},
        )
        ledger.append(
            EventType.TASK_CREATED,
            run_id="r1",
            round_id=0,
            payload={"task_id": "T1", "bounty": 10},
        )

        events = list(ledger.iter_events())
        assert len(events) == 2
        assert events[0].type == EventType.RUN_CREATED
        assert events[1].type == EventType.TASK_CREATED

    def test_verify_chain_passes(self) -> None:
        ledger = InMemoryLedger()
        ledger.append(EventType.RUN_CREATED, run_id="r1", round_id=0)
        ledger.append(
            EventType.TASK_CREATED, run_id="r1", round_id=0, payload={"task_id": "T1", "bounty": 5}
        )
        ledger.verify_chain()

    def test_verify_chain_detects_tampering(self) -> None:
        ledger = InMemoryLedger()
        ledger.append(EventType.RUN_CREATED, run_id="r1", round_id=0)
        ledger.append(
            EventType.TASK_CREATED, run_id="r1", round_id=0, payload={"task_id": "T1", "bounty": 5}
        )

        # Tamper with payload without recomputing hash.
        ledger._events[1].payload["bounty"] = 999

        with pytest.raises(ValueError, match="hash mismatch"):
            ledger.verify_chain()

    def test_iter_events_returns_copies(self) -> None:
        ledger = InMemoryLedger()
        ledger.append(EventType.RUN_CREATED, run_id="r1", round_id=0)
        ledger.append(
            EventType.TASK_CREATED, run_id="r1", round_id=0, payload={"task_id": "T1", "bounty": 5}
        )

        events = list(ledger.iter_events())
        events[1].payload["bounty"] = 999

        # Mutating iterated events should not mutate internal ledger state.
        ledger.verify_chain()
        original = list(ledger.iter_events())[1]
        assert original.payload["bounty"] == 5

    def test_reset_clears(self) -> None:
        ledger = InMemoryLedger()
        ledger.append(EventType.RUN_CREATED, run_id="r1", round_id=0)
        assert len(list(ledger.iter_events())) == 1

        ledger.reset()
        assert len(list(ledger.iter_events())) == 0

    def test_prev_hash_chain(self) -> None:
        ledger = InMemoryLedger()
        e1 = ledger.append(EventType.RUN_CREATED, run_id="r1", round_id=0)
        e2 = ledger.append(
            EventType.TASK_CREATED, run_id="r1", round_id=0, payload={"task_id": "T1", "bounty": 1}
        )

        assert e1.prev_hash is None
        assert e2.prev_hash == e1.hash


class TestDeterministicEngine:
    def test_full_run_deterministic(self) -> None:
        """Run a complete scenario using InMemoryLedger + deterministic mode."""
        ledger = InMemoryLedger()
        engine = ClearinghouseEngine(
            ledger=ledger,
            settings=EngineSettings(max_concurrency=1, deterministic=True),
        )

        tasks = [
            TaskSpec(
                id="T1", title="First", bounty=20, deps=[], acceptance=[CommandSpec(cmd="true")]
            ),
            TaskSpec(
                id="T2",
                title="Second",
                bounty=30,
                deps=["T1"],
                acceptance=[CommandSpec(cmd="true")],
            ),
        ]
        workers = [
            WorkerRuntime(worker_id="w1", reputation=1.0),
        ]

        engine.create_run(
            run_id="det-test",
            payment_rule=PaymentRule.ASK,
            workers=workers,
            tasks=tasks,
        )

        bidder = FixedBidder()
        executor = AlwaysPassExecutor()

        for _ in range(20):
            engine.step(bidder=bidder, executor=executor)
            state = replay_ledger(events=list(ledger.iter_events()))
            if all(t.status == "DONE" for t in state.tasks.values()):
                break

        state = replay_ledger(events=list(ledger.iter_events()))
        assert state.tasks["T1"].status == "DONE"
        assert state.tasks["T2"].status == "DONE"

        ledger.verify_chain()

    def test_deterministic_produces_same_events_twice(self) -> None:
        """Two identical deterministic runs produce identical event sequences."""

        def _run_once() -> list[str]:
            ledger = InMemoryLedger()
            engine = ClearinghouseEngine(
                ledger=ledger,
                settings=EngineSettings(max_concurrency=1, deterministic=True),
            )
            tasks = [
                TaskSpec(
                    id="T1", title="Task", bounty=10, deps=[], acceptance=[CommandSpec(cmd="true")]
                ),
            ]
            workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]
            engine.create_run(
                run_id="det",
                payment_rule=PaymentRule.ASK,
                workers=workers,
                tasks=tasks,
            )
            bidder = FixedBidder()
            executor = AlwaysPassExecutor()
            for _ in range(10):
                engine.step(bidder=bidder, executor=executor)
                state = replay_ledger(events=list(ledger.iter_events()))
                if all(t.status == "DONE" for t in state.tasks.values()):
                    break
            # Return event types as a stable sequence (event_ids/timestamps differ).
            return [e.type.value for e in ledger.iter_events()]

        run1 = _run_once()
        run2 = _run_once()
        assert run1 == run2
        assert len(run1) > 0

    def test_workers_get_paid_in_memory(self) -> None:
        """Settlement works correctly with in-memory ledger."""
        ledger = InMemoryLedger()
        engine = ClearinghouseEngine(
            ledger=ledger,
            settings=EngineSettings(max_concurrency=1, deterministic=True),
        )

        tasks = [
            TaskSpec(
                id="T1", title="Pay me", bounty=50, deps=[], acceptance=[CommandSpec(cmd="true")]
            ),
        ]
        workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]

        engine.create_run(
            run_id="pay-test",
            payment_rule=PaymentRule.ASK,
            workers=workers,
            tasks=tasks,
        )

        for _ in range(10):
            engine.step(bidder=FixedBidder(), executor=AlwaysPassExecutor())
            state = replay_ledger(events=list(ledger.iter_events()))
            if all(t.status == "DONE" for t in state.tasks.values()):
                break

        state = replay_ledger(events=list(ledger.iter_events()))
        assert state.workers["w1"].balance > 0
