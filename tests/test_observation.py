"""Tests for the per-agent observation builder."""

from __future__ import annotations

import pytest

from agent_economy.observation import build_observation
from agent_economy.schemas import (
    CommandSpec,
    DerivedState,
    DiscussionMessage,
    PaymentRule,
    TaskRuntime,
    TaskSpec,
    WorkerRuntime,
)
from datetime import datetime, UTC


def _make_state() -> DerivedState:
    return DerivedState(
        run_id="obs-test",
        round_id=3,
        tasks={
            "T1": TaskRuntime(
                task_id="T1",
                status="DONE",
                bounty_current=20,
                bounty_original=20,
                fail_count=0,
                assigned_worker=None,
            ),
            "T2": TaskRuntime(
                task_id="T2",
                status="ASSIGNED",
                bounty_current=40,
                bounty_original=40,
                fail_count=1,
                assigned_worker="w1",
            ),
            "T3": TaskRuntime(
                task_id="T3",
                status="TODO",
                bounty_current=30,
                bounty_original=30,
                fail_count=0,
                assigned_worker=None,
            ),
        },
        workers={
            "w1": WorkerRuntime(
                worker_id="w1",
                balance=15.0,
                reputation=1.06,
                assigned_task="T2",
                wins=2,
                completions=1,
                failures=0,
            ),
            "w2": WorkerRuntime(
                worker_id="w2",
                balance=-3.0,
                reputation=0.8,
                assigned_task=None,
                wins=1,
                completions=0,
                failures=1,
            ),
        },
        payment_rule=PaymentRule.ASK,
        discussion_history=[
            DiscussionMessage(
                sender="system", message="hello", ts=datetime(2025, 1, 1, tzinfo=UTC)
            ),
        ],
    )


class TestBuildObservation:
    def test_self_is_own_worker(self) -> None:
        obs = build_observation(_make_state(), "w1")
        assert obs["self"]["worker_id"] == "w1"
        assert obs["self"]["balance"] == 15.0
        assert obs["self"]["reputation"] == 1.06
        assert obs["self"]["assigned_task"] == "T2"

    def test_other_worker_state_hidden(self) -> None:
        obs = build_observation(_make_state(), "w1")
        # No per-worker detail about w2 should appear anywhere except
        # in the aggregate market stats.
        flat = str(obs)
        # w2's balance and reputation must not leak.
        assert "-3.0" not in flat
        assert "0.8" not in flat

    def test_tasks_are_public(self) -> None:
        obs = build_observation(_make_state(), "w2")
        task_ids = {t["task_id"] for t in obs["tasks"]}
        assert task_ids == {"T1", "T2", "T3"}
        # Assigned flag is a boolean (not the worker id).
        t2 = next(t for t in obs["tasks"] if t["task_id"] == "T2")
        assert t2["assigned"] is True
        assert "assigned_worker" not in t2

    def test_market_aggregates(self) -> None:
        obs = build_observation(_make_state(), "w1")
        m = obs["market"]
        assert m["num_workers"] == 2
        assert m["num_idle_workers"] == 1
        assert m["tasks_done"] == 1
        assert m["tasks_assigned"] == 1
        assert m["tasks_todo"] == 1
        assert m["tasks_review"] == 0

    def test_round_id_present(self) -> None:
        obs = build_observation(_make_state(), "w1")
        assert obs["round_id"] == 3

    def test_discussion_included(self) -> None:
        obs = build_observation(_make_state(), "w1")
        assert len(obs["discussion"]) == 1
        assert obs["discussion"][0]["sender"] == "system"

    def test_unknown_worker_raises(self) -> None:
        with pytest.raises(KeyError):
            build_observation(_make_state(), "w_nonexistent")

    def test_with_task_specs(self) -> None:
        specs = {
            "T1": TaskSpec(
                id="T1",
                title="Setup",
                description="Set up project",
                bounty=20,
                deps=[],
                acceptance=[CommandSpec(cmd="true")],
            ),
            "T3": TaskSpec(
                id="T3",
                title="Tests",
                description="Add tests",
                bounty=30,
                deps=["T1"],
                acceptance=[CommandSpec(cmd="true")],
            ),
        }
        obs = build_observation(_make_state(), "w1", task_specs=specs)
        t1 = next(t for t in obs["tasks"] if t["task_id"] == "T1")
        assert t1["title"] == "Setup"
        assert t1["description"] == "Set up project"
        assert t1["deps"] == []

        t3 = next(t for t in obs["tasks"] if t["task_id"] == "T3")
        assert t3["deps"] == ["T1"]
        assert t3["verify_mode"] == "commands"

        # T2 has no spec provided, so no extra fields.
        t2 = next(t for t in obs["tasks"] if t["task_id"] == "T2")
        assert "title" not in t2

    def test_without_task_specs(self) -> None:
        obs = build_observation(_make_state(), "w1")
        for t in obs["tasks"]:
            assert "title" not in t
            assert "description" not in t
