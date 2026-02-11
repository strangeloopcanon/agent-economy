"""Tests for the gym-compatible InstitutionEnv wrapper."""

from __future__ import annotations

import pytest

from tests.helpers import AlwaysFailExecutor, AlwaysPassExecutor
from agent_economy.engine import BidResult
from agent_economy.gym_env import InstitutionEnv
from agent_economy.schemas import (
    Bid,
    CommandSpec,
    TaskSpec,
    WorkerRuntime,
)


def _single_task_env(**overrides):
    defaults = dict(
        tasks=[
            TaskSpec(
                id="T1",
                title="Do it",
                bounty=20,
                deps=[],
                acceptance=[CommandSpec(cmd="true")],
            )
        ],
        workers=[WorkerRuntime(worker_id="rl", reputation=1.0)],
        agent_id="rl",
        executor=AlwaysPassExecutor(),
    )
    defaults.update(overrides)
    return InstitutionEnv(**defaults)


class TestInstitutionEnv:
    def test_reset_returns_obs_and_info(self) -> None:
        env = _single_task_env()
        obs, info = env.reset()

        assert obs["round_id"] == 0
        assert obs["self"]["worker_id"] == "rl"
        assert info["tasks_total"] == 1
        assert info["tasks_done"] == 0

    def test_step_without_reset_raises(self) -> None:
        env = _single_task_env()
        with pytest.raises(RuntimeError, match="reset"):
            env.step({"task_id": "T1", "ask": 5, "p_success": 0.9, "eta_minutes": 10})

    def test_single_task_completes(self) -> None:
        env = _single_task_env()
        obs, info = env.reset()

        action = {"task_id": "T1", "ask": 5, "self_assessed_p_success": 0.9, "eta_minutes": 10}
        obs, reward, terminated, truncated, info = env.step(action)

        assert terminated is True
        assert truncated is False
        assert reward > 0  # got paid
        assert info["tasks_done"] == 1

    def test_reward_is_balance_delta(self) -> None:
        env = _single_task_env()
        env.reset()

        action = {"task_id": "T1", "ask": 5, "self_assessed_p_success": 0.9, "eta_minutes": 10}
        _, reward, _, _, _ = env.step(action)

        state = env.state
        assert state is not None
        assert reward == state.workers["rl"].balance

    def test_nonzero_starting_balance_uses_delta_only(self) -> None:
        env = _single_task_env(
            workers=[WorkerRuntime(worker_id="rl", balance=10.0, reputation=1.0)]
        )
        env.reset()

        action = {"task_id": "T1", "ask": 5, "self_assessed_p_success": 0.9, "eta_minutes": 10}
        _, reward, terminated, _, info = env.step(action)

        state = env.state
        assert state is not None
        assert terminated is True
        assert info["tasks_done"] == 1
        assert reward == 5.0
        assert state.workers["rl"].balance == 15.0

    def test_failed_task_negative_reward(self) -> None:
        env = _single_task_env(executor=AlwaysFailExecutor())
        env.reset()

        action = {"task_id": "T1", "ask": 5, "self_assessed_p_success": 0.9, "eta_minutes": 10}
        _, reward, terminated, _, _ = env.step(action)

        # Fail penalty means negative balance change.
        assert reward < 0
        assert terminated is False  # task reopened, not exhausted yet

    def test_none_action_skips_bidding(self) -> None:
        env = _single_task_env()
        env.reset()

        obs, reward, terminated, truncated, info = env.step(None)
        # No bid submitted -> no assignment -> no reward.
        assert reward == 0.0
        assert terminated is False

    def test_list_action(self) -> None:
        env = _single_task_env()
        env.reset()

        action = [{"task_id": "T1", "ask": 5, "self_assessed_p_success": 0.9, "eta_minutes": 10}]
        _, reward, terminated, _, _ = env.step(action)
        assert reward > 0
        assert terminated is True

    def test_truncation_on_max_rounds(self) -> None:
        env = _single_task_env(executor=AlwaysFailExecutor(), max_rounds=2)
        env.reset()

        action = {"task_id": "T1", "ask": 5, "self_assessed_p_success": 0.9, "eta_minutes": 10}
        # Round 1
        _, _, terminated, truncated, _ = env.step(action)
        assert terminated is False
        assert truncated is False

        # Round 2 -> truncated
        _, _, terminated, truncated, _ = env.step(action)
        # Task not done, but max rounds hit.
        assert truncated is True

    def test_dependency_ordering(self) -> None:
        tasks = [
            TaskSpec(
                id="T1", title="First", bounty=10, deps=[], acceptance=[CommandSpec(cmd="true")]
            ),
            TaskSpec(
                id="T2",
                title="Second",
                bounty=10,
                deps=["T1"],
                acceptance=[CommandSpec(cmd="true")],
            ),
        ]
        env = InstitutionEnv(
            tasks=tasks,
            workers=[WorkerRuntime(worker_id="rl", reputation=1.0)],
            agent_id="rl",
            executor=AlwaysPassExecutor(),
        )
        env.reset()

        # Bid on T1 first.
        env.step({"task_id": "T1", "ask": 3, "self_assessed_p_success": 0.9, "eta_minutes": 5})
        # Now T2 should be available.
        _, _, terminated, _, info = env.step(
            {"task_id": "T2", "ask": 3, "self_assessed_p_success": 0.9, "eta_minutes": 5}
        )
        assert terminated is True
        assert info["tasks_done"] == 2

    def test_invalid_agent_id_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            InstitutionEnv(
                tasks=[
                    TaskSpec(
                        id="T1", title="X", bounty=1, deps=[], acceptance=[CommandSpec(cmd="true")]
                    )
                ],
                workers=[WorkerRuntime(worker_id="w1")],
                agent_id="nonexistent",
                executor=AlwaysPassExecutor(),
            )

    def test_state_property(self) -> None:
        env = _single_task_env()
        assert env.state is None

        env.reset()
        assert env.state is not None
        assert env.state.run_id.startswith("rl-")

    def test_multi_worker_with_background_bidder(self) -> None:
        class _NpcBidder:
            def get_bids(self, *, worker, ready_tasks, round_id, discussion_history=(), **kw):
                return BidResult(
                    bids=[
                        Bid(task_id=rt.spec.id, ask=8, self_assessed_p_success=0.8, eta_minutes=15)
                        for rt in ready_tasks
                    ][:1]
                )

        env = InstitutionEnv(
            tasks=[
                TaskSpec(
                    id="T1", title="A", bounty=20, deps=[], acceptance=[CommandSpec(cmd="true")]
                ),
                TaskSpec(
                    id="T2", title="B", bounty=20, deps=[], acceptance=[CommandSpec(cmd="true")]
                ),
            ],
            workers=[
                WorkerRuntime(worker_id="rl", reputation=1.0),
                WorkerRuntime(worker_id="npc", reputation=1.0),
            ],
            agent_id="rl",
            executor=AlwaysPassExecutor(),
            background_bidder=_NpcBidder(),
        )
        obs, _ = env.reset()
        assert obs["market"]["num_workers"] == 2

        # RL agent bids on T1, NPC bids on whatever is available.
        action = {"task_id": "T1", "ask": 5, "self_assessed_p_success": 0.9, "eta_minutes": 10}
        obs, reward, terminated, _, info = env.step(action)
        # At least one task should be done.
        assert info["tasks_done"] >= 1
