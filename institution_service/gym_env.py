"""Gym-compatible environment wrapper for the institution-service engine.

Provides a single-agent ``InstitutionEnv`` that translates between the
gym ``reset()``/``step(action)`` interface and the engine's protocol-based
callbacks (``Bidder``/``Executor``).

The RL agent controls one worker's bidding decisions.  Execution and
(optionally) other workers' bids are handled by injected callables.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from institution_service.engine import (
    Bidder,
    BidResult,
    ClearinghouseEngine,
    CostEstimator,
    EngineSettings,
    Executor,
    ReadyTask,
)
from institution_service.ledger import InMemoryLedger, Ledger
from institution_service.observation import build_observation
from institution_service.schemas import (
    Bid,
    DerivedState,
    DiscussionMessage,
    PaymentRule,
    TaskSpec,
    WorkerRuntime,
)
from institution_service.state import SettlementPolicy, replay_ledger


# Max engine.step() calls per env.step() to drive through the full
# bid -> clear -> assign -> execute -> settle cycle.
_MAX_ENGINE_STEPS_PER_ROUND = 20


class _AgentBidder:
    """Internal bidder that yields the RL agent's action for the controlled
    worker and defers to a background bidder for everyone else."""

    def __init__(
        self,
        *,
        agent_id: str,
        background_bidder: Bidder | None,
    ) -> None:
        self._agent_id = agent_id
        self._background_bidder = background_bidder
        self._pending_action: list[Bid] | None = None

    def set_action(self, bids: list[Bid]) -> None:
        self._pending_action = list(bids)

    def get_bids(
        self,
        *,
        worker: WorkerRuntime,
        ready_tasks: Sequence[ReadyTask],
        round_id: int,
        discussion_history: Sequence[DiscussionMessage] = (),
    ) -> BidResult:
        if worker.worker_id == self._agent_id:
            bids = self._pending_action or []
            self._pending_action = None
            return BidResult(bids=bids)

        if self._background_bidder is not None:
            return self._background_bidder.get_bids(
                worker=worker,
                ready_tasks=ready_tasks,
                round_id=round_id,
                discussion_history=discussion_history,
            )
        return BidResult()


def _load_task_specs(tasks: list[TaskSpec]) -> dict[str, TaskSpec]:
    return {t.id: t for t in tasks}


class InstitutionEnv:
    """Single-agent gym-style environment over ``ClearinghouseEngine``.

    The RL agent controls one worker's bidding strategy.  Each ``step()``
    call runs one market round and returns ``(obs, reward, terminated,
    truncated, info)``.

    Usage::

        env = InstitutionEnv(
            tasks=[...],
            workers=[WorkerRuntime(worker_id="rl-agent", ...)],
            agent_id="rl-agent",
            executor=MyExecutor(),
        )
        obs, info = env.reset()
        while True:
            action = policy(obs)  # list of bid dicts
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    """

    def __init__(
        self,
        *,
        tasks: list[TaskSpec],
        workers: list[WorkerRuntime],
        agent_id: str,
        executor: Executor,
        settlement: SettlementPolicy | None = None,
        payment_rule: PaymentRule = PaymentRule.ASK,
        max_rounds: int = 100,
        cost_estimator: CostEstimator | None = None,
        background_bidder: Bidder | None = None,
    ) -> None:
        if agent_id not in {w.worker_id for w in workers}:
            raise ValueError(f"agent_id {agent_id!r} not found in workers")

        self._tasks = list(tasks)
        self._workers = list(workers)
        self._agent_id = agent_id
        self._executor = executor
        self._settlement = settlement or SettlementPolicy()
        self._payment_rule = payment_rule
        self._max_rounds = max_rounds
        self._cost_estimator = cost_estimator
        self._background_bidder = background_bidder
        self._task_specs = _load_task_specs(self._tasks)

        # Mutable per-episode state (initialised in reset).
        self._ledger: Ledger | None = None
        self._engine: ClearinghouseEngine | None = None
        self._bidder: _AgentBidder | None = None
        self._prev_balance: float = 0.0
        self._round: int = 0

    def reset(self, *, run_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode.

        Returns:
            ``(observation, info)`` tuple.
        """
        run_id = run_id or f"rl-{uuid.uuid4().hex[:8]}"

        self._ledger = InMemoryLedger()
        self._engine = ClearinghouseEngine(
            ledger=self._ledger,
            settlement=self._settlement,
            settings=EngineSettings(max_concurrency=1, deterministic=True),
        )
        self._engine.create_run(
            run_id=run_id,
            payment_rule=self._payment_rule,
            workers=self._workers,
            tasks=self._tasks,
        )
        self._bidder = _AgentBidder(
            agent_id=self._agent_id,
            background_bidder=self._background_bidder,
        )
        state = self._current_state()
        self._prev_balance = float(state.workers[self._agent_id].balance)
        self._round = 0
        obs = build_observation(state, self._agent_id, task_specs=self._task_specs)
        info = self._build_info(state=state)
        return obs, info

    def step(
        self, action: dict[str, Any] | list[dict[str, Any]] | None = None
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Run one market round with the agent's bid action.

        Args:
            action: Bid(s) for this round.  Accepts a single bid dict, a
                list of bid dicts, or ``None`` to decline bidding.  Each
                bid dict should contain ``task_id``, ``ask``,
                ``p_success`` (or ``self_assessed_p_success``), and
                ``eta_minutes``.

        Returns:
            ``(observation, reward, terminated, truncated, info)``
        """
        if self._engine is None or self._ledger is None or self._bidder is None:
            raise RuntimeError("call reset() before step()")

        # Parse action into Bid objects.
        bids = self._parse_action(action)
        self._bidder.set_action(bids)

        # The engine's step() may require multiple calls to drive through a
        # full bid -> clear -> assign -> execute -> settle cycle (the first
        # call submits bids and assignments, subsequent calls finalise the
        # inflight futures).  Loop until the agent's balance changes (meaning
        # settlement occurred), all tasks finish, or no progress is made.
        prev_balance = self._prev_balance
        prev_events_len = len(self._ledger)
        state = self._current_state()

        for _ in range(_MAX_ENGINE_STEPS_PER_ROUND):
            self._engine.step(
                bidder=self._bidder,
                executor=self._executor,
                cost_estimator=self._cost_estimator,
            )
            cur_events_len = len(self._ledger)
            state = self._current_state()
            agent_balance = state.workers[self._agent_id].balance
            all_done = all(t.status == "DONE" for t in state.tasks.values())

            # Stop if settlement happened, tasks finished, or engine stalled.
            if agent_balance != prev_balance:
                break
            if all_done:
                break
            if cur_events_len == prev_events_len:
                break  # no progress
            prev_events_len = cur_events_len

        self._round += 1

        # Reward = change in agent's balance this round.
        agent_state = state.workers[self._agent_id]
        reward = agent_state.balance - self._prev_balance
        self._prev_balance = agent_state.balance

        # Terminal conditions.
        all_done = all(t.status == "DONE" for t in state.tasks.values())
        # Check if all remaining tasks have exhausted max_attempts.
        all_exhausted = all(
            state.tasks[tid].status == "DONE"
            or state.tasks[tid].fail_count >= self._task_specs[tid].max_attempts
            for tid in state.tasks
        )
        terminated = all_done or all_exhausted
        truncated = self._round >= self._max_rounds and not terminated

        obs = build_observation(state, self._agent_id, task_specs=self._task_specs)
        info = self._build_info(state=state)
        return obs, reward, terminated, truncated, info

    @property
    def state(self) -> DerivedState | None:
        """Current full market state, or None if not yet reset."""
        if self._ledger is None:
            return None
        return self._current_state()

    def _current_state(self) -> DerivedState:
        if self._ledger is None:
            raise RuntimeError("environment has no ledger; call reset() first")
        return replay_ledger(
            events=list(self._ledger.iter_events()),
            settlement=self._settlement,
        )

    def _parse_action(self, action: dict[str, Any] | list[dict[str, Any]] | None) -> list[Bid]:
        if action is None:
            return []
        if isinstance(action, dict):
            action = [action]
        bids: list[Bid] = []
        for raw in action:
            if isinstance(raw, Bid):
                bids.append(raw)
            else:
                bids.append(Bid.model_validate(raw))
        return bids

    def _build_info(self, *, state: DerivedState) -> dict[str, Any]:
        return {
            "run_id": state.run_id,
            "round_id": state.round_id,
            "tasks_done": sum(1 for t in state.tasks.values() if t.status == "DONE"),
            "tasks_total": len(state.tasks),
        }
