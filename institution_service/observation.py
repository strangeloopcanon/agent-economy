"""Per-agent observation builder for partial observability.

Filters the full market state (DerivedState) to what a specific worker
should see, hiding private information about other workers.
"""

from __future__ import annotations

from typing import Any

from institution_service.schemas import DerivedState, TaskSpec


def build_observation(
    state: DerivedState,
    worker_id: str,
    *,
    task_specs: dict[str, TaskSpec] | None = None,
) -> dict[str, Any]:
    """Build a per-agent observation from the full market state.

    Separates information into:
    - ``self``: the requesting worker's full private state
    - ``tasks``: public task information (specs, bounties, statuses)
    - ``market``: aggregate market statistics (no per-worker detail)
    - ``round_id``: current round
    - ``discussion``: public discussion history

    Other workers' balances, reputations, and individual stats are hidden.

    Args:
        state: Full derived state from ``replay_ledger()``.
        worker_id: The worker requesting the observation.
        task_specs: Optional task specifications for richer task context.
            When provided, each task includes ``title``, ``description``,
            ``deps``, ``max_attempts``, and ``verify_mode``.

    Returns:
        Observation dict suitable for use as RL state input.

    Raises:
        KeyError: If ``worker_id`` is not present in ``state.workers``.
    """
    worker = state.workers[worker_id]

    # --- Self (private) ---
    self_obs: dict[str, Any] = {
        "worker_id": worker.worker_id,
        "worker_type": worker.worker_type.value,
        "balance": worker.balance,
        "reputation": worker.reputation,
        "assigned_task": worker.assigned_task,
        "wins": worker.wins,
        "completions": worker.completions,
        "failures": worker.failures,
    }
    if worker.model_ref is not None:
        self_obs["model_ref"] = worker.model_ref

    # --- Tasks (public) ---
    tasks_obs: list[dict[str, Any]] = []
    for task_id, rt in state.tasks.items():
        task_entry: dict[str, Any] = {
            "task_id": task_id,
            "status": rt.status,
            "bounty_current": rt.bounty_current,
            "bounty_original": rt.bounty_original,
            "fail_count": rt.fail_count,
            "assigned": rt.assigned_worker is not None,
        }
        if task_specs and task_id in task_specs:
            spec = task_specs[task_id]
            task_entry["title"] = spec.title
            task_entry["description"] = spec.description
            task_entry["deps"] = list(spec.deps)
            task_entry["max_attempts"] = spec.max_attempts
            task_entry["verify_mode"] = spec.verify_mode.value
        tasks_obs.append(task_entry)

    # --- Market (aggregate, no per-worker detail) ---
    n_workers = len(state.workers)
    n_idle = sum(1 for w in state.workers.values() if w.assigned_task is None)
    n_tasks_todo = sum(1 for t in state.tasks.values() if t.status == "TODO")
    n_tasks_done = sum(1 for t in state.tasks.values() if t.status == "DONE")
    n_tasks_assigned = sum(1 for t in state.tasks.values() if t.status == "ASSIGNED")
    n_tasks_review = sum(1 for t in state.tasks.values() if t.status == "REVIEW")
    market_obs: dict[str, Any] = {
        "num_workers": n_workers,
        "num_idle_workers": n_idle,
        "tasks_todo": n_tasks_todo,
        "tasks_assigned": n_tasks_assigned,
        "tasks_review": n_tasks_review,
        "tasks_done": n_tasks_done,
        "payment_rule": state.payment_rule.value,
    }

    # --- Discussion (public) ---
    discussion_obs = [
        {
            "sender": msg.sender,
            "message": msg.message,
            "ts": msg.ts.isoformat(),
        }
        for msg in state.discussion_history
    ]

    return {
        "round_id": state.round_id,
        "self": self_obs,
        "tasks": tasks_obs,
        "market": market_obs,
        "discussion": discussion_obs,
    }
