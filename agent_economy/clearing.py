from __future__ import annotations

from dataclasses import dataclass

from agent_economy.schemas import Bid, TaskRuntime, WorkerRuntime


@dataclass(frozen=True)
class BidSubmission:
    worker_id: str
    bid: Bid
    expected_cost: float = 0.0


@dataclass(frozen=True)
class Assignment:
    task_id: str
    worker_id: str
    bid: Bid
    score: float
    expected_cost: float = 0.0
    score_breakdown: dict[str, float] | None = None


def score_bid_breakdown(
    *,
    bounty: int,
    reputation: float,
    bid: Bid,
    expected_cost: float = 0.0,
) -> dict[str, float]:
    bounty_f = float(bounty)
    reputation_f = float(reputation)
    ask_f = float(bid.ask)
    expected_cost_f = float(expected_cost)
    p_success_f = float(bid.self_assessed_p_success)

    # Penalize low-confidence bids beyond their lower expected value: failing burns market time.
    # Scale penalty with reputation: high-rep workers (1.25) face full penalty,
    # low-rep workers (0.5) face reduced penalty to give them a recovery path.
    # Formula: penalty_scale = (reputation - 0.5) / 0.75 clamped to [0, 1]
    penalty_scale = max(0.0, min(1.0, (reputation_f - 0.5) / 0.75))
    failure_penalty = penalty_scale * 0.5 * bounty_f
    score = (
        reputation_f * p_success_f * bounty_f
        - ask_f
        - expected_cost_f
        - (1.0 - p_success_f) * failure_penalty
    )
    return {
        "bounty": bounty_f,
        "reputation": reputation_f,
        "p_success": p_success_f,
        "ask": ask_f,
        "expected_cost": expected_cost_f,
        "failure_penalty": float(failure_penalty),
        "score": float(score),
    }


def score_bid(*, bounty: int, reputation: float, bid: Bid, expected_cost: float = 0.0) -> float:
    return float(
        score_bid_breakdown(
            bounty=bounty,
            reputation=reputation,
            bid=bid,
            expected_cost=expected_cost,
        )["score"]
    )


def choose_assignments(
    *,
    ready_tasks: list[TaskRuntime],
    available_workers: list[WorkerRuntime],
    bids_by_task: dict[str, list[BidSubmission]],
) -> list[Assignment]:
    tasks_by_id = {t.task_id: t for t in ready_tasks}
    workers_by_id = {w.worker_id: w for w in available_workers}

    def _r6(v: float) -> float:
        return round(float(v), 6)

    def _desc_str_key(s: str) -> tuple[int, ...]:
        # Used for deterministic descending ordering of IDs within sort keys.
        # Terminator ensures correct reverse ordering when one string is a prefix of another.
        return tuple([-ord(ch) for ch in s] + [1])

    candidates: dict[tuple[str, str], Assignment] = {}
    for task_id, subs in bids_by_task.items():
        task = tasks_by_id.get(task_id)
        if task is None:
            continue
        for sub in subs:
            worker = workers_by_id.get(sub.worker_id)
            if worker is None:
                continue

            bid = sub.bid
            breakdown = score_bid_breakdown(
                bounty=task.bounty_current,
                reputation=worker.reputation,
                bid=bid,
                expected_cost=sub.expected_cost,
            )
            score = float(breakdown["score"])
            if score <= 0:
                continue

            key = (task_id, worker.worker_id)
            cur = candidates.get(key)
            if cur is not None and _r6(cur.score) > _r6(score):
                continue

            candidates[key] = Assignment(
                task_id=task_id,
                worker_id=worker.worker_id,
                bid=bid,
                score=float(score),
                expected_cost=sub.expected_cost,
                score_breakdown=breakdown,
            )

    edges = list(candidates.values())
    edges.sort(
        key=lambda a: (
            -_r6(a.score),
            -_r6(workers_by_id[a.worker_id].reputation),
            int(a.bid.eta_minutes),
            a.task_id,
            _desc_str_key(a.worker_id),
        )
    )

    used_tasks: set[str] = set()
    used_workers: set[str] = set()
    selected: list[Assignment] = []
    for a in edges:
        if a.task_id in used_tasks:
            continue
        if a.worker_id in used_workers:
            continue
        used_tasks.add(a.task_id)
        used_workers.add(a.worker_id)
        selected.append(a)

    return selected
