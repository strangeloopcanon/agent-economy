from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from institution_service.ledger import HashChainedLedger
from institution_service.schemas import EventType, LedgerEvent


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _attempt_key(*, round_id: int, task_id: str, worker_id: str) -> tuple[int, str, str]:
    return (int(round_id), str(task_id), str(worker_id))


def _assignment_record_from_event(event: LedgerEvent) -> dict[str, Any]:
    payload = event.payload or {}
    task_id = str(payload.get("task_id") or "")
    worker_id = str(payload.get("worker_id") or "")
    bid = payload.get("bid") if isinstance(payload.get("bid"), dict) else {}
    return {
        "run_id": event.run_id,
        "assigned_round_id": int(event.round_id),
        "task_id": task_id,
        "worker_id": worker_id,
        "action": {
            "ask": _as_int(bid.get("ask"), default=0),
            "p_success": _as_float(
                bid.get("self_assessed_p_success", bid.get("p_success")),
                default=0.0,
            ),
            "eta_minutes": _as_int(bid.get("eta_minutes"), default=0),
            "notes": bid.get("notes"),
        },
        "award": {
            "score": _as_float(payload.get("score"), default=0.0),
            "expected_cost": _as_float(payload.get("expected_cost"), default=0.0),
            "score_snapshot": payload.get("score_snapshot"),
        },
    }


def extract_attempt_transitions(*, events: list[LedgerEvent]) -> list[dict[str, Any]]:
    pending_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transitions: list[dict[str, Any]] = []
    transition_indices_by_key: dict[tuple[int, str, str], list[int]] = defaultdict(list)

    for event in events:
        payload = event.payload or {}
        event_type = event.type

        if event_type == EventType.TASK_ASSIGNED:
            assignment = _assignment_record_from_event(event)
            task_id = str(assignment.get("task_id") or "")
            if task_id:
                pending_by_task[task_id].append(assignment)
            continue

        if event_type != EventType.TASK_COMPLETED:
            continue

        task_id = str(payload.get("task_id") or "")
        worker_id = str(payload.get("worker_id") or "")
        if not task_id or not worker_id:
            continue

        pending = pending_by_task.get(task_id) or []
        assignment_index = None
        for index, item in enumerate(pending):
            if str(item.get("worker_id") or "") == worker_id:
                assignment_index = index
                break
        if assignment_index is None:
            # Fallback for legacy/manual settlement flows without visible assignment.
            assignment = {
                "run_id": event.run_id,
                "assigned_round_id": None,
                "task_id": task_id,
                "worker_id": worker_id,
                "action": {},
                "award": {"score": None, "expected_cost": None, "score_snapshot": None},
            }
        else:
            assignment = pending.pop(assignment_index)

        key = _attempt_key(round_id=event.round_id, task_id=task_id, worker_id=worker_id)
        transition_indices_by_key[key].append(len(transitions))

        transitions.append(
            {
                "run_id": event.run_id,
                "task_id": task_id,
                "worker_id": worker_id,
                "assigned_round_id": assignment.get("assigned_round_id"),
                "completed_round_id": int(event.round_id),
                "action": assignment.get("action") or {},
                "award": assignment.get("award") or {},
                "outcome": {
                    "success": bool(payload.get("success", False)),
                    "verify_status": str(payload.get("verify_status") or ""),
                    "patch_kind": payload.get("patch_kind"),
                    "sandbox": payload.get("sandbox"),
                },
                "payment": 0.0,
                "penalties": [],
                "reward": 0.0,
            }
        )

    for event in events:
        payload = event.payload or {}
        task_id = str(payload.get("task_id") or "")
        worker_id = str(payload.get("worker_id") or "")
        if not task_id or not worker_id:
            continue

        key = _attempt_key(round_id=event.round_id, task_id=task_id, worker_id=worker_id)
        indices = transition_indices_by_key.get(key) or []
        if not indices:
            continue
        transition = transitions[indices[-1]]

        if event.type == EventType.PAYMENT_MADE:
            transition["payment"] = float(transition["payment"]) + _as_float(
                payload.get("amount"), default=0.0
            )
            continue

        if event.type == EventType.PENALTY_APPLIED:
            penalty = {
                "amount": _as_float(payload.get("amount"), default=0.0),
                "reason": str(payload.get("reason") or "penalty"),
            }
            if "base_penalty" in payload:
                penalty["base_penalty"] = _as_float(payload.get("base_penalty"), default=0.0)
            if "confidence_penalty" in payload:
                penalty["confidence_penalty"] = _as_float(
                    payload.get("confidence_penalty"), default=0.0
                )
            if "reported_p_success" in payload:
                penalty["reported_p_success"] = _as_float(
                    payload.get("reported_p_success"), default=0.0
                )
            transition["penalties"].append(penalty)

    for transition in transitions:
        total_penalty = sum(
            _as_float(p.get("amount"), default=0.0) for p in list(transition.get("penalties") or [])
        )
        transition["reward"] = float(
            _as_float(transition.get("payment"), default=0.0) - total_penalty
        )

    return transitions


def extract_attempt_transitions_from_run(*, run_dir: Path) -> list[dict[str, Any]]:
    ledger = HashChainedLedger(Path(run_dir) / "ledger.jsonl")
    return extract_attempt_transitions(events=list(ledger.iter_events()))
