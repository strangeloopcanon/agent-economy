from __future__ import annotations

from collections import defaultdict

from institution_service.ledger import HashChainedLedger
from institution_service.schemas import EventType, VerifyStatus
from institution_service.state import replay_ledger


def release_judges_holdbacks(*, ledger: HashChainedLedger) -> int:
    """
    Release withheld (holdback) payments once a run fully completes.

    A holdback is recorded as `holdback_amount` on the PASS `task_completed` event. This function
    appends `payment_made` events with `reason=holdback_release` for any remaining unpaid holdbacks,
    but only when the run is in a terminal state: all tasks are `DONE` or `REVIEW`.
    """

    events = list(ledger.iter_events())
    if not events:
        return 0

    state = replay_ledger(events=events)
    if any(t.status not in {"DONE", "REVIEW"} for t in state.tasks.values()):
        return 0

    holdbacks: dict[tuple[str, str], float] = {}
    meta: dict[tuple[str, str], dict] = {}
    for e in events:
        if getattr(e, "type", None) != EventType.TASK_COMPLETED:
            continue
        p = getattr(e, "payload", {}) or {}
        if str(p.get("verify_status") or "") != VerifyStatus.PASS.value:
            continue
        try:
            hb = float(p.get("holdback_amount") or 0.0)
        except Exception:
            continue
        if hb <= 0:
            continue
        task_id = str(p.get("task_id") or "").strip()
        worker_id = str(p.get("worker_id") or "").strip()
        if not task_id or not worker_id:
            continue
        holdbacks[(task_id, worker_id)] = hb

        bid = p.get("bid") if isinstance(p.get("bid"), dict) else {}
        meta[(task_id, worker_id)] = {
            "bounty": p.get("bounty_current"),
            "ask": bid.get("ask"),
        }

    if not holdbacks:
        return 0

    released: dict[tuple[str, str], float] = defaultdict(float)
    for e in events:
        if getattr(e, "type", None) != EventType.PAYMENT_MADE:
            continue
        p = getattr(e, "payload", {}) or {}
        if str(p.get("reason") or "") != "holdback_release":
            continue
        task_id = str(p.get("task_id") or "").strip()
        worker_id = str(p.get("worker_id") or "").strip()
        if not task_id or not worker_id:
            continue
        try:
            released[(task_id, worker_id)] += float(p.get("amount") or 0.0)
        except Exception:
            continue

    appended = 0
    for (task_id, worker_id), holdback_amount in sorted(holdbacks.items()):
        remaining = float(holdback_amount) - float(released.get((task_id, worker_id), 0.0))
        if remaining <= 0:
            continue
        info = meta.get((task_id, worker_id), {})
        ledger.append(
            EventType.PAYMENT_MADE,
            run_id=state.run_id,
            round_id=state.round_id,
            payload={
                "task_id": task_id,
                "worker_id": worker_id,
                "amount": remaining,
                "payment_rule": state.payment_rule.value,
                "bounty": info.get("bounty"),
                "ask": info.get("ask"),
                "reason": "holdback_release",
            },
        )
        appended += 1

    if appended:
        ledger.verify_chain()
    return appended
