from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UsageTotals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "UsageTotals") -> "UsageTotals":
        return UsageTotals(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


def _parse_ts(value: str) -> datetime:
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _coerce_usage(raw: Any) -> UsageTotals:
    if not isinstance(raw, dict):
        return UsageTotals()
    try:
        return UsageTotals(
            calls=int(raw.get("calls") or 0),
            input_tokens=int(raw.get("input_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
        )
    except Exception:
        return UsageTotals()


def summarize_run(*, run_dir: Path) -> dict[str, Any]:
    ledger = run_dir / "ledger.jsonl"
    if not ledger.exists():
        raise SystemExit(f"missing ledger: {ledger}")

    task_ids: set[str] = set()
    done_task_ids: set[str] = set()
    max_round_id = 0

    first_ts: datetime | None = None
    last_ts: datetime | None = None

    usage_by_worker: dict[str, UsageTotals] = defaultdict(UsageTotals)
    bid_usage_by_worker: dict[str, UsageTotals] = defaultdict(UsageTotals)
    patch_usage_by_worker: dict[str, UsageTotals] = defaultdict(UsageTotals)

    bid_cost = 0.0
    patch_cost = 0.0

    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)

        ts = _parse_ts(e.get("ts") or "")
        if first_ts is None:
            first_ts = ts
        last_ts = ts

        max_round_id = max(max_round_id, int(e.get("round_id") or 0))
        typ = str(e.get("type") or "")
        payload = e.get("payload") or {}

        if typ == "task_created":
            tid = str(payload.get("task_id") or "").strip()
            if tid:
                task_ids.add(tid)

        if typ == "task_completed":
            tid = str(payload.get("task_id") or "").strip()
            if tid and str(payload.get("verify_status") or "") == "PASS":
                done_task_ids.add(tid)

        if typ in {"bid_submitted", "patch_submitted"}:
            worker_id = str(payload.get("worker_id") or "").strip()
            if worker_id:
                usage = _coerce_usage(payload.get("llm_usage"))
                usage_by_worker[worker_id] = usage_by_worker[worker_id] + usage
                if typ == "bid_submitted":
                    bid_usage_by_worker[worker_id] = bid_usage_by_worker[worker_id] + usage
                else:
                    patch_usage_by_worker[worker_id] = patch_usage_by_worker[worker_id] + usage

        if typ == "penalty_applied":
            if not isinstance(payload, dict):
                continue
            reason = payload.get("reason")
            amount = payload.get("amount")
            if not isinstance(amount, (int, float)):
                continue
            if reason == "bid_usage_cost":
                bid_cost += float(amount)
            if reason == "usage_cost":
                patch_cost += float(amount)

    span_s = None
    if first_ts and last_ts:
        span_s = (last_ts - first_ts).total_seconds()

    total_usage = UsageTotals()
    total_bid_usage = UsageTotals()
    total_patch_usage = UsageTotals()
    for wid in usage_by_worker:
        total_usage += usage_by_worker[wid]
        total_bid_usage += bid_usage_by_worker[wid]
        total_patch_usage += patch_usage_by_worker[wid]

    return {
        "run_dir": str(run_dir),
        "tasks_total": len(task_ids),
        "tasks_done": len(done_task_ids),
        "round_id_max": max_round_id,
        "ledger_span_s": span_s,
        "usage_total": total_usage.__dict__,
        "usage_bid": total_bid_usage.__dict__,
        "usage_patch": total_patch_usage.__dict__,
        "cost_bid_usage": round(bid_cost, 4),
        "cost_patch_usage": round(patch_cost, 4),
        "workers": {
            wid: {
                "usage_total": usage_by_worker[wid].__dict__,
                "usage_bid": bid_usage_by_worker[wid].__dict__,
                "usage_patch": patch_usage_by_worker[wid].__dict__,
            }
            for wid in sorted(usage_by_worker)
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize an agent-economy run directory.")
    p.add_argument("run_dir", nargs="+", type=Path)
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = p.parse_args()

    summaries = [summarize_run(run_dir=d) for d in args.run_dir]

    if args.json:
        print(json.dumps(summaries, indent=2))
        return

    for s in summaries:
        span = s["ledger_span_s"]
        span_str = "n/a" if span is None else f"{span:.1f}s"
        print(
            f"{s['run_dir']}: done={s['tasks_done']}/{s['tasks_total']} "
            f"rounds={s['round_id_max']} span={span_str} "
            f"tok_in={s['usage_total']['input_tokens']} tok_out={s['usage_total']['output_tokens']} "
            f"cost_bid={s['cost_bid_usage']} cost_patch={s['cost_patch_usage']}"
        )
        for wid, w in s["workers"].items():
            u = w["usage_total"]
            print(f"  - {wid}: calls={u['calls']} in={u['input_tokens']} out={u['output_tokens']}")


if __name__ == "__main__":
    main()
