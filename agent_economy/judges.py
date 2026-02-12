from __future__ import annotations

import difflib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_economy.json_extract import extract_json_object
from agent_economy.llm_router import LLMRouter
from agent_economy.schemas import SubmissionKind, TaskSpec, VerifyStatus, WorkerRuntime, WorkerType
from agent_economy.verify import CommandResult

from agent_economy.worker_specs import CommandWorkerSpec


class JudgeDecision(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)


def aggregate_judge_votes(
    *, decisions: list[JudgeDecision], min_passes: int | None
) -> VerifyStatus:
    if not decisions:
        return VerifyStatus.INFRA

    required = int(min_passes) if min_passes is not None else (len(decisions) // 2 + 1)
    passes = sum(1 for d in decisions if d.verdict == "PASS")
    if passes >= required:
        return VerifyStatus.PASS
    # Pure-majority settlement: if the required threshold isn't met, it's a FAIL.
    return VerifyStatus.FAIL


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def build_unified_diff_summary(
    *,
    workspace_dir: Path,
    sandbox_dir: Path,
    rel_paths: list[str],
    max_bytes: int = 120_000,
    max_files: int = 25,
) -> str:
    chunks: list[str] = []
    used = 0
    for rel in rel_paths[:max_files]:
        before_path = workspace_dir / rel
        after_path = sandbox_dir / rel

        before = _read_text(before_path) if before_path.exists() and before_path.is_file() else ""
        after = _read_text(after_path) if after_path.exists() and after_path.is_file() else ""

        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                n=3,
            )
        )
        if not diff.strip():
            continue

        if used + len(diff) > max_bytes:
            remain = max(0, max_bytes - used)
            if remain > 0:
                chunks.append(diff[:remain])
            chunks.append("\n... (diff truncated)\n")
            break

        chunks.append(diff)
        used += len(diff)

    if not chunks:
        return "(no textual diff available)"
    return "\n".join(chunks).rstrip() + "\n"


def _format_command_results(results: list[CommandResult]) -> str:
    if not results:
        return "(none)\n"

    def _tail(text: str, *, max_chars: int = 2000) -> str:
        text = text or ""
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 16] + "\n... (truncated)\n"

    blocks: list[str] = []
    for r in results:
        blocks.append(
            "\n".join(
                [
                    f"- cmd: {r.cmd}",
                    f"  returncode: {r.returncode}",
                    f"  timed_out: {r.timed_out}",
                    f"  duration_s: {r.duration_s:.2f}",
                    "  stdout:",
                    _tail(r.stdout),
                    "  stderr:",
                    _tail(r.stderr),
                ]
            )
        )
    return "\n".join(blocks).rstrip() + "\n"


def judge_system_prompt() -> str:
    return "\n".join(
        [
            "You are a strict verifier for a task submission.",
            "You are given: task description, command outputs (if any), and either a unified diff or submission text.",
            "Decide whether the submission satisfies the task and is safe/correct.",
            "",
            "Rules:",
            "- If the submission does not implement the task, verdict=FAIL.",
            "- If the submission introduces obvious bugs, unsafe behavior, or ignores constraints, verdict=FAIL.",
            "- Be conservative: when in doubt, verdict=FAIL and explain what’s missing.",
            "",
            "Return JSON only with fields:",
            '{ "verdict": "PASS"|"FAIL", "confidence": 0..1, "reasons": [..], "required_changes": [..] }',
        ]
    )


def judge_user_prompt(
    *,
    task: TaskSpec,
    public: list[CommandResult],
    hidden: list[CommandResult],
    diff_text: str,
    submission_kind: SubmissionKind,
    submission_text: str | None,
) -> str:
    lines: list[str] = [
        f"Task: {task.id} — {task.title}",
        task.description.strip() or "(no description)",
        "",
        "Submission kind:",
        submission_kind.value,
        "",
        "Public command results:",
        _format_command_results(public),
        "Hidden command results:",
        _format_command_results(hidden),
    ]
    if submission_kind == SubmissionKind.PATCH:
        lines.extend(["Unified diff:", diff_text])
    else:
        lines.extend(["Submission text:", submission_text or "(missing submission text)"])
    return "\n".join(lines)


@dataclass(frozen=True)
class JudgeCall:
    worker_id: str
    worker_type: str
    decision: JudgeDecision
    raw_text: str


def _tail(text: str, *, max_chars: int = 2000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "\n... (truncated)\n"


def _run_command_judge(
    *,
    cmd: str,
    cwd: Path,
    env: dict[str, str],
    payload: dict[str, Any],
    timeout_sec: int | None,
) -> tuple[JudgeDecision, str]:
    raw_in = json.dumps(payload, ensure_ascii=False) + "\n"
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=True,
        text=True,
        input=raw_in,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    raw_out = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(
            f"judge_cmd failed: rc={proc.returncode}\nstdout:\n{_tail(proc.stdout)}\nstderr:\n{_tail(proc.stderr)}"
        )
    parsed = extract_json_object(raw_out)
    decision = JudgeDecision.model_validate(parsed)
    return decision, raw_out


def run_judges_with_workers(
    *,
    llm: LLMRouter | None,
    judge_workers: list[WorkerRuntime],
    command_specs: dict[str, CommandWorkerSpec],
    task: TaskSpec,
    public: list[CommandResult],
    hidden: list[CommandResult],
    diff_text: str,
    submission_kind: SubmissionKind,
    submission_text: str | None,
    required_passes: int,
    max_output_tokens: int = 1200,
    cwd: Path,
) -> tuple[VerifyStatus, list[JudgeCall]]:
    sys = judge_system_prompt()
    user = judge_user_prompt(
        task=task,
        public=public,
        hidden=hidden,
        diff_text=diff_text,
        submission_kind=submission_kind,
        submission_text=submission_text,
    )

    calls: list[JudgeCall] = []
    for w in judge_workers:
        if w.worker_type == WorkerType.MODEL_AGENT:
            if llm is None or not w.model_ref:
                continue
            decision, _usage, raw = llm.call_json(
                model_ref=w.model_ref,
                system=sys,
                user=user,
                schema=JudgeDecision,
                temperature=0.0,
                max_output_tokens=max_output_tokens,
            )
            calls.append(
                JudgeCall(
                    worker_id=w.worker_id,
                    worker_type=w.worker_type.value,
                    decision=decision,
                    raw_text=raw,
                )
            )
            continue

        if w.worker_type == WorkerType.EXTERNAL_WORKER:
            spec = command_specs.get(w.worker_id)
            if spec is None or not spec.judge_cmd:
                continue
            env = dict(os.environ)
            env.update({str(k): str(v) for k, v in (spec.env or {}).items()})
            env.setdefault("AE_WORKER_ID", w.worker_id)
            env.setdefault("INST_WORKER_ID", w.worker_id)

            payload: dict[str, Any] = {
                "schema_version": 1,
                "worker_id": w.worker_id,
                "task": task.model_dump(mode="json"),
                "public": [asdict(r) for r in public],
                "hidden": [asdict(r) for r in hidden],
                "diff": diff_text,
                "submission_kind": submission_kind.value,
                "submission_text": submission_text,
            }
            decision, raw = _run_command_judge(
                cmd=str(spec.judge_cmd),
                cwd=cwd,
                env=env,
                payload=payload,
                timeout_sec=spec.timeout_sec,
            )
            calls.append(
                JudgeCall(
                    worker_id=w.worker_id,
                    worker_type=w.worker_type.value,
                    decision=decision,
                    raw_text=raw,
                )
            )
            continue

    status = aggregate_judge_votes(
        decisions=[c.decision for c in calls],
        min_passes=int(required_passes),
    )
    return status, calls
