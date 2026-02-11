from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_economy.json_extract import extract_json_object
from agent_economy.llm_router import LLMRouter
from agent_economy.planner import (
    DecompositionPlan,
    planner_system_prompt,
    planner_user_prompt,
    plan_revision_prompt,
)
from agent_economy.schemas import WorkerRuntime, WorkerType, DiscussionMessage
from agent_economy.worker_specs import CommandWorkerSpec


@dataclass(frozen=True)
class PlanCall:
    worker_id: str
    worker_type: str
    plan: DecompositionPlan
    raw_text: str


def _tail(text: str, *, max_chars: int = 2000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "\n... (truncated)\n"


def _run_plan_cmd(
    *,
    cmd: str,
    cwd: Path,
    env: dict[str, str],
    payload: dict[str, Any],
    timeout_sec: int | None,
) -> tuple[DecompositionPlan, str]:
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
            f"plan_cmd failed: rc={proc.returncode}\nstdout:\n{_tail(proc.stdout)}\nstderr:\n{_tail(proc.stderr)}"
        )
    parsed = extract_json_object(raw_out)
    plan = DecompositionPlan.model_validate(parsed)
    return plan, raw_out


def decompose_with_worker(
    *,
    llm: LLMRouter | None,
    planner: WorkerRuntime,
    command_specs: dict[str, CommandWorkerSpec],
    goal: str,
    max_tasks: int,
    file_list: list[str],
    allowed_paths: list[str],
    context_files: dict[str, str] | None = None,
    max_output_tokens: int = 2000,
    cwd: Path,
) -> PlanCall:
    if planner.worker_type == WorkerType.MODEL_AGENT:
        if llm is None:
            raise ValueError("missing LLM router for model planner worker")
        if not planner.model_ref:
            raise ValueError("missing model_ref for planner worker")

        sys = planner_system_prompt()
        user = planner_user_prompt(
            goal=goal,
            max_tasks=max_tasks,
            file_list=file_list,
            allowed_paths=allowed_paths,
            context_files=context_files,
        )
        resp, _usage, raw = llm.call_json(
            model_ref=planner.model_ref,
            system=sys,
            user=user,
            schema=DecompositionPlan,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        )
        plan = (
            resp if isinstance(resp, DecompositionPlan) else DecompositionPlan.model_validate(resp)
        )
        return PlanCall(
            worker_id=planner.worker_id,
            worker_type=planner.worker_type.value,
            plan=plan,
            raw_text=raw,
        )

    if planner.worker_type == WorkerType.EXTERNAL_WORKER:
        spec = command_specs.get(planner.worker_id)
        if spec is None or not spec.plan_cmd:
            raise ValueError("external planner worker missing plan_cmd")

        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in (spec.env or {}).items()})
        env.setdefault("AE_WORKER_ID", planner.worker_id)
        env.setdefault("INST_WORKER_ID", planner.worker_id)

        payload: dict[str, Any] = {
            "schema_version": 1,
            "worker_id": planner.worker_id,
            "goal": goal,
            "max_tasks": int(max_tasks),
            "allowed_paths": list(allowed_paths),
            "files": list(file_list),
        }
        plan, raw = _run_plan_cmd(
            cmd=str(spec.plan_cmd),
            cwd=cwd,
            env=env,
            payload=payload,
            timeout_sec=spec.timeout_sec,
        )
        return PlanCall(
            worker_id=planner.worker_id,
            worker_type=planner.worker_type.value,
            plan=plan,
            raw_text=raw,
        )

    raise ValueError(f"unsupported planner worker type: {planner.worker_type.value}")


def revise_with_worker(
    *,
    llm: LLMRouter | None,
    planner: WorkerRuntime,
    command_specs: dict[str, CommandWorkerSpec],
    goal: str,
    failed_task_id: str,
    failed_task_title: str,
    failed_task_description: str,
    fail_count: int,
    completed_task_ids: list[str],
    remaining_task_ids: list[str],
    file_list: list[str],
    allowed_paths: list[str],
    discussion_history: list[DiscussionMessage],
    failure_notes: str | None = None,
    max_output_tokens: int = 2000,
    cwd: Path,
) -> PlanCall:
    """Request a plan revision from a worker after repeated task failures."""
    if planner.worker_type == WorkerType.MODEL_AGENT:
        if llm is None:
            raise ValueError("missing LLM router for model planner worker")
        if not planner.model_ref:
            raise ValueError("missing model_ref for planner worker")

        sys = planner_system_prompt()
        user = plan_revision_prompt(
            goal=goal,
            failed_task_id=failed_task_id,
            failed_task_title=failed_task_title,
            failed_task_description=failed_task_description,
            fail_count=fail_count,
            completed_task_ids=completed_task_ids,
            remaining_task_ids=remaining_task_ids,
            file_list=file_list,
            allowed_paths=allowed_paths,
            discussion_history=discussion_history,
            failure_notes=failure_notes,
        )
        resp, _usage, raw = llm.call_json(
            model_ref=planner.model_ref,
            system=sys,
            user=user,
            schema=DecompositionPlan,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        )
        plan = (
            resp if isinstance(resp, DecompositionPlan) else DecompositionPlan.model_validate(resp)
        )
        return PlanCall(
            worker_id=planner.worker_id,
            worker_type=planner.worker_type.value,
            plan=plan,
            raw_text=raw,
        )

    if planner.worker_type == WorkerType.EXTERNAL_WORKER:
        spec = command_specs.get(planner.worker_id)
        if spec is None or not spec.plan_cmd:
            raise ValueError("external planner worker missing plan_cmd")

        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in (spec.env or {}).items()})
        env.setdefault("AE_WORKER_ID", planner.worker_id)
        env.setdefault("INST_WORKER_ID", planner.worker_id)

        payload: dict[str, Any] = {
            "schema_version": 1,
            "worker_id": planner.worker_id,
            "revision": True,
            "goal": goal,
            "failed_task_id": failed_task_id,
            "failed_task_title": failed_task_title,
            "failed_task_description": failed_task_description,
            "fail_count": fail_count,
            "completed_task_ids": list(completed_task_ids),
            "remaining_task_ids": list(remaining_task_ids),
            "allowed_paths": list(allowed_paths),
            "files": list(file_list),
            "failure_notes": failure_notes,
        }
        plan, raw = _run_plan_cmd(
            cmd=str(spec.plan_cmd),
            cwd=cwd,
            env=env,
            payload=payload,
            timeout_sec=spec.timeout_sec,
        )
        return PlanCall(
            worker_id=planner.worker_id,
            worker_type=planner.worker_type.value,
            plan=plan,
            raw_text=raw,
        )

    raise ValueError(f"unsupported planner worker type: {planner.worker_type.value}")
