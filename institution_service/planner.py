from __future__ import annotations

from heapq import heappop, heappush

from pydantic import BaseModel, Field

from institution_service.schemas import DiscussionMessage


class PlannedTask(BaseModel):
    id: str
    title: str
    description: str = ""
    deps: list[str] = Field(default_factory=list)
    files_hint: list[str] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)


class DecompositionPlan(BaseModel):
    tasks: list[PlannedTask] = Field(min_length=1)
    notes: str | None = None


def validate_plan_dag(*, plan: DecompositionPlan) -> None:
    ids = [t.id for t in plan.tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("plan has duplicate task ids")
    id_set = set(ids)

    for t in plan.tasks:
        for dep in t.deps:
            if dep not in id_set:
                raise ValueError(f"plan references unknown dep {dep!r} for task_id={t.id!r}")

    in_deg = {tid: 0 for tid in ids}
    children: dict[str, list[str]] = {tid: [] for tid in ids}
    for t in plan.tasks:
        in_deg[t.id] = len(t.deps)
        for dep in t.deps:
            children[dep].append(t.id)

    queue = [tid for tid, d in in_deg.items() if d == 0]
    seen = 0
    while queue:
        cur = queue.pop()
        seen += 1
        for nxt in children[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)

    if seen != len(ids):
        raise ValueError("plan is not a DAG (cycle detected)")


def toposort_plan(*, plan: DecompositionPlan) -> DecompositionPlan:
    validate_plan_dag(plan=plan)

    ids = [t.id for t in plan.tasks]
    index = {tid: i for i, tid in enumerate(ids)}

    in_deg = {tid: 0 for tid in ids}
    children: dict[str, list[str]] = {tid: [] for tid in ids}
    task_by_id = {t.id: t for t in plan.tasks}

    for t in plan.tasks:
        in_deg[t.id] = len(t.deps)
        for dep in t.deps:
            children[dep].append(t.id)

    heap: list[tuple[int, str]] = []
    for tid, d in in_deg.items():
        if d == 0:
            heappush(heap, (index[tid], tid))

    ordered: list[PlannedTask] = []
    while heap:
        _, tid = heappop(heap)
        ordered.append(task_by_id[tid])
        for nxt in children[tid]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                heappush(heap, (index[nxt], nxt))

    if len(ordered) != len(ids):
        raise ValueError("plan is not a DAG (cycle detected)")

    return DecompositionPlan(tasks=ordered, notes=plan.notes)


def planner_system_prompt() -> str:
    return "\n".join(
        [
            "You are a senior engineer decomposing a large goal into small, checkable subtasks.",
            "Return JSON only.",
        ]
    )


def planner_user_prompt(
    *,
    goal: str,
    max_tasks: int,
    file_list: list[str],
    allowed_paths: list[str],
    context_files: dict[str, str] | None = None,
) -> str:
    files_preview = "\n".join(file_list[:400])
    if len(file_list) > 400:
        files_preview += f"\n... ({len(file_list) - 400} more files)"

    allowed_preview = "\n".join(f"- {p}" for p in allowed_paths)

    context_preview = ""
    if context_files:
        parts: list[str] = []
        for path, content in context_files.items():
            safe_path = str(path).strip()
            safe_content = str(content or "").rstrip()
            parts.append(f"BEGIN_FILE {safe_path}\n{safe_content}\nEND_FILE")
        if parts:
            context_preview = "\n".join(["Relevant files (snippets):", *parts, ""])

    lines = [
        "Decompose this goal into a dependency-ordered task DAG with per-task acceptance checks.",
        "",
        f"Goal:\n{goal.strip()}",
        "",
        f"Constraints:\n- max_tasks: {max_tasks}\n- allowed_paths:\n{allowed_preview}",
        "",
        "Repo file list (paths):",
        files_preview,
        "",
    ]
    if context_preview:
        lines.append(context_preview.rstrip())
        lines.append("")

    lines.extend(
        [
            "Rules:",
            "- Make tasks small enough to complete in one patch.",
            "- Prefer behavior-preserving refactors before semantic changes.",
            "- Use stable ids like T1, T2, ...",
            "- deps must reference earlier ids.",
            "- files_hint should list likely relevant paths (keep it short).",
            "- acceptance should be 1-3 shell commands that should PASS when that task is complete.",
            "  Prefer narrow test commands (e.g., `python -m pytest -q path/to/test_file.py`).",
            "  Prefer exactly 1 acceptance command per task; split tasks instead of bundling many checks.",
            "  If multiple test files exist, prefer one task per test file and model shared deps explicitly.",
            "  If an acceptance command is end-to-end (CLI/integration), make it a final task and list explicit deps.",
            "  Avoid pytest node selectors (e.g. `::test_name`) unless you can see the test exists in the provided snippets.",
            "",
            "Return JSON schema:",
            '{ "tasks": [ { "id": "T1", "title": "...", "description": "...", "deps": ["T0"], "files_hint": ["path"], "acceptance": ["cmd"] } ], "notes": "optional" }',
        ]
    )
    return "\n".join(lines)


def plan_revision_prompt(
    *,
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
) -> str:
    """Generate a prompt for revising a plan after repeated task failures."""
    files_preview = "\n".join(file_list[:200])
    if len(file_list) > 200:
        files_preview += f"\n... ({len(file_list) - 200} more files)"

    allowed_preview = "\n".join(f"- {p}" for p in allowed_paths)

    lines = [
        "A task in our plan has failed repeatedly. Revise the plan to address this.",
        "",
        f"Overall goal:\n{goal.strip()}",
        "",
    ]

    if discussion_history:
        lines.append("Public Discussion Board:")
        for msg in discussion_history[-20:]:
            lines.append(f"[{msg.ts.strftime('%H:%M:%S')}] {msg.sender}: {msg.message}")
        lines.append("")

    lines.extend(
        [
            "Failed task:",
            f"- ID: {failed_task_id}",
            f"- Title: {failed_task_title}",
            f"- Description: {failed_task_description}",
            f"- Fail count: {fail_count}",
        ]
    )
    if failure_notes:
        lines.append(f"- Last failure notes: {failure_notes}")

    completed_str = ", ".join(completed_task_ids) if completed_task_ids else "(none)"
    remaining_str = ", ".join(remaining_task_ids) if remaining_task_ids else "(none)"
    lines.extend(
        [
            "",
            f"Completed tasks: {completed_str}",
            f"Remaining tasks (including failed): {remaining_str}",
            "",
            f"Constraints:\n- allowed_paths:\n{allowed_preview}",
            "",
            "Repo file list (paths):",
            files_preview,
            "",
            "Instructions:",
            "- Analyze why the task might be failing.",
            "- Consider splitting the failed task into smaller subtasks.",
            "- Consider adding prerequisite tasks that might be missing.",
            "- Keep completed task IDs stable (don't re-add them).",
            "- Use new IDs for new tasks (e.g., T1a, T1b if splitting T1).",
            "- Preserve deps to completed tasks where needed.",
            "",
            "Return JSON schema:",
            (
                '{ "tasks": [ { "id": "...", "title": "...", "description": "...", '
                '"deps": [...], "files_hint": ["path"], "acceptance": ["cmd"] } ], '
                '"notes": "explain what changed" }'
            ),
        ]
    )
    return "\n".join(lines)
