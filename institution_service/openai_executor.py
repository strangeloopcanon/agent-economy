from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from institution_service.engine import ExecutionOutcome, ReadyTask
from institution_service.judges import build_unified_diff_summary, run_judges_with_workers
from institution_service.llm_router import LLMRouter
from institution_service.openai_bidder import DEFAULT_PERSONAS
from institution_service.prompts import patch_prompt, system_prompt
from institution_service.sandbox import (
    Sandbox,
    apply_file_blocks,
    apply_unified_diff,
    apply_unified_diff_path,
    artifact_for,
    build_patch_from_dirs,
    enforce_allowed_paths,
    extract_file_blocks,
    extract_git_diff,
    parse_patch_changes,
    write_command_results_json,
    write_text_atomic,
)
from institution_service.schemas import (
    Bid,
    TaskRuntime,
    TaskSpec,
    VerifyMode,
    VerifyStatus,
    WorkerRuntime,
    DiscussionMessage,
)
from institution_service.verify import CommandResult, all_passed, run_commands
from institution_service.worker_refs import resolve_worker_refs
from institution_service.worker_specs import CommandWorkerSpec


def _read_hint_files(*, root: Path, rel_paths: list[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for rel_path in rel_paths:
        name = Path(rel_path).name
        if name == ".env" or name.startswith(".env."):
            files[rel_path] = "<redacted>"
            continue
        p = root / rel_path
        if p.exists():
            if p.is_file():
                try:
                    files[rel_path] = p.read_text(encoding="utf-8")
                except Exception:
                    files[rel_path] = "<unreadable>"
            elif p.is_dir():
                children = [
                    str(child.relative_to(root))
                    for child in sorted(p.rglob("*"))
                    if child.is_file()
                ]
                preview = "\n".join(children[:50])
                if len(children) > 50:
                    preview += f"\n... ({len(children) - 50} more files)"
                files[rel_path] = f"<directory>\n{preview}\n"
            else:
                files[rel_path] = "<not a regular file>"
        else:
            files[rel_path] = "<missing>"
    return files


@dataclass(frozen=True)
class ExecutorSettings:
    max_patch_output_tokens: int = 6000
    scrub_secrets_in_verification: bool = True
    judge_workers: list[str] = field(default_factory=list)
    judge_max_output_tokens: int = 1200
    judge_include_self: bool = True


class OpenAIExecutor:
    def __init__(
        self,
        *,
        llm: LLMRouter,
        workspace_dir: Path,
        run_dir: Path,
        workers: list[WorkerRuntime],
        command_specs: dict[str, CommandWorkerSpec] | None = None,
        settings: ExecutorSettings | None = None,
    ) -> None:
        self._llm = llm
        self._workspace_dir = workspace_dir
        self._run_dir = run_dir
        self._workers = list(workers)
        self._command_specs = dict(command_specs or {})
        self._settings = settings or ExecutorSettings()
        self._sandbox = Sandbox(run_dir=run_dir)

    def execute(
        self,
        *,
        worker: WorkerRuntime,
        task: TaskSpec,
        bid: Bid,
        round_id: int,
        discussion_history: list[DiscussionMessage],
    ) -> ExecutionOutcome:
        _ = bid
        if not worker.model_ref:
            return ExecutionOutcome(status=VerifyStatus.INFRA, notes="missing model_ref")

        sandbox_dir = self._sandbox.create(
            task_id=task.id, worker_id=worker.worker_id, round_id=round_id
        )
        try:
            sandbox_rel = str(sandbox_dir.relative_to(self._run_dir))
        except Exception:
            sandbox_rel = str(sandbox_dir)
        work_dir = sandbox_dir / "workspace"
        self._sandbox.copy_workspace(workspace_dir=self._workspace_dir, sandbox_dir=work_dir)

        hint_files = _read_hint_files(root=work_dir, rel_paths=list(task.files_hint))
        ready = ReadyTask(
            spec=task,
            runtime=TaskRuntime(
                task_id=task.id, bounty_current=task.bounty, bounty_original=task.bounty
            ),
        )

        persona = DEFAULT_PERSONAS.get(worker.worker_id)
        sys = system_prompt(worker=worker, persona=None if persona is None else persona.persona)
        user = patch_prompt(task=ready, files=hint_files, discussion_history=discussion_history)

        write_text_atomic(sandbox_dir / "prompt_system.txt", sys)
        write_text_atomic(sandbox_dir / "prompt_user.txt", user)

        patch_artifacts = [
            artifact_for(
                sandbox_dir / "prompt_system.txt",
                name="prompt_system.txt",
                media_type="text/plain",
                root=self._run_dir,
            ),
            artifact_for(
                sandbox_dir / "prompt_user.txt",
                name="prompt_user.txt",
                media_type="text/plain",
                root=self._run_dir,
            ),
        ]

        try:
            raw, usage = self._llm.call_text(
                model_ref=worker.model_ref,
                system=sys,
                user=user,
                max_output_tokens=self._settings.max_patch_output_tokens,
            )
        except Exception as e:
            err_path = sandbox_dir / "llm_error.txt"
            write_text_atomic(err_path, f"{type(e).__name__}: {e}\n")
            patch_artifacts.append(
                artifact_for(
                    err_path,
                    name="llm_error.txt",
                    media_type="text/plain",
                    root=self._run_dir,
                )
            )
            return ExecutionOutcome(
                status=VerifyStatus.INFRA,
                notes="llm_call_failed",
                patch_artifacts=patch_artifacts,
                sandbox_rel=sandbox_rel,
                patch_kind="none",
            )

        llm_usage = {
            "calls": int(getattr(usage, "calls", 0) or 0),
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }

        write_text_atomic(sandbox_dir / "model_raw.txt", raw)
        patch_artifacts.append(
            artifact_for(
                sandbox_dir / "model_raw.txt",
                name="model_raw.txt",
                media_type="text/plain",
                root=self._run_dir,
            )
        )

        # Parse patch output (diff preferred; file blocks accepted).
        applied_kind = "none"
        touched: list[str] = []
        try:
            if "diff --git " in raw:
                patch_text = extract_git_diff(raw)
                diff_changes = parse_patch_changes(patch_text)
                touched = sorted(
                    {p for ch in diff_changes for p in [ch.old_path, ch.new_path] if p is not None}
                )
                enforce_allowed_paths(paths=touched, allowed=task.allowed_paths)
                patch_path = apply_unified_diff(
                    patch_text=patch_text, cwd=work_dir, patch_path=sandbox_dir / "patch.diff"
                )
                patch_artifacts.append(
                    artifact_for(
                        patch_path, name="patch.diff", media_type="text/x-diff", root=self._run_dir
                    )
                )
                applied_kind = "diff"
            elif "BEGIN_FILE " in raw:
                files = extract_file_blocks(raw)
                touched = sorted(files.keys())
                enforce_allowed_paths(paths=touched, allowed=task.allowed_paths)
                apply_file_blocks(files=files, cwd=work_dir)
                fileblocks_path = sandbox_dir / "patch_files.json"
                write_text_atomic(
                    fileblocks_path, json.dumps(files, ensure_ascii=False, indent=2) + "\n"
                )
                patch_artifacts.append(
                    artifact_for(
                        fileblocks_path,
                        name="patch_files.json",
                        media_type="application/json",
                        root=self._run_dir,
                    )
                )
                patch = build_patch_from_dirs(base_dir=self._workspace_dir, work_dir=work_dir)
                if not patch.patch_text.strip():
                    return ExecutionOutcome(
                        status=VerifyStatus.FAIL,
                        notes="no workspace changes produced",
                        patch_artifacts=patch_artifacts,
                        sandbox_rel=sandbox_rel,
                        patch_kind="files",
                        llm_usage=llm_usage,
                    )
                enforce_allowed_paths(paths=list(patch.touched_paths), allowed=task.allowed_paths)
                patch_path = sandbox_dir / "patch.diff"
                write_text_atomic(patch_path, patch.patch_text)
                patch_artifacts.append(
                    artifact_for(
                        patch_path,
                        name="patch.diff",
                        media_type="text/x-diff",
                        root=self._run_dir,
                    )
                )
                applied_kind = "files"
            else:
                return ExecutionOutcome(
                    status=VerifyStatus.FAIL,
                    notes="no patch found (expected diff --git or BEGIN_FILE blocks)",
                    patch_artifacts=patch_artifacts,
                    sandbox_rel=sandbox_rel,
                    patch_kind="none",
                    llm_usage=llm_usage,
                )
        except Exception as e:
            err_path = sandbox_dir / "patch_apply_error.txt"
            msg = f"{type(e).__name__}: {e}\n"
            if isinstance(e, subprocess.CalledProcessError):
                if e.stderr:
                    msg += f"\n--- stderr ---\n{e.stderr}\n"
                if e.stdout:
                    msg += f"\n--- stdout ---\n{e.stdout}\n"
            write_text_atomic(err_path, msg)
            patch_artifacts.append(
                artifact_for(
                    err_path,
                    name="patch_apply_error.txt",
                    media_type="text/plain",
                    root=self._run_dir,
                )
            )
            return ExecutionOutcome(
                status=VerifyStatus.FAIL,
                notes="patch apply failed",
                patch_artifacts=patch_artifacts,
                sandbox_rel=sandbox_rel,
                patch_kind=applied_kind,
                llm_usage=llm_usage,
            )

        public: list[CommandResult] = []
        hidden: list[CommandResult] = []
        status = VerifyStatus.PASS

        if task.verify_mode == VerifyMode.MANUAL:
            if task.acceptance:
                public = run_commands(
                    commands=list(task.acceptance),
                    cwd=work_dir,
                    scrub_secrets=self._settings.scrub_secrets_in_verification,
                )
            status = VerifyStatus.MANUAL_REVIEW
        else:
            if task.acceptance:
                public = run_commands(
                    commands=list(task.acceptance),
                    cwd=work_dir,
                    scrub_secrets=self._settings.scrub_secrets_in_verification,
                )
                if not all_passed(public):
                    status = (
                        VerifyStatus.TIMEOUT
                        if any(r.timed_out for r in public)
                        else VerifyStatus.FAIL
                    )

            if status == VerifyStatus.PASS and task.hidden_acceptance:
                hidden = run_commands(
                    commands=list(task.hidden_acceptance),
                    cwd=work_dir,
                    scrub_secrets=self._settings.scrub_secrets_in_verification,
                )
                if not all_passed(hidden):
                    status = (
                        VerifyStatus.TIMEOUT
                        if any(r.timed_out for r in hidden)
                        else VerifyStatus.FAIL
                    )

        verify_path = sandbox_dir / "verify.json"
        write_command_results_json(verify_path, public=public, hidden=hidden)
        verification_artifacts = [
            artifact_for(
                verify_path, name="verify.json", media_type="application/json", root=self._run_dir
            )
        ]

        if status == VerifyStatus.PASS and task.verify_mode == VerifyMode.JUDGES:
            judge_spec = task.judges
            refs = (
                list(judge_spec.workers)
                if judge_spec is not None and judge_spec.workers
                else list(self._settings.judge_workers)
            )
            include_self = (
                bool(judge_spec.include_self)
                if judge_spec is not None
                else self._settings.judge_include_self
            )
            min_passes = None if judge_spec is None else judge_spec.min_passes

            judge_workers = resolve_worker_refs(refs, workers=self._workers)
            if include_self:
                judge_workers = [worker] + [
                    w for w in judge_workers if w.worker_id != worker.worker_id
                ]

            if not judge_workers:
                status = VerifyStatus.INFRA
            else:
                required_passes = (
                    int(min_passes) if min_passes is not None else (len(judge_workers) // 2 + 1)
                )
                required_passes = max(1, min(required_passes, len(judge_workers)))

                diff_text = build_unified_diff_summary(
                    workspace_dir=self._workspace_dir,
                    sandbox_dir=work_dir,
                    rel_paths=touched,
                )
                diff_path = sandbox_dir / "diff_for_judges.diff"
                write_text_atomic(diff_path, diff_text)
                verification_artifacts.append(
                    artifact_for(
                        diff_path,
                        name="diff_for_judges.diff",
                        media_type="text/x-diff",
                        root=self._run_dir,
                    )
                )

                try:
                    judge_status, judge_calls = run_judges_with_workers(
                        llm=self._llm,
                        judge_workers=judge_workers,
                        command_specs=self._command_specs,
                        task=task,
                        public=public,
                        hidden=hidden,
                        diff_text=diff_text,
                        required_passes=required_passes,
                        max_output_tokens=self._settings.judge_max_output_tokens,
                        cwd=work_dir,
                    )
                except Exception as e:
                    err_path = sandbox_dir / "judges_error.txt"
                    write_text_atomic(err_path, f"{type(e).__name__}: {e}\n")
                    verification_artifacts.append(
                        artifact_for(
                            err_path,
                            name="judges_error.txt",
                            media_type="text/plain",
                            root=self._run_dir,
                        )
                    )
                    status = VerifyStatus.INFRA
                else:
                    judges_path = sandbox_dir / "judges.json"
                    passes = sum(1 for c in judge_calls if c.decision.verdict == "PASS")
                    payload = {
                        "status": judge_status.value,
                        "judge_workers": [w.worker_id for w in judge_workers],
                        "required_passes": required_passes,
                        "votes_total": len(judge_calls),
                        "passes": passes,
                        "decisions": [
                            {
                                "worker_id": c.worker_id,
                                "worker_type": c.worker_type,
                                **c.decision.model_dump(),
                            }
                            for c in list(judge_calls)
                        ],
                    }
                    write_text_atomic(
                        judges_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
                    )
                    verification_artifacts.append(
                        artifact_for(
                            judges_path,
                            name="judges.json",
                            media_type="application/json",
                            root=self._run_dir,
                        )
                    )
                    for call in judge_calls:
                        safe = "".join(
                            ch if (ch.isalnum() or ch in "-._") else "_" for ch in call.worker_id
                        )
                        raw_path = sandbox_dir / f"judge_{safe}.txt"
                        write_text_atomic(raw_path, call.raw_text)
                        verification_artifacts.append(
                            artifact_for(
                                raw_path,
                                name=raw_path.name,
                                media_type="text/plain",
                                root=self._run_dir,
                            )
                        )
                    status = judge_status

        return ExecutionOutcome(
            status=status,
            notes=f"patch_applied={applied_kind}",
            patch_artifacts=patch_artifacts,
            verification_artifacts=verification_artifacts,
            sandbox_rel=sandbox_rel,
            patch_kind=applied_kind,
            llm_usage=llm_usage,
        )

    def integrate(
        self,
        *,
        worker: WorkerRuntime,
        task: TaskSpec,
        bid: Bid,
        round_id: int,
        outcome: ExecutionOutcome,
    ) -> ExecutionOutcome:
        _ = worker, bid, round_id
        if outcome.status != VerifyStatus.PASS:
            return outcome

        by_name = {a.name: a for a in list(outcome.patch_artifacts)}
        a = by_name.get("patch.diff")
        if a is None or not a.path:
            return ExecutionOutcome(
                status=VerifyStatus.INFRA,
                notes="missing_patch_diff_for_integration",
                patch_artifacts=list(outcome.patch_artifacts),
                verification_artifacts=list(outcome.verification_artifacts),
                sandbox_rel=outcome.sandbox_rel,
                patch_kind=outcome.patch_kind,
                llm_usage=outcome.llm_usage,
            )

        patch_path = self._run_dir / a.path
        sandbox_dir = self._run_dir / (outcome.sandbox_rel or "")

        try:
            enforce_allowed_paths(
                paths=[
                    p
                    for ch in parse_patch_changes(patch_path.read_text(encoding="utf-8"))
                    for p in (ch.old_path, ch.new_path)
                    if p is not None
                ],
                allowed=task.allowed_paths,
            )
            apply_unified_diff_path(patch_path=patch_path, cwd=self._workspace_dir)
        except Exception as e:
            err_path = sandbox_dir / "integrate_error.txt"
            write_text_atomic(err_path, f"{type(e).__name__}: {e}\n")
            verification_artifacts = list(outcome.verification_artifacts)
            verification_artifacts.append(
                artifact_for(
                    err_path,
                    name="integrate_error.txt",
                    media_type="text/plain",
                    root=self._run_dir,
                )
            )
            return ExecutionOutcome(
                status=VerifyStatus.INFRA,
                notes="workspace_apply_failed",
                patch_artifacts=list(outcome.patch_artifacts),
                verification_artifacts=verification_artifacts,
                sandbox_rel=outcome.sandbox_rel,
                patch_kind=outcome.patch_kind,
                llm_usage=outcome.llm_usage,
            )

        return outcome
