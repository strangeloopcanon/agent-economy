from __future__ import annotations

from agent_economy.llm_openai import Usage
from agent_economy.openai_executor import OpenAIExecutor
from agent_economy.schemas import (
    Bid,
    CommandSpec,
    SubmissionKind,
    TaskSpec,
    VerifyStatus,
    WorkerRuntime,
)


class StubLLMRouter:
    def call_text(
        self,
        *,
        model_ref: str,
        system: str,
        user: str,
        max_output_tokens: int,
    ) -> tuple[str, Usage]:
        _ = model_ref, system, user, max_output_tokens
        return "This is the answer.", Usage(calls=1, input_tokens=10, output_tokens=5)


def test_openai_executor_text_submission_passes_without_patch_diff(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "README.md").write_text("seed\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    worker = WorkerRuntime(worker_id="w1", model_ref="gpt-5-mini")
    task = TaskSpec(
        id="T1",
        title="Answer a question",
        bounty=10,
        submission_kind=SubmissionKind.TEXT,
        verify_mode="commands",
        acceptance=[CommandSpec(cmd="test -f .agent_economy/submission.txt")],
    )
    bid = Bid(task_id="T1", ask=5, self_assessed_p_success=0.8, eta_minutes=10)

    executor = OpenAIExecutor(
        llm=StubLLMRouter(),  # type: ignore[arg-type]
        workspace_dir=workspace_dir,
        run_dir=run_dir,
        workers=[worker],
    )

    outcome = executor.execute(worker=worker, task=task, bid=bid, round_id=0, discussion_history=[])
    assert outcome.status == VerifyStatus.PASS
    assert outcome.submission_kind == SubmissionKind.TEXT
    assert any(a.name == "submission.txt" for a in outcome.submission_artifacts)

    integrated = executor.integrate(worker=worker, task=task, bid=bid, round_id=0, outcome=outcome)
    assert integrated.status == VerifyStatus.PASS
