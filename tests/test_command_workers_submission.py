from __future__ import annotations

from agent_economy.command_workers import CommandExecutor
from agent_economy.schemas import (
    Bid,
    CommandSpec,
    SubmissionKind,
    TaskSpec,
    VerifyStatus,
    WorkerRuntime,
    WorkerType,
)
from agent_economy.worker_specs import CommandWorkerSpec


def test_command_executor_text_submission_passes_without_workspace_patch(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "seed.txt").write_text("seed\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    worker = WorkerRuntime(worker_id="ext", worker_type=WorkerType.EXTERNAL_WORKER)
    spec = CommandWorkerSpec(
        worker_id="ext",
        exec_cmd="printf 'External worker answer'",
        fixed_bid={"ask": 1, "p_success": 0.8, "eta_minutes": 5},
    )
    task = TaskSpec(
        id="T1",
        title="Answer from external worker",
        bounty=5,
        submission_kind=SubmissionKind.TEXT,
        verify_mode="commands",
        acceptance=[CommandSpec(cmd="test -f .agent_economy/submission.txt")],
    )
    bid = Bid(task_id="T1", ask=1, self_assessed_p_success=0.8, eta_minutes=5)

    executor = CommandExecutor(
        workspace_dir=workspace_dir,
        run_dir=run_dir,
        workers=[worker],
        specs={"ext": spec},
    )
    outcome = executor.execute(worker=worker, task=task, bid=bid, round_id=0)
    assert outcome.status == VerifyStatus.PASS
    assert outcome.submission_kind == SubmissionKind.TEXT
    assert any(a.name == "submission.txt" for a in outcome.submission_artifacts)

    integrated = executor.integrate(worker=worker, task=task, bid=bid, round_id=0, outcome=outcome)
    assert integrated.status == VerifyStatus.PASS
