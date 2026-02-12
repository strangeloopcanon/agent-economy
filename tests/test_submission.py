from __future__ import annotations

import json

from agent_economy.schemas import SubmissionKind
from agent_economy.submission import (
    normalize_submission_output,
    persist_submission,
    submission_workspace_relpath,
)


def test_normalize_submission_output_text_trims_and_newline() -> None:
    out = normalize_submission_output(raw_output="  hello world  ", kind=SubmissionKind.TEXT)
    assert out == "hello world\n"


def test_normalize_submission_output_json_accepts_wrapped_object() -> None:
    out = normalize_submission_output(
        raw_output='Result:\n{"ok": true, "count": 2}\n',
        kind=SubmissionKind.JSON,
    )
    assert json.loads(out) == {"ok": True, "count": 2}


def test_persist_submission_writes_artifact_and_workspace_copy(tmp_path) -> None:
    sandbox_dir = tmp_path / "sandbox"
    work_dir = tmp_path / "work"
    sandbox_dir.mkdir()
    work_dir.mkdir()

    artifact_path, workspace_path = persist_submission(
        sandbox_dir=sandbox_dir,
        work_dir=work_dir,
        normalized_output="answer\n",
        kind=SubmissionKind.TEXT,
    )
    assert artifact_path.read_text(encoding="utf-8") == "answer\n"
    expected_workspace = work_dir / submission_workspace_relpath(kind=SubmissionKind.TEXT)
    assert workspace_path == expected_workspace
    assert workspace_path.read_text(encoding="utf-8") == "answer\n"
