from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_economy.json_extract import extract_json_object
from agent_economy.sandbox import write_text_atomic
from agent_economy.schemas import SubmissionKind


SUBMISSION_DIR = ".agent_economy"


def submission_workspace_relpath(*, kind: SubmissionKind) -> str:
    if kind == SubmissionKind.TEXT:
        return f"{SUBMISSION_DIR}/submission.txt"
    if kind == SubmissionKind.JSON:
        return f"{SUBMISSION_DIR}/submission.json"
    raise ValueError(f"unsupported submission kind for workspace file: {kind.value}")


def submission_artifact_name(*, kind: SubmissionKind) -> str:
    if kind == SubmissionKind.TEXT:
        return "submission.txt"
    if kind == SubmissionKind.JSON:
        return "submission.json"
    raise ValueError(f"unsupported submission kind for artifact: {kind.value}")


def submission_media_type(*, kind: SubmissionKind) -> str:
    if kind == SubmissionKind.TEXT:
        return "text/plain"
    if kind == SubmissionKind.JSON:
        return "application/json"
    raise ValueError(f"unsupported submission kind for media type: {kind.value}")


def normalize_submission_output(*, raw_output: str, kind: SubmissionKind) -> str:
    if kind == SubmissionKind.TEXT:
        text = raw_output.strip()
        if not text:
            raise ValueError("empty text submission")
        return text + "\n"

    if kind == SubmissionKind.JSON:
        text = raw_output.strip()
        if not text:
            raise ValueError("empty json submission")
        parsed: Any
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = extract_json_object(raw_output)
        return json.dumps(parsed, ensure_ascii=False, indent=2) + "\n"

    raise ValueError(f"unsupported submission kind for normalization: {kind.value}")


def persist_submission(
    *,
    sandbox_dir: Path,
    work_dir: Path,
    normalized_output: str,
    kind: SubmissionKind,
) -> tuple[Path, Path]:
    name = submission_artifact_name(kind=kind)
    artifact_path = sandbox_dir / name
    write_text_atomic(artifact_path, normalized_output)

    workspace_rel = submission_workspace_relpath(kind=kind)
    workspace_path = work_dir / workspace_rel
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(workspace_path, normalized_output)
    return artifact_path, workspace_path
