from __future__ import annotations


_IGNORED_PARTS = {
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".venv",
    ".env",
    ".beads",
    ".codex",
    "runs",
    "dist",
    "build",
    ".ruff_cache",
    ".DS_Store",
}


def is_ignored_workspace_part(part: str) -> bool:
    return part in _IGNORED_PARTS or part.endswith(".egg-info")


def copytree_ignore(_: str, names: list[str]) -> set[str]:
    return {n for n in names if is_ignored_workspace_part(n)}
