from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def repo_root() -> Path:
    # Project root is the directory that contains the `agent_economy/` package.
    # (This repo was previously nested in a larger workspace; avoid assuming that layout.)
    return Path(__file__).resolve().parents[1]


def load_env() -> None:
    # Prefer a project-local `.env` (so installed CLIs behave consistently even when
    # invoked from another working directory). Fall back to searching from CWD.
    root_env = repo_root() / ".env"
    env_path = str(root_env) if root_env.exists() else (find_dotenv(usecwd=True) or str(root_env))
    load_dotenv(env_path)


@dataclass(frozen=True)
class InstitutionSettings:
    openai_api_key: str | None
    openai_base_url: str | None
    anthropic_api_key: str | None
    anthropic_base_url: str | None
    google_api_key: str | None
    ollama_base_url: str


def load_settings() -> InstitutionSettings:
    load_env()
    return InstitutionSettings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None,
        ollama_base_url=(os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip(),
    )
