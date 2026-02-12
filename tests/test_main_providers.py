from __future__ import annotations

import pytest

from agent_economy.config import InstitutionSettings
from agent_economy.main import _llm_router_for_workers
from agent_economy.schemas import WorkerRuntime, WorkerType


def test_llm_router_for_workers_supports_ollama_without_openai_key() -> None:
    settings = InstitutionSettings(
        openai_api_key=None,
        openai_base_url=None,
        anthropic_api_key=None,
        anthropic_base_url=None,
        google_api_key=None,
        ollama_base_url="http://127.0.0.1:11434",
    )
    workers = [
        WorkerRuntime(
            worker_id="qwen-8b",
            worker_type=WorkerType.MODEL_AGENT,
            model_ref="ollama:qwen3:8b",
        )
    ]

    llm = _llm_router_for_workers(settings=settings, workers=workers)
    assert llm.ollama is not None
    assert llm.openai is None


def test_llm_router_for_workers_requires_openai_key_when_openai_models_present() -> None:
    settings = InstitutionSettings(
        openai_api_key=None,
        openai_base_url=None,
        anthropic_api_key=None,
        anthropic_base_url=None,
        google_api_key=None,
        ollama_base_url="http://127.0.0.1:11434",
    )
    workers = [
        WorkerRuntime(
            worker_id="openai-worker",
            worker_type=WorkerType.MODEL_AGENT,
            model_ref="openai:gpt-5-mini",
        )
    ]

    with pytest.raises(SystemExit, match="OPENAI_API_KEY is required"):
        _llm_router_for_workers(settings=settings, workers=workers)


def test_llm_router_for_workers_requires_anthropic_key_when_anthropic_models_present() -> None:
    settings = InstitutionSettings(
        openai_api_key=None,
        openai_base_url=None,
        anthropic_api_key=None,
        anthropic_base_url=None,
        google_api_key=None,
        ollama_base_url="http://127.0.0.1:11434",
    )
    workers = [
        WorkerRuntime(
            worker_id="anthropic-worker",
            worker_type=WorkerType.MODEL_AGENT,
            model_ref="anthropic:claude-sonnet-4-5",
        )
    ]

    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is required"):
        _llm_router_for_workers(settings=settings, workers=workers)


def test_llm_router_for_workers_requires_google_key_when_google_models_present() -> None:
    settings = InstitutionSettings(
        openai_api_key=None,
        openai_base_url=None,
        anthropic_api_key=None,
        anthropic_base_url=None,
        google_api_key=None,
        ollama_base_url="http://127.0.0.1:11434",
    )
    workers = [
        WorkerRuntime(
            worker_id="google-worker",
            worker_type=WorkerType.MODEL_AGENT,
            model_ref="google:gemini-2.5-pro",
        )
    ]

    with pytest.raises(SystemExit, match="GOOGLE_API_KEY or GEMINI_API_KEY is required"):
        _llm_router_for_workers(settings=settings, workers=workers)
