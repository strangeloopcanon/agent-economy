from __future__ import annotations

import pytest

from agent_economy.config import InstitutionSettings
from agent_economy.main import _llm_router_for_workers
from agent_economy.schemas import WorkerRuntime, WorkerType


def test_llm_router_for_workers_supports_ollama_without_openai_key() -> None:
    settings = InstitutionSettings(
        openai_api_key=None,
        openai_base_url=None,
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
