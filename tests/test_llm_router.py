from __future__ import annotations

from pydantic import BaseModel

from agent_economy.llm_openai import Usage
from agent_economy.llm_router import LLMRouter


class _DummySchema(BaseModel):
    ok: bool


class _FakeClient:
    def __init__(self) -> None:
        self.last_model: str | None = None

    def call_text(self, **kwargs):
        self.last_model = str(kwargs["model"])
        return "pong", Usage(calls=1, input_tokens=2, output_tokens=3)

    def call_json(self, **kwargs):
        self.last_model = str(kwargs["model"])
        schema = kwargs["schema"]
        return (
            schema.model_validate({"ok": True}),
            Usage(calls=1, input_tokens=1, output_tokens=1),
            '{"ok":true}',
        )


def test_router_routes_ollama_calls() -> None:
    ollama = _FakeClient()
    router = LLMRouter(ollama=ollama)

    text, usage = router.call_text(model_ref="ollama:qwen3:8b", system="s", user="u")
    assert text == "pong"
    assert usage.calls == 1
    assert ollama.last_model == "qwen3:8b"

    parsed, _usage, raw = router.call_json(
        model_ref="local:qwen3:3b",
        system="s",
        user="u",
        schema=_DummySchema,
    )
    assert parsed.ok is True
    assert raw == '{"ok":true}'
    assert ollama.last_model == "qwen3:3b"


def test_router_routes_anthropic_and_google_calls() -> None:
    anthropic = _FakeClient()
    google = _FakeClient()
    router = LLMRouter(anthropic=anthropic, google=google)

    text, usage = router.call_text(
        model_ref="anthropic:claude-sonnet-4-5",
        system="s",
        user="u",
    )
    assert text == "pong"
    assert usage.calls == 1
    assert anthropic.last_model == "claude-sonnet-4-5"

    parsed, _usage, raw = router.call_json(
        model_ref="gemini:gemini-2.5-pro",
        system="s",
        user="u",
        schema=_DummySchema,
    )
    assert parsed.ok is True
    assert raw == '{"ok":true}'
    assert google.last_model == "gemini-2.5-pro"


def test_router_requires_ollama_client_when_provider_is_ollama() -> None:
    router = LLMRouter()
    try:
        router.call_text(model_ref="ollama:qwen3:8b", system="s", user="u")
    except ValueError as e:
        assert "missing Ollama client" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_router_requires_anthropic_client_when_provider_is_anthropic() -> None:
    router = LLMRouter()
    try:
        router.call_text(model_ref="claude:claude-sonnet-4-5", system="s", user="u")
    except ValueError as e:
        assert "missing Anthropic client" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_router_requires_google_client_when_provider_is_google() -> None:
    router = LLMRouter()
    try:
        router.call_text(model_ref="google:gemini-2.5-pro", system="s", user="u")
    except ValueError as e:
        assert "missing Google client" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
