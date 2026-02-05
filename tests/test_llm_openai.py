from __future__ import annotations

import pytest

from institution_service.llm_openai import (
    OpenAIJSONClient,
    _resolve_max_retries,
)


class _FakeUsage:
    def __init__(self, *, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, *, output_text: str, input_tokens: int, output_tokens: int) -> None:
        self.output_text = output_text
        self.usage = _FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens)


class _FakeResponses:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs) -> object:
        _ = kwargs
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.responses = _FakeResponses(outcomes)


def _make_client(*, outcomes: list[object]) -> tuple[OpenAIJSONClient, _FakeResponses]:
    client = OpenAIJSONClient(api_key="test-key", base_url=None)
    fake = _FakeClient(outcomes)
    client._client = fake  # type: ignore[assignment]
    return client, fake.responses


def test_resolve_max_retries_rejects_invalid_and_too_high_values() -> None:
    with pytest.raises(ValueError, match="must be an int"):
        _resolve_max_retries(requested=3, env_retries_raw="abc")
    with pytest.raises(ValueError, match=r"<= 3"):
        _resolve_max_retries(requested=3, env_retries_raw="4")
    with pytest.raises(ValueError, match=r"<= 3"):
        _resolve_max_retries(requested=9, env_retries_raw=None)


def test_call_text_retries_only_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INST_OPENAI_MAX_RETRIES", "3")
    monkeypatch.setattr(
        "institution_service.llm_openai._is_transient_error",
        lambda err: isinstance(err, TimeoutError),
    )
    client, fake_responses = _make_client(
        outcomes=[
            TimeoutError("t1"),
            TimeoutError("t2"),
            _FakeResponse(output_text="ok", input_tokens=11, output_tokens=7),
        ]
    )

    text, usage = client.call_text(model="x", system="s", user="u")

    assert text == "ok"
    assert usage.calls == 1
    assert usage.input_tokens == 11
    assert usage.output_tokens == 7
    assert fake_responses.calls == 3


def test_call_text_fails_fast_on_deterministic_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INST_OPENAI_MAX_RETRIES", raising=False)
    monkeypatch.setattr("institution_service.llm_openai._is_transient_error", lambda err: False)
    client, fake_responses = _make_client(outcomes=[ValueError("schema fail")])

    with pytest.raises(ValueError, match="schema fail"):
        client.call_text(model="x", system="s", user="u", max_retries=3)
    assert fake_responses.calls == 1
