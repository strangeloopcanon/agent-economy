from __future__ import annotations

import os
import random
import time
from typing import Any

from anthropic import Anthropic
from pydantic import BaseModel

from agent_economy.json_extract import extract_json_object
from agent_economy.llm_openai import Usage, _resolve_max_retries

_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _status_code_from_error(err: Exception) -> int | None:
    raw = getattr(err, "status_code", None)
    if raw is None:
        response = getattr(err, "response", None)
        raw = getattr(response, "status_code", None)
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


def _is_transient_anthropic_error(err: Exception) -> bool:
    if isinstance(err, TimeoutError):
        return True
    status = _status_code_from_error(err)
    if status is not None:
        return status in _TRANSIENT_STATUS_CODES
    name = type(err).__name__.lower()
    if "ratelimit" in name or "timeout" in name or "connection" in name:
        return True
    return False


def _usage_from_response(resp: Any) -> Usage:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return Usage(calls=1, input_tokens=0, output_tokens=0)

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None and isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

    return Usage(
        calls=1,
        input_tokens=int(input_tokens or 0),
        output_tokens=int(output_tokens or 0),
    )


class AnthropicJSONClient:
    def __init__(self, *, api_key: str, base_url: str | None) -> None:
        # Explicitly disable timeouts: long-running model calls are allowed.
        self._client = Anthropic(api_key=api_key, base_url=base_url, timeout=None)

    def call_text(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_output_tokens: int = 3000,
        reasoning_effort: str | None = None,
        text_verbosity: str | None = None,
        max_retries: int = 3,
    ) -> tuple[str, Usage]:
        _ = reasoning_effort, text_verbosity
        max_retries = _resolve_max_retries(
            requested=max_retries,
            env_retries_raw=(
                os.getenv("AE_ANTHROPIC_MAX_RETRIES") or os.getenv("INST_ANTHROPIC_MAX_RETRIES")
            ),
        )

        last_err: Exception | None = None
        total_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for attempt in range(max_retries):
            try:
                text, usage = self._call_text_once(
                    model=model,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                total_calls += usage.calls
                total_input_tokens += usage.input_tokens
                total_output_tokens += usage.output_tokens
                return text, Usage(
                    calls=total_calls,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )
            except Exception as e:
                if not _is_transient_anthropic_error(e):
                    raise
                last_err = e
                if attempt == max_retries - 1:
                    break
                time.sleep(0.5 * (2**attempt) + random.random() * 0.2)

        raise RuntimeError(
            f"Anthropic transient request failed after {max_retries} attempts: {last_err}"
        ) from last_err

    def call_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
        max_output_tokens: int = 1500,
        reasoning_effort: str | None = None,
        text_verbosity: str | None = None,
        max_retries: int = 3,
    ) -> tuple[BaseModel, Usage, str]:
        text, usage = self.call_text(
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
            max_retries=max_retries,
        )
        parsed = extract_json_object(text)
        return schema.model_validate(parsed), usage, text

    def _call_text_once(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_output_tokens: int,
    ) -> tuple[str, Usage]:
        resp = self._client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=int(max_output_tokens),
            temperature=float(temperature),
        )
        text_parts: list[str] = []
        for block in list(getattr(resp, "content", []) or []):
            btype = getattr(block, "type", None)
            if btype is None and isinstance(block, dict):
                btype = block.get("type")
            if btype != "text":
                continue
            value = getattr(block, "text", None)
            if value is None and isinstance(block, dict):
                value = block.get("text")
            if value:
                text_parts.append(str(value))
        text = "".join(text_parts).strip()
        if not text:
            text = str(getattr(resp, "content", "") or "")
        return text, _usage_from_response(resp)
