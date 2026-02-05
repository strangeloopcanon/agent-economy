from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from institution_service.json_extract import extract_json_object

_MAX_OUTER_RETRIES = 3
_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Usage:
    calls: int
    input_tokens: int
    output_tokens: int


def _resolve_max_retries(*, requested: int, env_retries_raw: str | None) -> int:
    max_retries = int(requested)
    raw = str(env_retries_raw or "").strip()
    if raw:
        try:
            max_retries = int(raw)
        except Exception as e:
            raise ValueError("INST_OPENAI_MAX_RETRIES must be an int") from e
    if max_retries <= 0:
        raise ValueError("INST_OPENAI_MAX_RETRIES/max_retries must be > 0")
    if max_retries > _MAX_OUTER_RETRIES:
        raise ValueError(f"INST_OPENAI_MAX_RETRIES/max_retries must be <= {_MAX_OUTER_RETRIES}")
    return max_retries


def _status_code_from_error(err: Exception) -> int | None:
    raw = getattr(err, "status_code", None)
    if raw is None:
        response = getattr(err, "response", None)
        raw = getattr(response, "status_code", None)
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


def _is_transient_error(err: Exception) -> bool:
    if isinstance(
        err,
        (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, TimeoutError),
    ):
        return True
    if isinstance(err, APIStatusError):
        status = _status_code_from_error(err)
        return status in _TRANSIENT_STATUS_CODES
    status = _status_code_from_error(err)
    return status in _TRANSIENT_STATUS_CODES if status is not None else False


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


class OpenAIJSONClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
    ) -> None:
        # Explicitly disable timeouts: long-running model calls are allowed.
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=None)
        self._no_temperature: set[str] = set()
        self._no_reasoning: set[str] = set()
        self._no_text_verbosity: set[str] = set()

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
        max_retries = _resolve_max_retries(
            requested=max_retries,
            env_retries_raw=os.getenv("INST_OPENAI_MAX_RETRIES"),
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
                    reasoning_effort=reasoning_effort,
                    text_verbosity=text_verbosity,
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
                if not _is_transient_error(e):
                    raise
                last_err = e
                if attempt == max_retries - 1:
                    break
                time.sleep(0.5 * (2**attempt) + random.random() * 0.2)

        raise RuntimeError(
            f"OpenAI transient request failed after {max_retries} attempts: {last_err}"
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
        reasoning_effort: str | None,
        text_verbosity: str | None,
    ) -> tuple[str, Usage]:
        params: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            "max_output_tokens": max_output_tokens,
        }
        if model not in self._no_temperature:
            params["temperature"] = temperature
        if reasoning_effort and model not in self._no_reasoning:
            params["reasoning"] = {"effort": reasoning_effort}
        if text_verbosity and model not in self._no_text_verbosity:
            params["text"] = {"verbosity": text_verbosity}

        resp: Any | None = None
        for _attempt in range(3):
            try:
                resp = self._client.responses.create(**params)
                break
            except BadRequestError as e:
                msg = str(e)
                if "Unsupported parameter" in msg and "temperature" in msg:
                    self._no_temperature.add(model)
                    params.pop("temperature", None)
                    continue
                if "Unsupported parameter" in msg and (
                    "reasoning" in msg or "reasoning.effort" in msg
                ):
                    self._no_reasoning.add(model)
                    params.pop("reasoning", None)
                    continue
                if (
                    "Unsupported parameter" in msg and ("text" in msg or "text.verbosity" in msg)
                ) or ("text.verbosity" in msg):
                    self._no_text_verbosity.add(model)
                    params.pop("text", None)
                    continue
                raise

        if resp is None:  # pragma: no cover
            raise RuntimeError("OpenAI request failed after parameter fallbacks")

        text = getattr(resp, "output_text", None) or ""
        return text, _usage_from_response(resp)
