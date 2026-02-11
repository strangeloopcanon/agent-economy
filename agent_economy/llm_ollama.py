from __future__ import annotations

import json
import os
import random
import time
from typing import Any
from urllib import error, request

from pydantic import BaseModel

from agent_economy.json_extract import extract_json_object
from agent_economy.llm_openai import Usage, _resolve_max_retries

_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _is_transient_ollama_error(err: Exception) -> bool:
    if isinstance(err, TimeoutError):
        return True
    if isinstance(err, error.URLError):
        return True
    if isinstance(err, error.HTTPError):
        try:
            return int(err.code) in _TRANSIENT_STATUS_CODES
        except Exception:
            return False
    return False


class OllamaJSONClient:
    def __init__(self, *, base_url: str) -> None:
        self._base_url = str(base_url).rstrip("/")

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
                os.getenv("AE_OLLAMA_MAX_RETRIES") or os.getenv("INST_OLLAMA_MAX_RETRIES")
            ),
        )

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                return self._call_text_once(
                    model=model,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as e:
                if not _is_transient_ollama_error(e):
                    raise
                last_err = e
                if attempt == max_retries - 1:
                    break
                time.sleep(0.5 * (2**attempt) + random.random() * 0.2)

        raise RuntimeError(
            f"Ollama transient request failed after {max_retries} attempts: {last_err}"
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
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_output_tokens),
            },
        }
        req = request.Request(
            url=f"{self._base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=300) as resp:
            body_bytes = resp.read()
        body = body_bytes.decode("utf-8", errors="replace")
        data = json.loads(body)

        text = str(((data.get("message") or {}).get("content")) or "")
        prompt_eval = int(data.get("prompt_eval_count") or 0)
        eval_count = int(data.get("eval_count") or 0)
        usage = Usage(calls=1, input_tokens=prompt_eval, output_tokens=eval_count)
        return text, usage
