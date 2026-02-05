from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from institution_service.llm_ollama import OllamaJSONClient
from institution_service.llm_openai import OpenAIJSONClient, Usage
from institution_service.model_refs import split_provider_model

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass(frozen=True)
class LLMRouter:
    openai: OpenAIJSONClient | None = None
    ollama: OllamaJSONClient | None = None

    def call_text(
        self,
        *,
        model_ref: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_output_tokens: int = 3000,
        reasoning_effort: str | None = None,
        text_verbosity: str | None = None,
        max_retries: int = 3,
    ) -> tuple[str, Usage]:
        provider, model = split_provider_model(model_ref)
        if provider == "openai":
            if self.openai is None:
                raise ValueError("missing OpenAI client (set OPENAI_API_KEY)")
            return self.openai.call_text(
                model=model,
                system=system,
                user=user,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
                max_retries=max_retries,
            )
        if provider == "ollama":
            if self.ollama is None:
                raise ValueError("missing Ollama client (set OLLAMA_BASE_URL)")
            return self.ollama.call_text(
                model=model,
                system=system,
                user=user,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
                max_retries=max_retries,
            )

        raise ValueError(
            f"unsupported provider {provider!r} for model_ref={model_ref!r} "
            "(use an external worker, or route via an OpenAI-compatible gateway)"
        )

    def call_json(
        self,
        *,
        model_ref: str,
        system: str,
        user: str,
        schema: type[TModel],
        temperature: float = 0.0,
        max_output_tokens: int = 1500,
        reasoning_effort: str | None = None,
        text_verbosity: str | None = None,
        max_retries: int = 3,
    ) -> tuple[TModel, Usage, str]:
        provider, model = split_provider_model(model_ref)
        if provider == "openai":
            if self.openai is None:
                raise ValueError("missing OpenAI client (set OPENAI_API_KEY)")
            resp, usage, raw = self.openai.call_json(
                model=model,
                system=system,
                user=user,
                schema=schema,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
                max_retries=max_retries,
            )
            return resp, usage, raw
        if provider == "ollama":
            if self.ollama is None:
                raise ValueError("missing Ollama client (set OLLAMA_BASE_URL)")
            resp, usage, raw = self.ollama.call_json(
                model=model,
                system=system,
                user=user,
                schema=schema,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
                max_retries=max_retries,
            )
            return resp, usage, raw

        raise ValueError(
            f"unsupported provider {provider!r} for model_ref={model_ref!r} "
            "(use an external worker, or route via an OpenAI-compatible gateway)"
        )
