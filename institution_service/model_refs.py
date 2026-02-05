from __future__ import annotations


def split_provider_model(model_ref: str) -> tuple[str, str]:
    if ":" not in model_ref:
        return "openai", model_ref
    provider, model = model_ref.split(":", 1)
    provider = provider.strip().lower()
    if provider in {"local", "ollama"}:
        provider = "ollama"
    if provider in {"gemini"}:
        provider = "google"
    if provider in {"claude"}:
        provider = "anthropic"
    return provider, model.strip()
