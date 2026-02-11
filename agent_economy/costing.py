from __future__ import annotations

import json
import os
from dataclasses import dataclass

from agent_economy.model_refs import split_provider_model


@dataclass(frozen=True)
class Price:
    # Cost units per 1k tokens. These are "market units" (not USD) unless you set your own.
    input_per_1k: float
    output_per_1k: float


def _canonical_model_ref(model_ref: str) -> str:
    provider, model = split_provider_model(model_ref)
    return f"{provider}:{model}"


def load_pricing_from_env() -> dict[str, Price]:
    raw = (os.getenv("AE_PRICING_JSON") or os.getenv("INST_PRICING_JSON") or "").strip()
    if not raw:
        return {}

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("AE_PRICING_JSON/INST_PRICING_JSON must be a JSON object")

    out: dict[str, Price] = {}
    for k, v in parsed.items():
        if not isinstance(k, str):
            continue
        key = _canonical_model_ref(k)
        if isinstance(v, (int, float)):
            out[key] = Price(input_per_1k=float(v), output_per_1k=float(v))
            continue
        if isinstance(v, dict):
            inp = v.get("input_per_1k")
            outp = v.get("output_per_1k")
            if isinstance(inp, (int, float)) and isinstance(outp, (int, float)):
                out[key] = Price(input_per_1k=float(inp), output_per_1k=float(outp))
                continue
        raise ValueError(
            "AE_PRICING_JSON/INST_PRICING_JSON values must be a number or {input_per_1k, output_per_1k}"
        )
    return out


def default_price_for(model_ref: str) -> Price:
    _provider, model = split_provider_model(model_ref)
    m = model.lower()

    # Heuristic defaults (unitless). Override with AE_PRICING_JSON for your environment.
    if "mini" in m:
        return Price(input_per_1k=0.2, output_per_1k=0.6)
    if "codex" in m:
        return Price(input_per_1k=0.4, output_per_1k=1.2)
    if "pro" in m or "5.2" in m:
        return Price(input_per_1k=0.6, output_per_1k=1.8)
    return Price(input_per_1k=0.5, output_per_1k=1.5)


def price_for_model(*, pricing: dict[str, Price], model_ref: str) -> Price:
    key = _canonical_model_ref(model_ref)
    return pricing.get(key) or default_price_for(key)


def estimate_cost_units(*, input_tokens: float, output_tokens: float, price: Price) -> float:
    return (float(input_tokens) / 1000.0) * price.input_per_1k + (
        float(output_tokens) / 1000.0
    ) * price.output_per_1k
