from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from institution_service.config import repo_root
from institution_service.llm_openai import Usage
from institution_service.model_refs import split_provider_model
from institution_service.sandbox import write_text_atomic
from institution_service.schemas import EventType, LedgerEvent
from institution_service.schemas import WorkerRuntime


def canonical_model_ref(model_ref: str | None) -> str | None:
    if not model_ref:
        return None
    provider, model = split_provider_model(model_ref)
    return f"{provider}:{model}"


class PersistedWorkerStats(BaseModel):
    worker_id: str
    model_ref: str | None = None
    reputation: float = Field(default=1.0, ge=0.0, le=2.0)

    # Rough token-usage EMA for patch generation calls (for expected_cost estimation).
    patch_ema_input_tokens: float = Field(default=0.0, ge=0.0)
    patch_ema_output_tokens: float = Field(default=0.0, ge=0.0)


class PersistedWorkerState(BaseModel):
    schema_version: int = Field(default=1, ge=1)
    workers: dict[str, PersistedWorkerStats] = Field(default_factory=dict)


@dataclass(frozen=True)
class WorkerUsageSample:
    worker_id: str
    model_ref: str | None
    usage: Usage


def extract_patch_usage_samples(*, events: list[LedgerEvent]) -> list[WorkerUsageSample]:
    samples: list[WorkerUsageSample] = []
    for e in events:
        if e.type != EventType.PATCH_SUBMITTED:
            continue
        worker_id = str((e.payload or {}).get("worker_id") or "").strip()
        if not worker_id:
            continue
        raw = (e.payload or {}).get("llm_usage")
        if not isinstance(raw, dict):
            continue
        try:
            usage = Usage(
                calls=int(raw.get("calls") or 0),
                input_tokens=int(raw.get("input_tokens") or 0),
                output_tokens=int(raw.get("output_tokens") or 0),
            )
        except Exception:
            continue
        if usage.calls <= 0:
            continue
        samples.append(
            WorkerUsageSample(
                worker_id=worker_id,
                model_ref=str((e.payload or {}).get("model_ref") or "").strip() or None,
                usage=usage,
            )
        )
    return samples


def default_state_path() -> Path:
    raw = (os.getenv("INST_WORKER_STATE_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return repo_root() / ".beads" / "institution_service_workers.json"


def load_state(path: Path) -> PersistedWorkerState:
    if os.getenv("INST_WORKER_STATE_ISOLATION", "").lower() in ("1", "true", "yes"):
        return PersistedWorkerState()

    if not path.exists():
        return PersistedWorkerState()
    try:
        return PersistedWorkerState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return PersistedWorkerState()


def save_state(path: Path, state: PersistedWorkerState) -> None:
    if os.getenv("INST_WORKER_STATE_ISOLATION", "").lower() in ("1", "true", "yes"):
        return

    write_text_atomic(path, json.dumps(state.model_dump(mode="json"), indent=2) + "\n")


def apply_state_to_workers(
    *, state: PersistedWorkerState, workers: list[WorkerRuntime]
) -> list[WorkerRuntime]:
    for w in workers:
        stats = state.workers.get(str(getattr(w, "worker_id", "")))
        if stats is None:
            continue
        # Only apply if the model_ref matches (so rep doesn't silently transfer across models).
        if stats.model_ref and canonical_model_ref(stats.model_ref) != canonical_model_ref(
            getattr(w, "model_ref", None)
        ):
            continue
        if getattr(w, "reputation", None) is not None:
            w.reputation = float(stats.reputation)
    return workers


def _ema(old: float, new: float, *, alpha: float) -> float:
    if old <= 0:
        return float(new)
    return float(old) * (1.0 - alpha) + float(new) * alpha


def update_state_from_run(
    *,
    state: PersistedWorkerState,
    run_workers: dict[str, WorkerRuntime],
    patch_usages: list[WorkerUsageSample],
    alpha: float = 0.2,
) -> PersistedWorkerState:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    for worker_id, worker in run_workers.items():
        stats = state.workers.get(worker_id) or PersistedWorkerStats(worker_id=worker_id)
        stats.model_ref = getattr(worker, "model_ref", None) or stats.model_ref
        stats.reputation = float(getattr(worker, "reputation", 1.0))
        state.workers[worker_id] = stats

    for sample in patch_usages:
        stats = state.workers.get(sample.worker_id) or PersistedWorkerStats(
            worker_id=sample.worker_id
        )
        if stats.model_ref is None and sample.model_ref:
            stats.model_ref = sample.model_ref
        stats.patch_ema_input_tokens = _ema(
            stats.patch_ema_input_tokens, sample.usage.input_tokens, alpha=alpha
        )
        stats.patch_ema_output_tokens = _ema(
            stats.patch_ema_output_tokens, sample.usage.output_tokens, alpha=alpha
        )
        state.workers[sample.worker_id] = stats

    return state
