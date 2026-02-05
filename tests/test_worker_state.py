from __future__ import annotations

from datetime import UTC, datetime

from institution_service.cost_estimator import ExpectedCostEstimator
from institution_service.costing import Price
from institution_service.schemas import (
    Bid,
    EventType,
    LedgerEvent,
    TaskSpec,
    VerifyMode,
    WorkerRuntime,
)
from institution_service.worker_state import (
    PersistedWorkerState,
    PersistedWorkerStats,
    apply_state_to_workers,
    extract_patch_usage_samples,
    update_state_from_run,
)


def test_apply_state_to_workers_applies_reputation_when_model_matches() -> None:
    state = PersistedWorkerState(
        workers={"w1": PersistedWorkerStats(worker_id="w1", model_ref="gpt-5-mini", reputation=1.2)}
    )
    workers = [WorkerRuntime(worker_id="w1", model_ref="gpt-5-mini", reputation=1.0)]
    apply_state_to_workers(state=state, workers=workers)
    assert workers[0].reputation == 1.2


def test_extract_patch_usage_samples_reads_llm_usage_from_ledger_events() -> None:
    e = LedgerEvent(
        schema_version=1,
        event_id="e1",
        prev_hash=None,
        hash="h1",
        ts=datetime.now(tz=UTC),
        run_id="r1",
        round_id=0,
        type=EventType.PATCH_SUBMITTED,
        payload={
            "task_id": "T1",
            "worker_id": "w1",
            "model_ref": "gpt-5-mini",
            "llm_usage": {"calls": 1, "input_tokens": 123, "output_tokens": 456},
        },
        artifacts=[],
    )
    samples = extract_patch_usage_samples(events=[e])
    assert len(samples) == 1
    assert samples[0].worker_id == "w1"
    assert samples[0].usage.input_tokens == 123
    assert samples[0].usage.output_tokens == 456


def test_update_state_from_run_updates_patch_ema_tokens() -> None:
    run_workers = {"w1": WorkerRuntime(worker_id="w1", model_ref="gpt-5-mini", reputation=1.1)}
    e = LedgerEvent(
        schema_version=1,
        event_id="e1",
        prev_hash=None,
        hash="h1",
        ts=datetime.now(tz=UTC),
        run_id="r1",
        round_id=0,
        type=EventType.PATCH_SUBMITTED,
        payload={
            "task_id": "T1",
            "worker_id": "w1",
            "model_ref": "gpt-5-mini",
            "llm_usage": {"calls": 1, "input_tokens": 1000, "output_tokens": 2000},
        },
        artifacts=[],
    )
    samples = extract_patch_usage_samples(events=[e])
    state = update_state_from_run(
        state=PersistedWorkerState(),
        run_workers=run_workers,
        patch_usages=samples,
        alpha=0.2,
    )
    s = state.workers["w1"]
    assert s.reputation == 1.1
    assert s.patch_ema_input_tokens == 1000
    assert s.patch_ema_output_tokens == 2000


def test_expected_cost_estimator_uses_pricing_and_ema_tokens() -> None:
    state = PersistedWorkerState(
        workers={
            "w1": PersistedWorkerStats(
                worker_id="w1",
                model_ref="gpt-5-mini",
                reputation=1.0,
                patch_ema_input_tokens=2000,
                patch_ema_output_tokens=1000,
            )
        }
    )
    pricing = {"openai:gpt-5-mini": Price(input_per_1k=1.0, output_per_1k=2.0)}
    est = ExpectedCostEstimator(state=state, pricing=pricing)
    cost = est.expected_cost(
        worker=WorkerRuntime(worker_id="w1", model_ref="gpt-5-mini", reputation=1.0),
        task=TaskSpec(id="T1", title="t", bounty=10, verify_mode=VerifyMode.MANUAL, acceptance=[]),
        bid=Bid(task_id="T1", ask=1, self_assessed_p_success=1.0, eta_minutes=1),
        round_id=0,
    )
    assert cost == 4.0
