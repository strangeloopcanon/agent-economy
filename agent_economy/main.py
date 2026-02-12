from __future__ import annotations

import argparse
import json
import shlex
import shutil
import uuid
from pathlib import Path

from agent_economy.config import load_settings, repo_root
from agent_economy.config import InstitutionSettings
from agent_economy.command_workers import (
    CommandBidder,
    CommandExecutor,
    CommandExecutorSettings,
)
from agent_economy.clearing import BidSubmission, choose_assignments, score_bid
from agent_economy.engine import (
    ClearinghouseEngine,
    EngineSettings,
    ReadyTask,
    _confidence_penalty_amount,
    _load_task_specs_from_events,
    _penalty_amount,
)
from agent_economy.ledger import HashChainedLedger
from agent_economy.llm_anthropic import AnthropicJSONClient
from agent_economy.llm_google import GoogleJSONClient
from agent_economy.llm_openai import OpenAIJSONClient
from agent_economy.llm_ollama import OllamaJSONClient
from agent_economy.llm_router import LLMRouter
from agent_economy.model_refs import split_provider_model
from agent_economy.openai_bidder import OpenAIBidder
from agent_economy.openai_executor import ExecutorSettings, OpenAIExecutor
from agent_economy.planner import toposort_plan
from agent_economy.planner_workers import decompose_with_worker
from agent_economy.scenario import load_scenario
from agent_economy.sandbox import (
    apply_file_blocks,
    apply_unified_diff_path,
    enforce_allowed_paths,
    parse_patch_changes,
)
from agent_economy.schemas import (
    Bid,
    CommandSpec,
    EventType,
    JudgeSpec,
    PaymentRule,
    SubmissionKind,
    TaskRuntime,
    TaskSpec,
    VerifyMode,
    VerifyStatus,
    WorkerRuntime,
    WorkerType,
)
from agent_economy.state import SettlementPolicy, replay_ledger
from agent_economy.cost_estimator import ExpectedCostEstimator
from agent_economy.costing import load_pricing_from_env
from agent_economy.finalize import release_judges_holdbacks
from agent_economy.workspace_ignore import copytree_ignore, is_ignored_workspace_part
from agent_economy.worker_mux import MultiplexBidder, MultiplexExecutor
from agent_economy.worker_refs import resolve_worker_ref
from agent_economy.worker_specs import CommandWorkerSpec, load_worker_pool_from_json
from agent_economy.worker_state import (
    apply_state_to_workers,
    default_state_path,
    extract_patch_usage_samples,
    load_state,
    save_state,
    update_state_from_run,
)


def _existing_path(value: str) -> Path:
    p = Path(value)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"path not found: {value}")
    return p


def _scenario_path(value: str) -> Path:
    p = _existing_path(value)
    if p.suffix.lower() not in {".yml", ".yaml"}:
        raise argparse.ArgumentTypeError(f"scenario must be YAML: {value}")
    return p


def _default_workers_spec_from_env() -> dict[str, str]:
    import os

    raw = (
        os.getenv("AE_MODELS_JSON")
        or os.getenv("INST_MODELS_JSON")
        or os.getenv("ORCH_MODELS_JSON")
        or ""
    ).strip()
    if not raw:
        # Default mirrors the repo README; model_ref is treated as OpenAI by default.
        models = {
            "gpt-4o": "gpt-4o",
            "gpt-5-mini": "gpt-5-mini",
            "gpt-5.2-auto": "gpt-5.2",
            "gpt-5.1-codex": "gpt-5.1-codex",
            "gpt-5.2-xhigh": "gpt-5.2-pro",
        }
    else:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in parsed.items()
        ):
            raise ValueError(
                "AE_MODELS_JSON/INST_MODELS_JSON must be a JSON object of {worker_id: model_ref}"
            )
        models = dict(parsed)
    return models


def _load_worker_pool(
    *, workers_path: Path | None
) -> tuple[list[WorkerRuntime], dict[str, CommandWorkerSpec], dict[str, object]]:
    if workers_path is not None:
        spec_data: object = json.loads(workers_path.read_text(encoding="utf-8"))
    else:
        spec_data = _default_workers_spec_from_env()

    pool = load_worker_pool_from_json(spec_data)
    persisted = load_state(default_state_path())
    apply_state_to_workers(state=persisted, workers=pool.workers)
    return pool.workers, dict(pool.command_specs), {"workers_spec": spec_data}


def _load_judge_workers_from_env(
    *, workers: list[WorkerRuntime], command_specs: dict[str, CommandWorkerSpec]
) -> list[str]:
    import os

    raw = (
        os.getenv("AE_JUDGES_JSON")
        or os.getenv("AE_JUDGE_WORKERS_JSON")
        or os.getenv("INST_JUDGES_JSON")
        or os.getenv("INST_JUDGE_WORKERS_JSON")
        or ""
    ).strip()
    if raw:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return [str(x) for x in parsed if str(x).strip()]
        if isinstance(parsed, dict) and all(isinstance(v, str) for v in parsed.values()):
            return [str(v) for v in parsed.values() if str(v).strip()]
        raise ValueError(
            "AE_JUDGES_JSON/AE_JUDGE_WORKERS_JSON (or INST_JUDGES_JSON/INST_JUDGE_WORKERS_JSON) must be a JSON list[str] or object of {name: worker_ref}"
        )

    defaults: list[str] = []
    for w in workers:
        if w.worker_type == WorkerType.MODEL_AGENT and w.model_ref:
            defaults.append(w.worker_id)
            continue
        if w.worker_type == WorkerType.EXTERNAL_WORKER:
            spec = command_specs.get(w.worker_id)
            if spec is not None and bool(spec.judge_cmd):
                defaults.append(w.worker_id)
    return defaults[:3]


def _planner_worker_ref_from_env(
    *, workers: list[WorkerRuntime], command_specs: dict[str, CommandWorkerSpec]
) -> str:
    import os

    raw = (
        os.getenv("AE_PLANNER_WORKER")
        or os.getenv("AE_PLANNER_MODEL")
        or os.getenv("INST_PLANNER_WORKER")
        or os.getenv("INST_PLANNER_MODEL")
        or ""
    ).strip()
    if raw:
        return raw
    for w in workers:
        if w.worker_type == WorkerType.MODEL_AGENT and w.model_ref:
            return w.worker_id
        if w.worker_type == WorkerType.EXTERNAL_WORKER:
            spec = command_specs.get(w.worker_id)
            if spec is not None and bool(spec.plan_cmd):
                return w.worker_id
    raise ValueError(
        "no planner worker available (set AE_PLANNER_WORKER/AE_PLANNER_MODEL or provide a planner-capable worker)"
    )


def _openai_client(*, settings: InstitutionSettings) -> OpenAIJSONClient | None:
    if not settings.openai_api_key:
        return None
    return OpenAIJSONClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def _ollama_client(*, settings: InstitutionSettings) -> OllamaJSONClient:
    return OllamaJSONClient(base_url=settings.ollama_base_url)


def _anthropic_client(*, settings: InstitutionSettings) -> AnthropicJSONClient | None:
    if not settings.anthropic_api_key:
        return None
    return AnthropicJSONClient(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )


def _google_client(*, settings: InstitutionSettings) -> GoogleJSONClient | None:
    if not settings.google_api_key:
        return None
    return GoogleJSONClient(api_key=settings.google_api_key)


def _llm_supports_provider(*, llm: LLMRouter, provider: str) -> bool:
    if provider == "openai":
        return llm.openai is not None
    if provider == "ollama":
        return llm.ollama is not None
    if provider == "anthropic":
        return llm.anthropic is not None
    if provider == "google":
        return llm.google is not None
    return False


def _model_providers_for_workers(*, workers: list[WorkerRuntime]) -> set[str]:
    providers: set[str] = set()
    for w in workers:
        if w.worker_type != WorkerType.MODEL_AGENT:
            continue
        if not w.model_ref:
            continue
        provider, _model = split_provider_model(w.model_ref)
        providers.add(provider)
    return providers


def _llm_router_for_workers(
    *, settings: InstitutionSettings, workers: list[WorkerRuntime]
) -> LLMRouter:
    providers = _model_providers_for_workers(workers=workers)
    if not providers:
        return LLMRouter()

    unsupported = sorted(
        p for p in providers if p not in {"openai", "ollama", "anthropic", "google"}
    )
    if unsupported:
        raise SystemExit(
            "unsupported model providers for model workers: "
            + ", ".join(unsupported)
            + " (supported: openai, ollama, anthropic, google)"
        )

    openai_client = _openai_client(settings=settings) if "openai" in providers else None
    if "openai" in providers and openai_client is None:
        raise SystemExit("OPENAI_API_KEY is required for OpenAI model workers")

    ollama_client = _ollama_client(settings=settings) if "ollama" in providers else None
    anthropic_client = _anthropic_client(settings=settings) if "anthropic" in providers else None
    if "anthropic" in providers and anthropic_client is None:
        raise SystemExit("ANTHROPIC_API_KEY is required for Anthropic model workers")

    google_client = _google_client(settings=settings) if "google" in providers else None
    if "google" in providers and google_client is None:
        raise SystemExit(
            "GOOGLE_API_KEY or GEMINI_API_KEY is required for Google/Gemini model workers"
        )

    return LLMRouter(
        openai=openai_client,
        ollama=ollama_client,
        anthropic=anthropic_client,
        google=google_client,
    )


def _timeout_seconds_from_env(*, env_vars: tuple[str, ...], default: float | None) -> float | None:
    import os

    env_var = env_vars[0]
    raw = ""
    for candidate in env_vars:
        candidate_raw = (os.getenv(candidate) or "").strip()
        if candidate_raw:
            env_var = candidate
            raw = candidate_raw
            break
    if not raw:
        return default
    if raw.lower() in {"0", "none", "off", "false"}:
        return None
    try:
        value = float(raw)
    except Exception as e:
        raise ValueError(f"{env_var} must be a positive number, 0, or 'none'") from e
    if value <= 0:
        raise ValueError(f"{env_var} must be > 0 when set")
    return value


def _engine_settings(*, max_concurrency: int) -> EngineSettings:
    return EngineSettings(
        max_concurrency=int(max_concurrency),
        bid_timeout_seconds=_timeout_seconds_from_env(
            env_vars=("AE_BID_TIMEOUT_SECONDS", "INST_BID_TIMEOUT_SECONDS"),
            default=30.0,
        ),
        execution_timeout_seconds=_timeout_seconds_from_env(
            env_vars=("AE_EXECUTION_TIMEOUT_SECONDS", "INST_EXECUTION_TIMEOUT_SECONDS"),
            default=300.0,
        ),
    )


def _list_repo_files(*, root: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(is_ignored_workspace_part(part) for part in rel.parts):
            continue
        out.append(str(rel))
        if len(out) >= 5000:
            break
    return out


def _planner_context_files(*, root: Path, file_list: list[str]) -> dict[str, str]:
    def read_preview(p: Path, *, max_bytes: int) -> str:
        try:
            data = p.read_bytes()[:max_bytes]
        except Exception:
            return "<unreadable>"
        return data.decode("utf-8", errors="replace")

    candidates: list[str] = []
    for rel in file_list:
        name = Path(rel).name
        if name in {"README.md", "Makefile", "pyproject.toml"}:
            candidates.append(rel)
            continue
        if not rel.endswith(".py"):
            continue
        if (
            "/tests/" in rel
            or rel.startswith("tests/")
            or name.startswith("test_")
            or name.endswith("_test.py")
        ):
            candidates.append(rel)

    seen: set[str] = set()
    selected: list[str] = []
    for rel in candidates:
        if rel in seen:
            continue
        seen.add(rel)
        selected.append(rel)
        if len(selected) >= 8:
            break

    out: dict[str, str] = {}
    total_bytes = 0
    for rel in selected:
        p = root / rel
        if not p.exists() or not p.is_file():
            continue
        preview = read_preview(p, max_bytes=4000).rstrip()
        if not preview:
            continue
        out[rel] = preview
        total_bytes += len(preview.encode("utf-8", errors="ignore"))
        if total_bytes >= 24_000:
            break
    return out


def _acceptance_files_hint(*, root: Path, commands: list[str]) -> list[str]:
    root_resolved = root.resolve()

    def is_under_root(p: Path) -> bool:
        try:
            p.resolve().relative_to(root_resolved)
        except Exception:
            return False
        return True

    def candidate_paths(token: str) -> list[str]:
        token = str(token or "").strip()
        if not token or token.startswith("-"):
            return []
        if "::" in token:
            token = token.split("::", 1)[0]
        return [token]

    hints: list[str] = []
    for cmd in commands:
        cmd = str(cmd or "").strip()
        if not cmd:
            continue
        try:
            parts = shlex.split(cmd)
        except Exception:
            parts = cmd.split()
        for part in parts:
            for cand in candidate_paths(part):
                rel = Path(cand)
                if rel.is_absolute():
                    continue
                p = (root / rel).resolve()
                if not is_under_root(p):
                    continue
                if not p.exists():
                    continue
                try:
                    hints.append(str(p.relative_to(root_resolved)))
                except Exception:
                    continue

    # Add nearby pytest conftest.py files (often required to make imports work).
    for rel in list(hints):
        p = Path(rel)
        cur = p if (root / p).is_dir() else p.parent
        while True:
            conftest = cur / "conftest.py" if str(cur) not in (".", "") else Path("conftest.py")
            if (root / conftest).exists():
                hints.append(str(conftest))
            if cur == cur.parent:
                break
            cur = cur.parent

    return list(dict.fromkeys(hints))


def _copytree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=False, ignore=copytree_ignore)


def _task_status_counts(*, state) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in state.tasks.values():
        status = str(getattr(t, "status", "") or "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _print_report(*, state) -> None:
    counts = _task_status_counts(state=state)
    print(
        f"run_id={state.run_id} round_id={state.round_id} payment_rule={state.payment_rule.value} "
        f"tasks_done={counts.get('DONE', 0)} tasks_review={counts.get('REVIEW', 0)} "
        f"tasks_todo={counts.get('TODO', 0)}"
    )
    print("\nTasks:")
    for tid, t in sorted(state.tasks.items()):
        print(
            f"- {tid}: {t.status} "
            f"bounty={t.bounty_current} fails={t.fail_count} "
            f"assigned={t.assigned_worker}"
        )
    print("\nWorkers:")
    for wid, w in sorted(state.workers.items()):
        print(
            f"- {wid}: bal={w.balance:.2f} rep={w.reputation:.2f} "
            f"wins={w.wins} done={w.completions} fails={w.failures}"
        )


def _print_next_steps(*, run_dir: Path, workspace_dir: Path, state) -> None:
    print(f"\nRun directory: {run_dir}")
    print(f"Workspace (with applied patches): {workspace_dir}")
    print(f"Ledger: {run_dir / 'ledger.jsonl'}")
    print(f"Full report: agent-economy report --run-dir {run_dir}")
    if state.tasks and any(t.status == "REVIEW" for t in state.tasks.values()):
        print(f"Manual review pending: agent-economy review list --run-dir {run_dir}")


def _load_submission_artifact_path(*, run_dir: Path, patch_event) -> Path:
    by_name = {a.name: a for a in getattr(patch_event, "artifacts", [])}
    for name in ("patch.diff", "patch_files.json", "submission.txt", "submission.json"):
        a = by_name.get(name)
        if a is None or not a.path:
            continue
        p = run_dir / a.path
        if p.exists():
            return p
    raise SystemExit(
        "missing submission artifact (expected patch.diff, patch_files.json, submission.txt, or submission.json)"
    )


def _find_review_events(*, events: list, task_id: str, worker_id: str) -> tuple[object, object]:
    completed = None
    for e in reversed(events):
        if getattr(e, "type", None) != EventType.TASK_COMPLETED:
            continue
        p = getattr(e, "payload", {}) or {}
        if str(p.get("task_id") or "") != task_id:
            continue
        if str(p.get("worker_id") or "") != worker_id:
            continue
        if str(p.get("verify_status") or "") != VerifyStatus.MANUAL_REVIEW.value:
            continue
        completed = e
        break
    if completed is None:
        raise SystemExit(
            f"no MANUAL_REVIEW attempt found for task_id={task_id} worker_id={worker_id}"
        )

    patch = None
    round_id = int(getattr(completed, "round_id", 0) or 0)
    for e in reversed(events):
        if getattr(e, "type", None) != EventType.PATCH_SUBMITTED:
            continue
        if int(getattr(e, "round_id", 0) or 0) != round_id:
            continue
        p = getattr(e, "payload", {}) or {}
        if str(p.get("task_id") or "") != task_id:
            continue
        if str(p.get("worker_id") or "") != worker_id:
            continue
        patch = e
        break
    if patch is None:
        raise SystemExit(f"no PATCH_SUBMITTED found for task_id={task_id} worker_id={worker_id}")
    return completed, patch


def _run_rounds(
    *,
    ledger: HashChainedLedger,
    engine: ClearinghouseEngine,
    bidder,
    executor,
    cost_estimator: ExpectedCostEstimator,
    rounds: int,
) -> tuple[list, object]:
    after_events = list(ledger.iter_events())
    state = replay_ledger(events=after_events)

    for _ in range(int(rounds)):
        before = len(after_events)
        engine.step(bidder=bidder, executor=executor, cost_estimator=cost_estimator)
        after_events = list(ledger.iter_events())
        if len(after_events) == before:
            break
        state = replay_ledger(events=after_events)
        if state.tasks and all(t.status in {"DONE", "REVIEW"} for t in state.tasks.values()):
            break

    return after_events, state


def _persist_worker_state(*, run_workers, events: list) -> None:
    persist_path = default_state_path()
    persisted = load_state(persist_path)
    persisted = update_state_from_run(
        state=persisted,
        run_workers=run_workers,
        patch_usages=extract_patch_usage_samples(events=events),
    )
    save_state(persist_path, persisted)


def _finalize_run(*, run_dir: Path, ledger: HashChainedLedger) -> tuple[list, object]:
    release_judges_holdbacks(ledger=ledger)
    after_events = list(ledger.iter_events())
    state = replay_ledger(events=after_events)

    (run_dir / "state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
    _persist_worker_state(run_workers=state.workers, events=after_events)
    return after_events, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-economy")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init", help="create a new run directory + workspace + ledger")
    init_p.add_argument("--scenario", type=_scenario_path, required=True)
    init_p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="output run directory (default: runs/<run_id>)",
    )
    init_p.add_argument("--run-id", type=str, default=None)
    init_p.add_argument(
        "--workers",
        type=_existing_path,
        default=None,
        help="worker pool JSON (model_ref and/or exec_cmd workers)",
    )
    init_p.add_argument(
        "--workspace-src",
        type=_existing_path,
        default=None,
        help="source workspace directory to copy (overrides scenario.template_dir)",
    )
    init_p.add_argument("--payment-rule", choices=["ask", "bounty"], default="ask")
    init_p.add_argument("--overwrite", action="store_true")

    run_p = sub.add_parser("run", help="run market rounds for an existing run dir")
    run_p.add_argument("--run-dir", type=_existing_path, required=True)
    run_p.add_argument("--rounds", type=int, default=12)
    run_p.add_argument("--concurrency", type=int, default=1)
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="preview what would happen without executing",
    )

    # Config validation subcommand
    config_p = sub.add_parser("config", help="configuration utilities")
    config_sub = config_p.add_subparsers(dest="config_cmd", required=True)
    validate_p = config_sub.add_parser("validate", help="validate configuration")
    validate_p.add_argument(
        "--workers",
        type=Path,
        default=None,
        help="worker pool JSON to validate",
    )
    validate_p.add_argument(
        "--scenario",
        type=Path,
        default=None,
        help="scenario YAML to validate",
    )

    def _add_single_task_args(p: argparse.ArgumentParser, *, default_decompose: bool) -> None:
        p.add_argument(
            "prompt", nargs="?", default=None, help="optional goal/description shorthand"
        )
        p.add_argument("--task-id", type=str, default="T1")
        p.add_argument("--title", type=str, default=None)
        p.add_argument("--description", type=str, default="")
        p.add_argument("--bounty", type=int, default=40)
        p.add_argument(
            "--verify-mode",
            choices=["auto", "commands", "manual", "judges"],
            default="auto",
            help="verification mode (auto picks commands if --accept/--hidden-accept or --decompose, else judges)",
        )
        p.add_argument("--accept", action="append", default=[], help="public acceptance command")
        p.add_argument(
            "--hidden-accept", action="append", default=[], help="hidden acceptance command"
        )
        p.add_argument("--allowed-path", action="append", default=[], help="allowed patch path")
        p.add_argument("--files-hint", action="append", default=[], help="file path hint")
        p.add_argument(
            "--submission-kind",
            choices=["patch", "text", "json"],
            default="patch",
            help="worker submission type (patch modifies files, text/json submits an answer artifact)",
        )
        p.add_argument("--goal", type=str, default=None, help="high-level goal (for --decompose)")
        g = p.add_mutually_exclusive_group()
        g.add_argument(
            "--decompose",
            dest="decompose",
            action="store_true",
            help="use a planner to turn --goal into multiple dependent tasks",
        )
        g.add_argument(
            "--no-decompose",
            dest="decompose",
            action="store_false",
            help="skip planning and run a single task",
        )
        p.set_defaults(decompose=bool(default_decompose))
        p.add_argument(
            "--max-tasks", type=int, default=8, help="max tasks to create in --decompose"
        )
        p.add_argument(
            "--planner-worker",
            "--planner-model",
            type=str,
            default=None,
            help="planner worker ref (default: AE_PLANNER_WORKER/AE_PLANNER_MODEL or first capable worker)",
        )
        p.add_argument(
            "--judge-worker",
            "--judge-model",
            action="append",
            default=[],
            help="judge worker ref (worker_id or model_ref); repeatable",
        )
        p.add_argument("--judge-min-passes", type=int, default=None)
        p.add_argument("--no-self-judge", action="store_true")
        p.add_argument(
            "--workspace-src",
            type=_existing_path,
            default=None,
            help="source workspace directory to copy (default: current directory)",
        )
        p.add_argument(
            "--workers",
            type=_existing_path,
            default=None,
            help="worker pool JSON (model_ref and/or exec_cmd workers)",
        )
        p.add_argument(
            "--run-dir",
            type=Path,
            default=None,
            help="output run directory (default: runs/<run_id>)",
        )
        p.add_argument("--run-id", type=str, default=None)
        p.add_argument("--payment-rule", choices=["ask", "bounty"], default="ask")
        p.add_argument("--rounds", type=int, default=12)
        p.add_argument("--concurrency", type=int, default=1)
        p.add_argument("--overwrite", action="store_true")
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="preview what would happen without executing",
        )

    oneshot_p = sub.add_parser(
        "oneshot", help="create + run a one-off run dir (single task or --decompose)"
    )
    _add_single_task_args(oneshot_p, default_decompose=False)
    task_p = sub.add_parser("task", help="planning-first entrypoint (defaults to --decompose)")
    _add_single_task_args(task_p, default_decompose=True)

    review_p = sub.add_parser("review", help="list/approve/reject manual-review tasks")
    review_sub = review_p.add_subparsers(dest="review_cmd", required=True)

    review_list_p = review_sub.add_parser("list", help="list tasks awaiting manual review")
    review_list_p.add_argument("--run-dir", type=_existing_path, required=True)

    review_appr_p = review_sub.add_parser("approve", help="approve a manual-review patch")
    review_appr_p.add_argument("--run-dir", type=_existing_path, required=True)
    review_appr_p.add_argument("--task-id", type=str, required=True)

    review_rej_p = review_sub.add_parser("reject", help="reject a manual-review patch")
    review_rej_p.add_argument("--run-dir", type=_existing_path, required=True)
    review_rej_p.add_argument("--task-id", type=str, required=True)
    review_rej_p.add_argument(
        "--no-fault",
        action="store_true",
        help="treat rejection as no-fault (no penalty, no rep loss, no fail_count)",
    )

    rep_p = sub.add_parser("report", help="print a summary report for a run dir")

    rep_p.add_argument("--run-dir", type=_existing_path, required=True)

    inject_p = sub.add_parser("inject", help="inject a task into an existing run")
    _add_single_task_args(inject_p, default_decompose=False)

    # Dashboard subcommand
    dash_p = sub.add_parser("dashboard", help="launch real-time visualization dashboard")
    dash_p.add_argument("--run-dir", type=_existing_path, required=True)
    dash_p.add_argument("--port", type=int, default=8080, help="port to serve dashboard on")
    dash_p.add_argument(
        "--no-browser",
        action="store_true",
        help="don't automatically open browser",
    )

    args = parser.parse_args(argv)
    settings = load_settings()  # validates env, but avoid printing secrets

    if args.cmd == "config":
        if args.config_cmd == "validate":
            issues: list[str] = []
            checks_passed = 0
            worker_providers: set[str] | None = None

            # Check .env file
            env_path = repo_root() / ".env"
            if env_path.exists():
                print(f"✓ .env file found: {env_path}")
                checks_passed += 1
            else:
                issues.append(f"⚠ .env file not found at {env_path}")

            # Validate workers JSON if provided
            if args.workers:
                workers_path = Path(args.workers)
                if not workers_path.exists():
                    issues.append(f"✗ Workers file not found: {workers_path}")
                else:
                    try:
                        workers, _, _ = _load_worker_pool(workers_path=workers_path)
                        print(f"✓ Workers file valid: {len(workers)} workers loaded")
                        worker_providers = _model_providers_for_workers(workers=workers)
                        for w in workers:
                            print(f"  - {w.worker_id} ({w.worker_type.value})")
                        checks_passed += 1
                    except Exception as e:
                        issues.append(f"✗ Workers file invalid: {e}")

            require_openai = worker_providers is None or "openai" in worker_providers
            if require_openai:
                if settings.openai_api_key:
                    if settings.openai_api_key.startswith("sk-"):
                        print("✓ OPENAI_API_KEY is set (starts with sk-)")
                        checks_passed += 1
                    else:
                        issues.append("⚠ OPENAI_API_KEY does not start with 'sk-' (may be invalid)")
                else:
                    issues.append("⚠ OPENAI_API_KEY is not set")
            else:
                print("✓ OPENAI_API_KEY not required by current workers config")
                checks_passed += 1

            if worker_providers is not None and "ollama" in worker_providers:
                print(f"✓ OLLAMA_BASE_URL configured: {settings.ollama_base_url}")
                checks_passed += 1

            if worker_providers is not None and "anthropic" in worker_providers:
                if settings.anthropic_api_key:
                    print("✓ ANTHROPIC_API_KEY is set")
                    checks_passed += 1
                else:
                    issues.append("⚠ ANTHROPIC_API_KEY is not set")

            if worker_providers is not None and "google" in worker_providers:
                if settings.google_api_key:
                    print("✓ GOOGLE_API_KEY/GEMINI_API_KEY is set")
                    checks_passed += 1
                else:
                    issues.append("⚠ GOOGLE_API_KEY or GEMINI_API_KEY is not set")

            # Validate scenario if provided
            if args.scenario:
                scenario_path = Path(args.scenario)
                if not scenario_path.exists():
                    issues.append(f"✗ Scenario file not found: {scenario_path}")
                else:
                    try:
                        scenario = load_scenario(scenario_path)
                        print(f"✓ Scenario file valid: {scenario.scenario_id}")
                        print(f"  - {len(scenario.tasks)} tasks defined")
                        for t in scenario.tasks:
                            deps_str = f" (deps: {', '.join(t.deps)})" if t.deps else ""
                            print(f"    • {t.id}: {t.title}{deps_str}")
                        checks_passed += 1
                    except Exception as e:
                        issues.append(f"✗ Scenario file invalid: {e}")

            # Summary
            print()
            if issues:
                print("Issues found:")
                for issue in issues:
                    print(f"  {issue}")
                print(f"\n{checks_passed} checks passed, {len(issues)} issues found")
                return 1
            else:
                print(f"✓ All {checks_passed} checks passed")
                return 0

    if args.cmd == "dashboard":
        from agent_economy.dashboard.server import run_dashboard

        run_dir = Path(args.run_dir)
        run_dashboard(
            run_dir=run_dir,
            port=int(args.port),
            open_browser=not args.no_browser,
        )
        return 0

    if args.cmd == "init":
        scenario = load_scenario(Path(args.scenario))
        workers, _command_specs, cfg_extra = _load_worker_pool(workers_path=args.workers)

        run_id = str(args.run_id or uuid.uuid4())
        run_dir: Path = args.run_dir or (repo_root() / "runs" / run_id)
        if run_dir.exists():
            if not args.overwrite:
                raise SystemExit(f"run dir already exists (use --overwrite): {run_dir}")
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)

        workspace_dir = run_dir / "workspace"
        src = Path(args.workspace_src) if args.workspace_src else scenario.template_dir
        if src is None:
            raise SystemExit(
                "no workspace source provided (use --workspace-src or scenario.template_dir)"
            )
        _copytree(src, workspace_dir)

        ledger = HashChainedLedger(run_dir / "ledger.jsonl")
        engine_settings = EngineSettings()
        engine = ClearinghouseEngine(ledger=ledger, settings=engine_settings)
        engine.create_run(
            run_id=run_id,
            payment_rule=PaymentRule(args.payment_rule),
            workers=workers,
            tasks=scenario.tasks,
        )

        config = {
            "run_id": run_id,
            "scenario_path": str(Path(args.scenario).resolve()),
            "workspace_dir": str(workspace_dir.resolve()),
            "payment_rule": args.payment_rule,
            "workers": [w.model_dump() for w in workers],
            **cfg_extra,
        }
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        print(f"Initialized run_id={run_id} run_dir={run_dir}")
        return 0

    if args.cmd == "run":
        run_dir = Path(args.run_dir)
        cfg_path = run_dir / "run_config.json"
        if not cfg_path.exists():
            raise SystemExit(f"missing run_config.json in {run_dir} (run init first)")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        ledger = HashChainedLedger(run_dir / "ledger.jsonl")
        engine_settings = _engine_settings(max_concurrency=int(args.concurrency))
        engine = ClearinghouseEngine(ledger=ledger, settings=engine_settings)

        events = list(ledger.iter_events())
        if not events:
            raise SystemExit("empty ledger (run init first)")
        state = replay_ledger(events=events)
        task_specs = _load_task_specs_from_events(events=events)

        workspace_dir = Path(str(cfg.get("workspace_dir") or ""))
        if not workspace_dir.exists():
            raise SystemExit(f"workspace_dir not found: {workspace_dir}")

        run_workers = list(state.workers.values())
        has_model_agents = bool(_model_providers_for_workers(workers=run_workers))
        needs_judges = any(t.verify_mode == VerifyMode.JUDGES for t in task_specs.values())

        # Dry-run: print preview and exit
        if getattr(args, "dry_run", False):
            print("=== DRY RUN PREVIEW ===\n")
            print(f"Run Directory: {run_dir}")
            print(f"Workspace: {workspace_dir}")
            print(f"Rounds: {args.rounds}")
            print(f"Concurrency: {args.concurrency}")

            print(f"\nTasks ({len(task_specs)}):")
            for tid, t in sorted(task_specs.items()):
                runtime = state.tasks.get(tid)
                status = runtime.status if runtime else "TODO"
                bounty = runtime.bounty_current if runtime else t.bounty
                deps_str = f" (deps: {', '.join(t.deps)})" if t.deps else ""
                print(f"  • [{status}] {tid}: {t.title} (bounty={bounty}){deps_str}")
                if t.verify_mode == VerifyMode.COMMANDS:
                    for cmd in t.acceptance[:2]:
                        print(f"      acceptance: {cmd.cmd}")

            print(f"\nWorkers ({len(state.workers)}):")
            for wid, w in sorted(state.workers.items()):
                model_str = f" model={w.model_ref}" if w.model_ref else ""
                print(f"  • {wid} ({w.worker_type.value}) rep={w.reputation:.2f}{model_str}")

            print("\n=== END DRY RUN (no changes made) ===")
            return 0

        llm = _llm_router_for_workers(settings=settings, workers=run_workers)

        spec_data = cfg.get("workers_spec")
        command_specs: dict[str, CommandWorkerSpec] = {}
        if spec_data is not None:
            try:
                pool = load_worker_pool_from_json(spec_data)
            except Exception:
                pool = None
            if pool is not None:
                command_specs = dict(pool.command_specs)

        has_external = any(
            w.worker_type == WorkerType.EXTERNAL_WORKER for w in state.workers.values()
        )
        if has_external:
            missing = sorted(
                wid
                for wid, w in state.workers.items()
                if w.worker_type == WorkerType.EXTERNAL_WORKER and wid not in command_specs
            )
            if missing:
                raise SystemExit(
                    "missing command worker specs in run_config.json for: " + ", ".join(missing)
                )

        judge_workers = _load_judge_workers_from_env(
            workers=run_workers, command_specs=command_specs
        )
        if needs_judges:
            needs_default_judges = any(
                t.verify_mode == VerifyMode.JUDGES and (t.judges is None or not t.judges.workers)
                for t in task_specs.values()
            )
            if needs_default_judges and not judge_workers:
                raise SystemExit("judge workers required (set AE_JUDGES_JSON)")

            unknown: set[str] = set()
            for t in task_specs.values():
                if t.verify_mode != VerifyMode.JUDGES:
                    continue
                refs = (
                    list(t.judges.workers)
                    if t.judges is not None and t.judges.workers
                    else list(judge_workers)
                )
                for ref in refs:
                    if resolve_worker_ref(str(ref), workers=run_workers) is None:
                        unknown.add(str(ref))
            if unknown:
                raise SystemExit("unknown judge worker refs: " + ", ".join(sorted(unknown)))

        model_bidder = (
            None
            if not has_model_agents
            else OpenAIBidder(
                llm=llm,
                payment_rule=state.payment_rule,
                max_bids=engine_settings.max_bids_per_worker,
            )
        )
        model_executor = (
            None
            if not has_model_agents
            else OpenAIExecutor(
                llm=llm,
                workspace_dir=workspace_dir,
                run_dir=run_dir,
                workers=run_workers,
                command_specs=command_specs,
                settings=ExecutorSettings(judge_workers=judge_workers),
            )
        )

        ext_bidder = (
            None
            if not has_external
            else CommandBidder(
                workspace_dir=workspace_dir,
                payment_rule=state.payment_rule,
                specs=command_specs,
                max_bids=engine_settings.max_bids_per_worker,
            )
        )
        ext_executor = (
            None
            if not has_external
            else CommandExecutor(
                workspace_dir=workspace_dir,
                run_dir=run_dir,
                workers=run_workers,
                specs=command_specs,
                settings=CommandExecutorSettings(judge_workers=judge_workers),
                llm=llm,
            )
        )

        bidder = MultiplexBidder(model_bidder=model_bidder, external_bidder=ext_bidder)
        executor = MultiplexExecutor(model_executor=model_executor, external_executor=ext_executor)
        cost_estimator = ExpectedCostEstimator(
            state=load_state(default_state_path()),
            pricing=load_pricing_from_env(),
        )

        _run_rounds(
            ledger=ledger,
            engine=engine,
            bidder=bidder,
            executor=executor,
            cost_estimator=cost_estimator,
            rounds=int(args.rounds),
        )
        _, state = _finalize_run(run_dir=run_dir, ledger=ledger)
        _print_report(state=state)
        _print_next_steps(run_dir=run_dir, workspace_dir=workspace_dir, state=state)
        return 0

    if args.cmd == "inject":
        if not args.run_dir:
            raise SystemExit("--run-dir is required for inject")
        run_dir = Path(args.run_dir)
        ledger = HashChainedLedger(run_dir / "ledger.jsonl")

        # We need the current round_id to append correctly.
        # Reading the whole ledger is safe enough for now.
        events = list(ledger.iter_events())
        if not events:
            raise SystemExit("empty ledger (run init first)")
        state = replay_ledger(events=events)
        if str(args.task_id) in state.tasks:
            raise SystemExit(f"task_id already exists in run: {args.task_id}")

        # Create TaskSpec from CLI args
        raw_verify_mode = str(args.verify_mode)
        if raw_verify_mode == "auto":
            raw_verify_mode = (
                "commands" if (args.accept or args.hidden_accept or args.decompose) else "judges"
            )
        submission_kind = SubmissionKind(str(args.submission_kind))

        # acceptance/hidden_acceptance need to be parsed into CommandSpec
        acceptance = [CommandSpec(cmd=cmd) for cmd in args.accept]
        hidden_acceptance = [CommandSpec(cmd=cmd) for cmd in args.hidden_accept]

        task_spec = TaskSpec(
            id=args.task_id,
            title=args.title or args.task_id,
            description=args.description,
            bounty=args.bounty,
            verify_mode=VerifyMode(raw_verify_mode),
            submission_kind=submission_kind,
            acceptance=acceptance,
            hidden_acceptance=hidden_acceptance,
            allowed_paths=args.allowed_path if args.allowed_path else ["./"],
            files_hint=args.files_hint if args.files_hint else [],
        )

        engine = ClearinghouseEngine(ledger=ledger)
        engine.inject_task(
            run_id=state.run_id,
            round_id=state.round_id,  # Append to current round
            task=task_spec,
        )
        print(f"Injected Task {task_spec.id} into run {state.run_id} (round {state.round_id})")
        return 0

    if args.cmd in {"oneshot", "task"}:
        workers, command_specs, cfg_extra = _load_worker_pool(workers_path=args.workers)

        prompt = str(args.prompt or "").strip()
        if prompt and not args.goal:
            args.goal = prompt
        if prompt and not args.description:
            args.description = prompt

        run_id = str(args.run_id or uuid.uuid4())
        run_dir: Path = args.run_dir or (repo_root() / "runs" / run_id)
        if run_dir.exists():
            if not args.overwrite:
                raise SystemExit(f"run dir already exists (use --overwrite): {run_dir}")
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)

        workspace_dir = run_dir / "workspace"
        src = Path(args.workspace_src) if args.workspace_src else Path.cwd()
        _copytree(src, workspace_dir)

        allowed_paths = list(args.allowed_path) if args.allowed_path else ["./"]
        raw_verify_mode = str(args.verify_mode)
        if raw_verify_mode == "auto":
            raw_verify_mode = (
                "commands" if (args.accept or args.hidden_accept or args.decompose) else "judges"
            )
        verify_mode = VerifyMode(raw_verify_mode)
        submission_kind = SubmissionKind(str(args.submission_kind))

        judge_workers = (
            [str(m) for m in list(args.judge_worker) if str(m).strip()]
            if args.judge_worker
            else _load_judge_workers_from_env(workers=workers, command_specs=command_specs)
        )
        seen = set()
        judge_workers = [m for m in judge_workers if not (m in seen or seen.add(m))]

        judges = None
        if verify_mode == VerifyMode.JUDGES:
            unknown = sorted(
                {
                    ref
                    for ref in judge_workers
                    if resolve_worker_ref(str(ref), workers=workers) is None
                }
            )
            if unknown:
                raise SystemExit("unknown judge worker refs: " + ", ".join(unknown))
            judges = JudgeSpec(
                workers=judge_workers,
                min_passes=args.judge_min_passes,
                include_self=not bool(args.no_self_judge),
            )

        model_providers = _model_providers_for_workers(workers=workers)
        has_model_agents = bool(model_providers)
        has_external = any(w.worker_type == WorkerType.EXTERNAL_WORKER for w in workers)
        needs_judges = verify_mode == VerifyMode.JUDGES
        if needs_judges and not judge_workers:
            raise SystemExit("judge workers required (use --judge-worker or set AE_JUDGES_JSON)")

        # Dry-run: print preview and exit
        if getattr(args, "dry_run", False):
            print("=== DRY RUN PREVIEW ===\n")
            print(f"Command: {args.cmd}")
            print(f"Mode: {'decompose' if args.decompose else 'single task'}")
            print(f"Workspace Source: {src}")
            print(f"Run Directory: {run_dir} (would be created)")
            print(f"Rounds: {args.rounds}")
            print(f"Concurrency: {args.concurrency}")
            print(f"Verify Mode: {verify_mode.value}")
            print(f"Submission Kind: {submission_kind.value}")

            if args.decompose:
                print(f"\nGoal: {args.goal or args.description}")
                print(f"Max Tasks: {args.max_tasks}")
            else:
                print(f"\nTask: {args.task_id}")
                print(f"  Title: {args.title or args.task_id}")
                print(f"  Bounty: {args.bounty}")
                print(f"  Allowed Paths: {', '.join(allowed_paths)}")
                if args.accept:
                    print("  Acceptance Commands:")
                    for cmd in args.accept[:3]:
                        print(f"    • {cmd}")

            print(f"\nWorkers ({len(workers)}):")
            for w in workers:
                model_str = f" model={w.model_ref}" if w.model_ref else ""
                print(f"  • {w.worker_id} ({w.worker_type.value}){model_str}")

            if judge_workers:
                print(f"\nJudge Workers: {', '.join(judge_workers)}")

            # Clean up the run dir we created
            if run_dir.exists():
                shutil.rmtree(run_dir)

            print("\n=== END DRY RUN (no changes made) ===")
            return 0

        llm = _llm_router_for_workers(settings=settings, workers=workers)

        cost_estimator = ExpectedCostEstimator(
            state=load_state(default_state_path()),
            pricing=load_pricing_from_env(),
        )

        planner_meta: dict[str, object] | None = None
        tasks: list[TaskSpec] = []
        if args.decompose:
            import os

            goal = str(args.goal or args.description or "").strip()
            if not goal:
                raise SystemExit("--goal (or --description) is required when --decompose is set")

            planner_ref = str(args.planner_worker or "").strip()
            if planner_ref:
                planner_meta = {"selection": "cli", "planner_ref": planner_ref}
            else:
                env_ref = (
                    os.getenv("AE_PLANNER_WORKER")
                    or os.getenv("AE_PLANNER_MODEL")
                    or os.getenv("INST_PLANNER_WORKER")
                    or os.getenv("INST_PLANNER_MODEL")
                    or ""
                ).strip()
                if env_ref:
                    planner_ref = env_ref
                    planner_meta = {"selection": "env", "planner_ref": planner_ref}
                else:
                    planner_bounty = max(1, int(args.bounty))
                    plan_task = TaskSpec(
                        id="PLAN",
                        title="Planner: decompose goal into tasks",
                        description="\n".join(
                            [
                                "Produce a decomposition plan (task DAG) for the overall goal.",
                                "",
                                "Overall goal:",
                                goal,
                                "",
                                f"Constraints: max_tasks={int(args.max_tasks)}",
                                f"Allowed paths: {', '.join(allowed_paths)}",
                            ]
                        ).strip(),
                        deps=[],
                        bounty=planner_bounty,
                        verify_mode=VerifyMode.MANUAL,
                        allowed_paths=[str(p) for p in allowed_paths],
                    )
                    plan_rt = TaskRuntime(
                        task_id=plan_task.id,
                        bounty_current=planner_bounty,
                        bounty_original=planner_bounty,
                    )
                    plan_ready = ReadyTask(spec=plan_task, runtime=plan_rt)

                    candidates: list[WorkerRuntime] = []
                    for w in workers:
                        if w.worker_type == WorkerType.MODEL_AGENT:
                            if not w.model_ref:
                                continue
                            provider, _ = split_provider_model(w.model_ref)
                            if not _llm_supports_provider(llm=llm, provider=provider):
                                continue
                            candidates.append(w)
                            continue
                        if w.worker_type == WorkerType.EXTERNAL_WORKER:
                            spec = command_specs.get(w.worker_id)
                            if spec is None or not spec.plan_cmd:
                                continue
                            candidates.append(w)

                    if not candidates:
                        raise SystemExit(
                            "no planner-capable workers available (need a supported model worker or an external worker with plan_cmd)"
                        )

                    plan_model_bidder = (
                        None
                        if not any(w.worker_type == WorkerType.MODEL_AGENT for w in candidates)
                        else OpenAIBidder(
                            llm=llm,
                            payment_rule=PaymentRule(args.payment_rule),
                            max_bids=1,
                        )
                    )
                    plan_ext_bidder = (
                        None
                        if not any(w.worker_type == WorkerType.EXTERNAL_WORKER for w in candidates)
                        else CommandBidder(
                            workspace_dir=workspace_dir,
                            payment_rule=PaymentRule(args.payment_rule),
                            specs=command_specs,
                            max_bids=1,
                        )
                    )
                    plan_bidder = MultiplexBidder(
                        model_bidder=plan_model_bidder, external_bidder=plan_ext_bidder
                    )

                    bids_by_task: dict[str, list[BidSubmission]] = {plan_task.id: []}
                    bid_records: list[dict[str, object]] = []
                    for w in sorted(candidates, key=lambda w: w.worker_id):
                        bids: list[Bid] = []
                        err: str | None = None
                        try:
                            resp = plan_bidder.get_bids(
                                worker=w, ready_tasks=[plan_ready], round_id=0
                            )
                            for raw in list(resp.bids)[:1]:
                                if isinstance(raw, Bid):
                                    bids.append(raw)
                                else:
                                    try:
                                        bids.append(Bid.model_validate(raw))
                                    except Exception:
                                        continue
                        except Exception as e:
                            err = f"{type(e).__name__}: {e}"

                        if err is not None:
                            bid_records.append({"worker_id": w.worker_id, "error": err})
                            continue

                        if not bids:
                            bid_records.append({"worker_id": w.worker_id, "bids": []})
                            continue

                        bid = bids[0]
                        if bid.task_id != plan_task.id:
                            bid_records.append(
                                {"worker_id": w.worker_id, "bids": [bid.model_dump(mode="json")]}
                            )
                            continue

                        expected_cost = 0.0
                        try:
                            expected_cost = float(
                                cost_estimator.expected_cost(
                                    worker=w, task=plan_task, bid=bid, round_id=0
                                )
                            )
                        except Exception:
                            expected_cost = 0.0

                        score = float(
                            score_bid(
                                bounty=planner_bounty,
                                reputation=float(w.reputation),
                                bid=bid,
                                expected_cost=expected_cost,
                            )
                        )
                        bid_records.append(
                            {
                                "worker_id": w.worker_id,
                                "bid": bid.model_dump(mode="json"),
                                "score": round(score, 6),
                                "expected_cost": round(float(expected_cost), 6),
                            }
                        )
                        bids_by_task[plan_task.id].append(
                            BidSubmission(
                                worker_id=w.worker_id, bid=bid, expected_cost=expected_cost
                            )
                        )

                    assignments = choose_assignments(
                        ready_tasks=[plan_rt],
                        available_workers=candidates,
                        bids_by_task=bids_by_task,
                    )
                    winner = assignments[0] if assignments else None

                    planner_ref = (
                        winner.worker_id
                        if winner is not None
                        else _planner_worker_ref_from_env(
                            workers=workers, command_specs=command_specs
                        )
                    )
                    planner_meta = {
                        "selection": "market",
                        "task": plan_task.model_dump(mode="json"),
                        "bids": bid_records,
                        "winner": None
                        if winner is None
                        else {
                            "worker_id": winner.worker_id,
                            "bid": winner.bid.model_dump(mode="json"),
                            "score": round(float(winner.score), 6),
                            "expected_cost": round(float(winner.expected_cost), 6),
                        },
                        "planner_ref": planner_ref,
                    }

            planner_worker = resolve_worker_ref(planner_ref, workers=workers)
            if planner_worker is None:
                raise SystemExit(f"unknown planner worker ref: {planner_ref}")
            if planner_meta is None:
                planner_meta = {"selection": "unknown"}
            planner_meta["planner_worker_id"] = planner_worker.worker_id
            (run_dir / "plan_market.json").write_text(
                json.dumps(planner_meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            file_list = _list_repo_files(root=src)
            context_files = _planner_context_files(root=src, file_list=file_list)
            plan_call = decompose_with_worker(
                llm=llm,
                planner=planner_worker,
                command_specs=command_specs,
                goal=goal,
                max_tasks=int(args.max_tasks),
                file_list=file_list,
                allowed_paths=allowed_paths,
                context_files=context_files,
                cwd=workspace_dir,
            )
            (run_dir / "plan_raw.txt").write_text(plan_call.raw_text, encoding="utf-8")
            plan = plan_call.plan

            try:
                plan = toposort_plan(plan=plan)
            except ValueError as e:
                (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                raise SystemExit(f"invalid plan: {e}") from e
            (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

            for planned in plan.tasks:
                desc = planned.description.strip()
                if goal:
                    desc = f"Overall goal:\n{goal}\n\nSubtask:\n{desc}".strip()
                accept_cmds = list(planned.acceptance) if planned.acceptance else list(args.accept)
                hidden_cmds = list(args.hidden_accept)
                accept_hints = _acceptance_files_hint(
                    root=workspace_dir, commands=[*accept_cmds, *hidden_cmds]
                )
                files_hint = list(
                    dict.fromkeys([*planned.files_hint, *accept_hints, *list(args.files_hint)])
                )
                if verify_mode == VerifyMode.COMMANDS and (not accept_cmds) and (not hidden_cmds):
                    raise SystemExit(
                        f"planned task {planned.id} is missing acceptance commands "
                        "(provide --accept/--hidden-accept or have the planner emit per-task acceptance)"
                    )
                tasks.append(
                    TaskSpec(
                        id=str(planned.id),
                        title=str(planned.title),
                        description=desc,
                        deps=[str(d) for d in planned.deps],
                        bounty=int(args.bounty),
                        verify_mode=verify_mode,
                        submission_kind=submission_kind,
                        acceptance=[CommandSpec(cmd=str(c)) for c in accept_cmds],
                        hidden_acceptance=[CommandSpec(cmd=str(c)) for c in hidden_cmds],
                        judges=judges,
                        allowed_paths=[str(p) for p in allowed_paths],
                        files_hint=[str(p) for p in files_hint],
                    )
                )
        else:
            task_id = str(args.task_id)
            title = str(args.title or task_id)
            description = str(args.description or "")
            if not description.strip() and args.goal:
                description = str(args.goal)
            tasks.append(
                TaskSpec(
                    id=task_id,
                    title=title,
                    description=description,
                    deps=[],
                    bounty=int(args.bounty),
                    verify_mode=verify_mode,
                    submission_kind=submission_kind,
                    acceptance=[CommandSpec(cmd=str(c)) for c in list(args.accept)],
                    hidden_acceptance=[CommandSpec(cmd=str(c)) for c in list(args.hidden_accept)],
                    judges=judges,
                    allowed_paths=[str(p) for p in allowed_paths],
                    files_hint=[str(p) for p in list(args.files_hint)],
                )
            )

        ledger = HashChainedLedger(run_dir / "ledger.jsonl")
        engine_settings = _engine_settings(max_concurrency=int(args.concurrency))
        engine = ClearinghouseEngine(ledger=ledger, settings=engine_settings)
        engine.create_run(
            run_id=run_id,
            payment_rule=PaymentRule(args.payment_rule),
            workers=workers,
            tasks=tasks,
        )

        config = {
            "run_id": run_id,
            "scenario_path": None,
            "workspace_dir": str(workspace_dir.resolve()),
            "payment_rule": args.payment_rule,
            "workers": [w.model_dump() for w in workers],
            "tasks": [t.model_dump() for t in tasks],
            "decompose": bool(args.decompose),
            "goal": args.goal,
            "planner": planner_meta,
            "concurrency": int(engine_settings.max_concurrency),
            **cfg_extra,
        }
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        state = replay_ledger(events=list(ledger.iter_events()))
        model_bidder = (
            None
            if not has_model_agents
            else OpenAIBidder(
                llm=llm,
                payment_rule=state.payment_rule,
                max_bids=engine_settings.max_bids_per_worker,
            )
        )
        model_executor = (
            None
            if not has_model_agents
            else OpenAIExecutor(
                llm=llm,
                workspace_dir=workspace_dir,
                run_dir=run_dir,
                workers=workers,
                command_specs=command_specs,
                settings=ExecutorSettings(
                    judge_workers=judge_workers, judge_include_self=not args.no_self_judge
                ),
            )
        )

        ext_bidder = (
            None
            if not has_external
            else CommandBidder(
                workspace_dir=workspace_dir,
                payment_rule=state.payment_rule,
                specs=command_specs,
                max_bids=engine_settings.max_bids_per_worker,
            )
        )
        ext_executor = (
            None
            if not has_external
            else CommandExecutor(
                workspace_dir=workspace_dir,
                run_dir=run_dir,
                workers=workers,
                specs=command_specs,
                settings=CommandExecutorSettings(
                    judge_workers=judge_workers, judge_include_self=not args.no_self_judge
                ),
                llm=llm,
            )
        )

        bidder = MultiplexBidder(model_bidder=model_bidder, external_bidder=ext_bidder)
        executor = MultiplexExecutor(model_executor=model_executor, external_executor=ext_executor)
        cost_estimator = ExpectedCostEstimator(
            state=load_state(default_state_path()),
            pricing=load_pricing_from_env(),
        )

        _run_rounds(
            ledger=ledger,
            engine=engine,
            bidder=bidder,
            executor=executor,
            cost_estimator=cost_estimator,
            rounds=int(args.rounds),
        )
        _, state = _finalize_run(run_dir=run_dir, ledger=ledger)
        _print_report(state=state)
        _print_next_steps(run_dir=run_dir, workspace_dir=workspace_dir, state=state)
        return 0

    if args.cmd == "review":
        run_dir = Path(args.run_dir)
        ledger = HashChainedLedger(run_dir / "ledger.jsonl")
        ledger.verify_chain()
        events = list(ledger.iter_events())
        state = replay_ledger(events=events)
        task_specs = _load_task_specs_from_events(events=events)

        if args.review_cmd == "list":
            pending = [t for t in state.tasks.values() if t.status == "REVIEW"]
            if not pending:
                print("No tasks awaiting manual review.")
                return 0
            for t in sorted(pending, key=lambda x: x.task_id):
                wid = t.assigned_worker or "<unknown>"
                try:
                    _, patch = _find_review_events(events=events, task_id=t.task_id, worker_id=wid)
                    patch_path = _load_submission_artifact_path(run_dir=run_dir, patch_event=patch)
                    patch_rel = str(patch_path.relative_to(run_dir))
                except Exception as e:
                    patch_rel = f"<error: {e}>"
                title = task_specs.get(t.task_id).title if t.task_id in task_specs else t.task_id
                print(f"- {t.task_id}: {title} worker={wid} patch={patch_rel}")
            return 0

        task_id = str(args.task_id)
        rt = state.tasks.get(task_id)
        if rt is None:
            raise SystemExit(f"task not found: {task_id}")
        if rt.status != "REVIEW":
            raise SystemExit(f"task is not awaiting review: {task_id} (status={rt.status})")
        worker_id = rt.assigned_worker
        if not worker_id:
            raise SystemExit(f"missing assigned_worker for review task: {task_id}")

        completed, patch = _find_review_events(events=events, task_id=task_id, worker_id=worker_id)
        patch_path = _load_submission_artifact_path(run_dir=run_dir, patch_event=patch)
        spec = task_specs.get(task_id)
        if spec is None:
            raise SystemExit(f"missing TaskSpec for task_id={task_id}")

        cfg_path = run_dir / "run_config.json"
        if not cfg_path.exists():
            raise SystemExit(f"missing run_config.json in {run_dir}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        workspace_dir = Path(str(cfg.get("workspace_dir") or (run_dir / "workspace")))
        if not workspace_dir.exists():
            raise SystemExit(f"workspace_dir not found: {workspace_dir}")

        run_id = state.run_id
        round_id = state.round_id

        if args.review_cmd == "approve":
            if patch_path.name == "patch.diff":
                patch_text = patch_path.read_text(encoding="utf-8")
                changes = parse_patch_changes(patch_text)
                touched = sorted(
                    {p for ch in changes for p in [ch.old_path, ch.new_path] if p is not None}
                )
                enforce_allowed_paths(paths=touched, allowed=spec.allowed_paths)
                apply_unified_diff_path(patch_path=patch_path, cwd=workspace_dir)
            elif patch_path.name == "patch_files.json":
                files = json.loads(patch_path.read_text(encoding="utf-8"))
                if not isinstance(files, dict) or any(
                    not isinstance(k, str) or not isinstance(v, str) for k, v in files.items()
                ):
                    raise SystemExit("patch_files.json must be {path: full_contents}")
                touched = sorted(files.keys())
                enforce_allowed_paths(paths=touched, allowed=spec.allowed_paths)
                apply_file_blocks(files=files, cwd=workspace_dir)
            elif patch_path.name in {"submission.txt", "submission.json"}:
                # Non-patch submissions have no workspace edits to apply.
                pass
            else:
                raise SystemExit(f"unsupported submission artifact for review: {patch_path.name}")

            bid = (getattr(completed, "payload", {}) or {}).get("bid") or {}
            ask = int(bid.get("ask") or 0) if isinstance(bid, dict) else 0
            bounty = int(rt.bounty_current)
            amount = bounty if state.payment_rule == PaymentRule.BOUNTY else ask
            if amount <= 0:
                raise SystemExit("missing bid ask for manual approval settlement")

            ledger.append(
                EventType.VERIFICATION_PASSED,
                run_id=run_id,
                round_id=round_id,
                payload={
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "status": VerifyStatus.PASS.value,
                },
            )
            ledger.append(
                EventType.TASK_COMPLETED,
                run_id=run_id,
                round_id=round_id,
                payload={
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "success": True,
                    "verify_status": VerifyStatus.PASS.value,
                    "submission_kind": spec.submission_kind.value,
                    "bid": bid,
                    "bounty_current": bounty,
                },
            )
            ledger.append(
                EventType.PAYMENT_MADE,
                run_id=run_id,
                round_id=round_id,
                payload={
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "amount": amount,
                    "payment_rule": state.payment_rule.value,
                    "bounty": bounty,
                    "ask": ask,
                },
            )
            ledger.verify_chain()
            _, state2 = _finalize_run(run_dir=run_dir, ledger=ledger)
            _print_report(state=state2)
            return 0

        if args.review_cmd == "reject":
            if args.no_fault:
                status = VerifyStatus.INFRA
            else:
                status = VerifyStatus.FAIL

            ledger.append(
                EventType.VERIFICATION_FAILED,
                run_id=run_id,
                round_id=round_id,
                payload={"task_id": task_id, "worker_id": worker_id, "status": status.value},
            )
            ledger.append(
                EventType.TASK_COMPLETED,
                run_id=run_id,
                round_id=round_id,
                payload={
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "success": False,
                    "verify_status": status.value,
                    "submission_kind": spec.submission_kind.value,
                },
            )
            if status == VerifyStatus.FAIL:
                bid = (getattr(completed, "payload", {}) or {}).get("bid") or {}
                reported_p_success = 0.5
                if isinstance(bid, dict):
                    try:
                        reported_p_success = float(
                            bid.get("self_assessed_p_success")
                            or bid.get("p_success")
                            or bid.get("confidence")
                            or reported_p_success
                        )
                    except Exception:
                        reported_p_success = 0.5
                policy = SettlementPolicy()
                base_penalty = _penalty_amount(
                    bounty=int(rt.bounty_current),
                    policy=policy,
                )
                confidence_penalty = _confidence_penalty_amount(
                    base_penalty=base_penalty,
                    p_success=reported_p_success,
                    policy=policy,
                )
                penalty = int(base_penalty + confidence_penalty)
                ledger.append(
                    EventType.PENALTY_APPLIED,
                    run_id=run_id,
                    round_id=round_id,
                    payload={
                        "task_id": task_id,
                        "worker_id": worker_id,
                        "amount": penalty,
                        "reason": "verification_fail",
                        "base_penalty": base_penalty,
                        "confidence_penalty": confidence_penalty,
                        "reported_p_success": reported_p_success,
                        "confidence_penalty_floor": float(policy.confidence_penalty_floor),
                    },
                )

                state_mid = replay_ledger(events=list(ledger.iter_events()))
                t2 = state_mid.tasks[task_id]
                if t2.fail_count > 0 and t2.fail_count % 2 == 0:
                    new_bounty = min(
                        t2.bounty_original * 2,
                        max(t2.bounty_current + 1, round(t2.bounty_current * 1.10)),
                    )
                    if new_bounty != t2.bounty_current:
                        ledger.append(
                            EventType.BOUNTY_ADJUSTED,
                            run_id=run_id,
                            round_id=round_id,
                            payload={
                                "task_id": task_id,
                                "bounty_current": new_bounty,
                                "reason": "repeated_failures",
                            },
                        )

            ledger.verify_chain()
            _, state2 = _finalize_run(run_dir=run_dir, ledger=ledger)
            _print_report(state=state2)
            return 0

        raise AssertionError(f"unhandled review_cmd: {args.review_cmd}")

    if args.cmd == "report":
        run_dir = Path(args.run_dir)
        ledger = HashChainedLedger(run_dir / "ledger.jsonl")
        events = list(ledger.iter_events())
        state = replay_ledger(events=events)

        _print_report(state=state)
        return 0

    raise AssertionError(f"unhandled cmd: {args.cmd}")
