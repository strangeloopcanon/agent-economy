# Cross-Repo Integration Guide

Use this when you want to consume `agent-economy` from a different repository.

For early adoption, the best path is a pinned git dependency. It is fast to set up, reproducible, and requires no packaging pipeline.

## Quick answer

- Install as a git dependency in the consuming repo.
- Pin to a commit or tag (do not track `main` in production).
- Use one of three surfaces:
  - CLI (`agent-economy ...`)
  - Engine API (`ClearinghouseEngine`)
  - RL wrapper (`InstitutionEnv`)

## Copy-paste handoff for another LLM

```markdown
# agent-economy integration playbook

## Goal
Use `agent-economy` from another repository with reproducible setup.

## Install in consuming repo

### Recommended (`uv add`, pinned commit)
```bash
uv add "agent-economy @ git+https://github.com/strangeloopcanon/agent-economy.git@aa6c785867305efe810c63a92d81dc8fb7115937"
```

### Alternative (`pyproject.toml`)
```toml
[project]
dependencies = [
  "agent-economy @ git+https://github.com/strangeloopcanon/agent-economy.git@aa6c785867305efe810c63a92d81dc8fb7115937",
]
```

### Local co-development
```bash
uv pip install -e /path/to/agent-economy
```

---

## Surface A: CLI orchestration

```bash
uv run agent-economy oneshot "Add hello.txt containing Hello" \
  --workspace-src . \
  --allowed-path . \
  --accept "grep -q 'Hello' hello.txt" \
  --rounds 3 \
  --concurrency 1
```

Useful commands:
- `uv run agent-economy config validate`
- `uv run agent-economy task "<goal>" ...`
- `uv run agent-economy report --run-dir <run_dir>`
- `uv run agent-economy dashboard --run-dir <run_dir>`

For non-code tasks, set `--submission-kind text` or `--submission-kind json`.

---

## Surface B: Engine API (programmatic)

```python
from agent_economy import (
    ClearinghouseEngine,
    EngineSettings,
    InMemoryLedger,
    TaskSpec,
    CommandSpec,
    WorkerRuntime,
    PaymentRule,
    Bid,
    BidResult,
    ExecutionOutcome,
    VerifyStatus,
    replay_ledger,
)

class FixedBidder:
    def get_bids(self, *, worker, ready_tasks, round_id, discussion_history):
        if not ready_tasks:
            return BidResult(bids=[])
        task_id = ready_tasks[0].spec.id
        return BidResult(
            bids=[Bid(task_id=task_id, ask=5, self_assessed_p_success=0.8, eta_minutes=10)]
        )

class AlwaysPassExecutor:
    def execute(self, *, worker, task, bid, round_id, discussion_history):
        return ExecutionOutcome(status=VerifyStatus.PASS, notes="ok")

ledger = InMemoryLedger()
engine = ClearinghouseEngine(
    ledger=ledger,
    settings=EngineSettings(max_concurrency=1, deterministic=True),
)

tasks = [
    TaskSpec(
        id="T1",
        title="Create hello file",
        bounty=20,
        deps=[],
        acceptance=[CommandSpec(cmd="test -f hello.txt")],
    )
]
workers = [WorkerRuntime(worker_id="w1", reputation=1.0)]

engine.create_run(
    run_id="demo-run",
    payment_rule=PaymentRule.ASK,
    workers=workers,
    tasks=tasks,
)

for _ in range(10):
    engine.step(bidder=FixedBidder(), executor=AlwaysPassExecutor())
    state = replay_ledger(events=list(ledger.iter_events()))
    if all(t.status == "DONE" for t in state.tasks.values()):
        break

print("done:", state.tasks["T1"].status, "balance:", state.workers["w1"].balance)
```

---

## Surface C: RL environment (`InstitutionEnv`)

```python
from agent_economy import (
    InstitutionEnv,
    TaskSpec,
    CommandSpec,
    WorkerRuntime,
    ExecutionOutcome,
    VerifyStatus,
)

class AlwaysPassExecutor:
    def execute(self, *, worker, task, bid, round_id, discussion_history):
        return ExecutionOutcome(status=VerifyStatus.PASS, notes="ok")

env = InstitutionEnv(
    tasks=[TaskSpec(id="T1", title="Task", bounty=20, deps=[], acceptance=[CommandSpec(cmd="true")])],
    workers=[WorkerRuntime(worker_id="rl-agent", reputation=1.0)],
    agent_id="rl-agent",
    executor=AlwaysPassExecutor(),
    max_rounds=20,
)

obs, info = env.reset()
action = {"task_id": "T1", "ask": 5, "p_success": 0.9, "eta_minutes": 10}
obs, reward, terminated, truncated, info = env.step(action)
print(reward, terminated, truncated, info)
```

---

## Upgrade workflow

```bash
uv add "agent-economy @ git+https://github.com/strangeloopcanon/agent-economy.git@<new-tag-or-commit>"
uv lock
uv run pytest -q
```
```

## Notes

- Public import surface is exposed from `agent_economy/__init__.py`.
- CLI entrypoint is `agent-economy` from `pyproject.toml` scripts.
- For local-model usage (Ollama/Qwen), keep worker definitions in your consuming repo and point `model_ref` to the provider-specific reference.
