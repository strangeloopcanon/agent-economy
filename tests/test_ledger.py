from __future__ import annotations

import json

import pytest
from agent_economy.jsonutil import stable_json_dumps
from agent_economy.ledger import HashChainedLedger
from agent_economy.schemas import EventType


def test_hash_chained_ledger_verifies_and_detects_tampering(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = HashChainedLedger(ledger_path)
    run_id = "run-1"

    ledger.append(EventType.RUN_CREATED, run_id=run_id, round_id=0, payload={"payment_rule": "ask"})
    ledger.append(
        EventType.TASK_CREATED, run_id=run_id, round_id=0, payload={"task_id": "T1", "bounty": 10}
    )
    ledger.verify_chain()

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    event2 = json.loads(lines[1])
    event2["payload"]["bounty"] = 999  # tamper without recomputing hash
    lines[1] = stable_json_dumps(event2)
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        HashChainedLedger(ledger_path).verify_chain()
