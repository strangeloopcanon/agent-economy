from __future__ import annotations

import json
from pathlib import Path

from kvlite import KVStore


def test_dump_is_json_object(tmp_path: Path, now):
    path = tmp_path / "db.json"
    store = KVStore(now=now)
    store.set("k", "v")
    store.dump(path)
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict)
    assert "items" in payload
