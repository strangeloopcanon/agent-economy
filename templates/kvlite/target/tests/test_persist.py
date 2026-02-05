from __future__ import annotations

from pathlib import Path

from kvlite import KVStore


def test_dump_load_preserves_remaining_ttl(tmp_path: Path, clock, now):
    path = tmp_path / "db.json"
    store = KVStore(now=now)
    store.set("persist", "p")
    store.set("ttl", "t", ttl=10.0)
    clock.advance(3.0)

    store.dump(path)
    loaded = KVStore.load(path, now=now)

    assert loaded.get("persist") == "p"
    assert loaded.get("ttl") == "t"

    clock.advance(7.0)
    assert loaded.get("ttl") is None


def test_dump_skips_expired(tmp_path: Path, clock, now):
    path = tmp_path / "db.json"
    store = KVStore(now=now)
    store.set("alive", "a", ttl=10.0)
    store.set("dead", "d", ttl=1.0)
    clock.advance(2.0)
    store.dump(path)

    loaded = KVStore.load(path, now=now)
    assert loaded.get("dead") is None
    assert loaded.get("alive") == "a"
