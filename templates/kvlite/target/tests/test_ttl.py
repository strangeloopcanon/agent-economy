from __future__ import annotations

from kvlite import KVStore


def test_ttl_expires_on_get(clock, now):
    store = KVStore(now=now)
    store.set("k", "v", ttl=1.5)
    assert store.get("k") == "v"
    clock.advance(1.5)
    assert store.get("k") is None


def test_keys_lazily_purges_expired(clock, now):
    store = KVStore(now=now)
    store.set("k1", "v1", ttl=1.0)
    store.set("k2", "v2")
    clock.advance(2.0)
    assert store.keys() == ["k2"]


def test_reset_ttl(clock, now):
    store = KVStore(now=now)
    store.set("k", "v1", ttl=5.0)
    clock.advance(4.0)
    store.set("k", "v2", ttl=5.0)
    clock.advance(4.0)
    assert store.get("k") == "v2"
