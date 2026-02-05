from __future__ import annotations

from kvlite import KVStore


def test_ttl_zero_expires_immediately(clock, now):
    store = KVStore(now=now)
    store.set("k", "v", ttl=0.0)
    assert store.get("k") is None
    assert store.keys() == []


def test_get_default_after_expiry(clock, now):
    store = KVStore(now=now)
    store.set("k", "v", ttl=1.0)
    clock.advance(2.0)
    assert store.get("k", default="d") == "d"
