from __future__ import annotations

from kvlite import KVStore


def test_set_get_delete_keys_sorted(now):
    store = KVStore(now=now)
    assert store.get("missing") is None
    assert store.get("missing", default="x") == "x"

    store.set("b", "2")
    store.set("a", "1")
    assert store.get("a") == "1"
    assert store.get("b") == "2"

    assert store.keys() == ["a", "b"]
    assert store.keys(prefix="a") == ["a"]

    assert store.delete("a") is True
    assert store.get("a") is None
    assert store.delete("a") is False
