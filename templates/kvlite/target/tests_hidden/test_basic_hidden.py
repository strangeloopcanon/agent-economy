# ruff: noqa: RUF001

from __future__ import annotations

from kvlite import KVStore


def test_keys_sorted_prefix_unicode(now):
    store = KVStore(now=now)
    store.set("β", "2")
    store.set("α", "1")
    store.set("a ", "3")
    assert store.keys() == ["a ", "α", "β"]
    assert store.keys(prefix="a") == ["a "]
