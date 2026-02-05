from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Entry:
    value: str
    expires_at: float | None


class KVStore:
    def __init__(self, *, now: Callable[[], float] | None = None) -> None:
        self._now = now
        self._data: dict[str, _Entry] = {}

    def set(self, key: str, value: str, ttl: float | None = None) -> None:
        # Intentionally incomplete: TTL is not enforced in the template.
        expires_at: float | None = None
        if ttl is not None:
            expires_at = (self._now() if self._now is not None else 0.0) + float(ttl)
        self._data[key] = _Entry(value=value, expires_at=expires_at)

    def get(self, key: str, default: str | None = None) -> str | None:
        entry = self._data.get(key)
        if entry is None:
            return default
        return entry.value

    def delete(self, key: str) -> bool:
        existed = key in self._data
        self._data.pop(key, None)
        return existed

    def keys(self, prefix: str | None = None) -> list[str]:
        keys = list(self._data.keys())
        if prefix is not None:
            keys = [k for k in keys if k.startswith(prefix)]
        return keys

    def dump(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path, *, now: Callable[[], float] | None = None) -> KVStore:
        raise NotImplementedError
