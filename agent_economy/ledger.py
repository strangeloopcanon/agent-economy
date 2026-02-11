from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agent_economy.jsonutil import stable_json_dumps
from agent_economy.schemas import ArtifactRef, EventType, LedgerEvent


@runtime_checkable
class Ledger(Protocol):
    """Structural interface shared by all ledger backends."""

    def reset(self) -> None: ...

    def append(
        self,
        event_type: EventType,
        *,
        run_id: str,
        round_id: int,
        payload: dict[str, Any] | None = ...,
        artifacts: list[ArtifactRef] | None = ...,
        ts: datetime | None = ...,
        schema_version: int = ...,
        event_id: str | None = ...,
    ) -> LedgerEvent: ...

    def iter_events(self) -> Iterator[LedgerEvent]: ...

    def verify_chain(self) -> None: ...

    def __len__(self) -> int: ...


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compute_event_hash(
    *,
    schema_version: int,
    event_id: str,
    prev_hash: str | None,
    ts: datetime,
    run_id: str,
    round_id: int,
    event_type: EventType,
    payload: dict[str, Any],
    artifacts: list[ArtifactRef],
) -> str:
    to_hash = {
        "schema_version": schema_version,
        "event_id": event_id,
        "prev_hash": prev_hash,
        "ts": ts.isoformat(),
        "run_id": run_id,
        "round_id": round_id,
        "type": event_type.value,
        "payload": payload,
        "artifacts": [a.model_dump() for a in artifacts],
    }
    return _sha256_hex(stable_json_dumps(to_hash))


def _build_event(
    event_type: EventType,
    *,
    run_id: str,
    round_id: int,
    prev_hash: str | None,
    payload: dict[str, Any] | None = None,
    artifacts: list[ArtifactRef] | None = None,
    ts: datetime | None = None,
    schema_version: int = 1,
    event_id: str | None = None,
) -> LedgerEvent:
    """Construct a hash-chained LedgerEvent (shared by both ledger backends)."""
    payload = payload or {}
    artifacts = artifacts or []
    ts = ts or datetime.now(tz=UTC)
    event_id = event_id or str(uuid.uuid4())

    event_hash = _compute_event_hash(
        schema_version=schema_version,
        event_id=event_id,
        prev_hash=prev_hash,
        ts=ts,
        run_id=run_id,
        round_id=round_id,
        event_type=event_type,
        payload=payload,
        artifacts=artifacts,
    )
    return LedgerEvent(
        schema_version=schema_version,
        event_id=event_id,
        prev_hash=prev_hash,
        hash=event_hash,
        ts=ts,
        run_id=run_id,
        round_id=round_id,
        type=event_type,
        payload=payload,
        artifacts=artifacts,
    )


def _verify_event_chain(events: Iterable[LedgerEvent]) -> None:
    """Verify hash-chain integrity across a sequence of events."""
    prev_hash: str | None = None
    for event in events:
        expected = _compute_event_hash(
            schema_version=event.schema_version,
            event_id=event.event_id,
            prev_hash=prev_hash,
            ts=event.ts,
            run_id=event.run_id,
            round_id=event.round_id,
            event_type=event.type,
            payload=event.payload,
            artifacts=event.artifacts,
        )
        if event.prev_hash != prev_hash:
            raise ValueError("ledger prev_hash mismatch")
        if event.hash != expected:
            raise ValueError("ledger hash mismatch")
        prev_hash = event.hash


class HashChainedLedger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tail_hash: str | None = None

        if self._path.exists():
            self._tail_hash = self._read_last_hash()

    @property
    def path(self) -> Path:
        return self._path

    def reset(self) -> None:
        self._path.write_text("", encoding="utf-8")
        self._tail_hash = None

    def append(
        self,
        event_type: EventType,
        *,
        run_id: str,
        round_id: int,
        payload: dict[str, Any] | None = None,
        artifacts: list[ArtifactRef] | None = None,
        ts: datetime | None = None,
        schema_version: int = 1,
        event_id: str | None = None,
    ) -> LedgerEvent:
        event = _build_event(
            event_type,
            run_id=run_id,
            round_id=round_id,
            prev_hash=self._tail_hash,
            payload=payload,
            artifacts=artifacts,
            ts=ts,
            schema_version=schema_version,
            event_id=event_id,
        )
        with self._path.open("a", encoding="utf-8") as f:
            f.write(stable_json_dumps(event.model_dump(mode="json")))
            f.write("\n")
        self._tail_hash = event.hash
        return event

    def iter_events(self) -> Iterator[LedgerEvent]:
        if not self._path.exists():
            return iter(())
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield LedgerEvent.model_validate_json(line)

    def verify_chain(self) -> None:
        _verify_event_chain(self.iter_events())

    def __len__(self) -> int:
        if not self._path.exists():
            return 0
        count = 0
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _read_last_hash(self) -> str | None:
        try:
            with self._path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                read_size = min(size, 128 * 1024)
                f.seek(size - read_size)
                chunk = f.read(read_size).decode("utf-8", errors="ignore")
        except Exception:
            chunk = ""

        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            return None

        # If we didn't start at the beginning, the first line may be partial. Prefer the last line.
        last = lines[-1]
        try:
            event = LedgerEvent.model_validate_json(last)
            return event.hash
        except Exception:
            return None


class InMemoryLedger:
    """Drop-in replacement for HashChainedLedger backed by an in-memory list.

    Intended for RL episodes and tests where disk I/O is unnecessary overhead.
    Implements the same public interface as HashChainedLedger.
    """

    def __init__(self) -> None:
        self._events: list[LedgerEvent] = []
        self._tail_hash: str | None = None

    def reset(self) -> None:
        self._events.clear()
        self._tail_hash = None

    def append(
        self,
        event_type: EventType,
        *,
        run_id: str,
        round_id: int,
        payload: dict[str, Any] | None = None,
        artifacts: list[ArtifactRef] | None = None,
        ts: datetime | None = None,
        schema_version: int = 1,
        event_id: str | None = None,
    ) -> LedgerEvent:
        event = _build_event(
            event_type,
            run_id=run_id,
            round_id=round_id,
            prev_hash=self._tail_hash,
            payload=payload,
            artifacts=artifacts,
            ts=ts,
            schema_version=schema_version,
            event_id=event_id,
        )
        self._events.append(event)
        self._tail_hash = event.hash
        return event

    def iter_events(self) -> Iterator[LedgerEvent]:
        for event in self._events:
            yield event.model_copy(deep=True)

    def verify_chain(self) -> None:
        _verify_event_chain(self._events)

    def __len__(self) -> int:
        return len(self._events)
