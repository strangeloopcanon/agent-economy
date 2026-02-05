from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from institution_service.jsonutil import stable_json_dumps
from institution_service.schemas import ArtifactRef, EventType, LedgerEvent


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
        payload = payload or {}
        artifacts = artifacts or []
        ts = ts or datetime.now(tz=UTC)
        event_id = event_id or str(uuid.uuid4())

        event_hash = _compute_event_hash(
            schema_version=schema_version,
            event_id=event_id,
            prev_hash=self._tail_hash,
            ts=ts,
            run_id=run_id,
            round_id=round_id,
            event_type=event_type,
            payload=payload,
            artifacts=artifacts,
        )
        event = LedgerEvent(
            schema_version=schema_version,
            event_id=event_id,
            prev_hash=self._tail_hash,
            hash=event_hash,
            ts=ts,
            run_id=run_id,
            round_id=round_id,
            type=event_type,
            payload=payload,
            artifacts=artifacts,
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
        prev_hash: str | None = None
        for event in self.iter_events():
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
