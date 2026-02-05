from __future__ import annotations

from collections.abc import Sequence

from institution_service.schemas import WorkerRuntime


def resolve_worker_ref(ref: str, *, workers: Sequence[WorkerRuntime]) -> WorkerRuntime | None:
    """
    Resolve a worker reference to a registered worker.

    "ref" may be either:
    - a worker_id (preferred), or
    - a model_ref (back-compat / convenience).
    """

    ref = str(ref or "").strip()
    if not ref:
        return None

    for w in workers:
        if w.worker_id == ref:
            return w

    for w in workers:
        if w.model_ref and w.model_ref == ref:
            return w

    return None


def resolve_worker_refs(
    refs: Sequence[str], *, workers: Sequence[WorkerRuntime]
) -> list[WorkerRuntime]:
    out: list[WorkerRuntime] = []
    seen: set[str] = set()
    for ref in refs:
        w = resolve_worker_ref(str(ref), workers=workers)
        if w is None:
            continue
        if w.worker_id in seen:
            continue
        seen.add(w.worker_id)
        out.append(w)
    return out
