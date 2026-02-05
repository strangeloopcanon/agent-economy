from __future__ import annotations


def http_response(*, status: int, body: str = "", headers: dict[str, str] | None = None) -> str:
    raise NotImplementedError
