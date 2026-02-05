from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Request:
    method: str
    path: str
    headers: dict[str, str]
    body: str = ""

    def header(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name.lower(), default)


def parse_request(raw: str) -> Request:
    raise NotImplementedError
