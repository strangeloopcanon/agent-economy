from __future__ import annotations

from collections.abc import Callable

from routerlite.request import Request

Handler = Callable[..., tuple[int, str]]


class Router:
    def __init__(self) -> None:
        raise NotImplementedError

    def add(self, *, method: str, pattern: str, handler: Handler) -> None:
        raise NotImplementedError

    def match(self, *, method: str, path: str) -> tuple[Handler | None, dict[str, str]]:
        raise NotImplementedError

    def dispatch(self, *, request: Request) -> tuple[int, str]:
        handler, params = self.match(method=request.method, path=request.path)
        if handler is None:
            return 404, "not found"
        return handler(request, **params)
