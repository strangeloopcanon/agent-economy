from __future__ import annotations

from routerlite import Router, handle_request
from routerlite.request import Request


def test_handle_request_happy_path() -> None:
    router = Router()

    def hello(_req: Request, name: str) -> tuple[int, str]:
        return 200, f"Hello {name}"

    router.add(method="GET", pattern="/hello/:name", handler=hello)

    resp = handle_request(raw_request="GET /hello/Ada HTTP/1.1\r\n\r\n", router=router)
    assert resp.endswith("\r\n\r\nHello Ada")
    assert "Content-Length: 9\r\n" in resp


def test_handle_request_404() -> None:
    router = Router()
    resp = handle_request(raw_request="GET /missing HTTP/1.1\r\n\r\n", router=router)
    assert resp.startswith("HTTP/1.1 404 ")
