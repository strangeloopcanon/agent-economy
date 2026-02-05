from __future__ import annotations

from routerlite.request import parse_request


def test_parse_request_basic() -> None:
    req = parse_request("GET /hello HTTP/1.1\r\nHost: example.com\r\nX-Test:  1 \r\n\r\n")
    assert req.method == "GET"
    assert req.path == "/hello"
    assert req.body == ""
    assert req.headers == {"host": "example.com", "x-test": "1"}


def test_parse_request_body() -> None:
    req = parse_request("POST /submit HTTP/1.1\r\nContent-Type: text/plain\r\n\r\nhi\n")
    assert req.method == "POST"
    assert req.path == "/submit"
    assert req.headers["content-type"] == "text/plain"
    assert req.body == "hi\n"
