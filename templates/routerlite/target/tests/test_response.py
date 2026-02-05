from __future__ import annotations

from routerlite.response import http_response


def test_http_response_includes_content_length_and_crlf() -> None:
    out = http_response(status=200, body="hi")
    assert out.startswith("HTTP/1.1 200 ")
    assert "\r\n\r\nhi" in out
    assert "\n\n" not in out
    assert "Content-Length: 2\r\n" in out


def test_http_response_merges_headers() -> None:
    out = http_response(status=404, body="", headers={"X-Test": "1"})
    assert "X-Test: 1\r\n" in out
    assert "Content-Length: 0\r\n" in out
