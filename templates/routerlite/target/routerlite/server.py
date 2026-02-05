from __future__ import annotations

from routerlite.request import parse_request
from routerlite.response import http_response
from routerlite.router import Router


def handle_request(*, raw_request: str, router: Router) -> str:
    _ = (parse_request, http_response)
    raise NotImplementedError
