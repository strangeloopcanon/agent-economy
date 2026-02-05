from __future__ import annotations

from routerlite.request import Request
from routerlite.router import Router


def test_router_static_match() -> None:
    router = Router()

    def hello(_req: Request) -> tuple[int, str]:
        return 200, "ok"

    router.add(method="GET", pattern="/hello", handler=hello)

    handler, params = router.match(method="GET", path="/hello")
    assert handler is hello
    assert params == {}

    handler2, _params2 = router.match(method="POST", path="/hello")
    assert handler2 is None


def test_router_param_match() -> None:
    router = Router()

    def user(_req: Request, id: str) -> tuple[int, str]:
        return 200, id

    router.add(method="GET", pattern="/users/:id", handler=user)

    handler, params = router.match(method="GET", path="/users/123")
    assert handler is user
    assert params == {"id": "123"}

    handler2, params2 = router.match(method="GET", path="/users")
    assert handler2 is None
    assert params2 == {}
