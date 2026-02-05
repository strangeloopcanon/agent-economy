from __future__ import annotations

from snakelite.render import render_ascii
from snakelite.types import Direction, GameState, Point


def test_render_ascii_marks_head_body_food() -> None:
    state = GameState(
        width=4,
        height=3,
        snake=(Point(1, 1), Point(0, 1)),
        direction=Direction.RIGHT,
        food=Point(3, 0),
        score=0,
        steps=0,
        alive=True,
    )
    out = render_ascii(state)
    assert out.endswith("\n")
    lines = out.splitlines()
    assert lines[0] == "...*"
    assert lines[1] == "oH.."
