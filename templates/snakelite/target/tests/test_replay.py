from __future__ import annotations

import json
from pathlib import Path

import pytest

from snakelite.replay import load_replay, run_replay
from snakelite.types import Point


def test_load_replay_parses_points_and_moves(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(
        json.dumps(
            {
                "width": 4,
                "height": 3,
                "snake": [[1, 1], [0, 1]],
                "direction": "R",
                "food": [3, 1],
                "food_queue": [],
                "moves": ["R", None, "D"],
            }
        ),
        encoding="utf-8",
    )

    r = load_replay(p)
    assert r.width == 4
    assert r.height == 3
    assert r.snake[0] == Point(1, 1)
    assert r.moves[1] is None


def test_load_replay_validates_invariants(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(
        json.dumps(
            {
                "width": 4,
                "height": 3,
                "snake": [],
                "direction": "R",
                "food_queue": [],
                "moves": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_replay(p)


def test_run_replay_returns_final_state(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(
        json.dumps(
            {
                "width": 5,
                "height": 3,
                "snake": [[2, 1], [1, 1]],
                "direction": "R",
                "food": [4, 1],
                "food_queue": [],
                "moves": ["R", "R"],
            }
        ),
        encoding="utf-8",
    )
    r = load_replay(p)
    final = run_replay(r)
    assert final.score == 1
    assert final.alive is True
