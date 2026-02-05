from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from snakelite.types import Direction, GameState, Point


@dataclass(frozen=True, slots=True)
class Replay:
    width: int
    height: int
    snake: list[Point]
    direction: Direction
    food: Point | None
    food_queue: list[Point]
    moves: list[Direction | None]


def load_replay(path: str | Path) -> Replay:
    raise NotImplementedError


def run_replay(replay: Replay) -> GameState:
    raise NotImplementedError
