from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"

    @classmethod
    def from_str(cls, raw: str) -> "Direction":
        raw = str(raw).strip().upper()
        try:
            return Direction(raw)
        except Exception as e:  # pragma: no cover
            raise ValueError(f"invalid direction: {raw!r}") from e


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class GameState:
    width: int
    height: int
    snake: tuple[Point, ...]  # head first
    direction: Direction
    food: Point | None
    score: int
    steps: int
    alive: bool = True
