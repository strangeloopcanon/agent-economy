from __future__ import annotations

from collections.abc import Iterable

from snakelite.types import Direction, GameState, Point


class SnakeGame:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        snake: Iterable[Point],
        direction: Direction,
        food: Point | None,
        food_queue: Iterable[Point] | None = None,
        no_reverse: bool = True,
    ) -> None:
        raise NotImplementedError

    @property
    def state(self) -> GameState:
        raise NotImplementedError

    def step(self, action: Direction | None) -> GameState:
        raise NotImplementedError
