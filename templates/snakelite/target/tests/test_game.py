from __future__ import annotations

from snakelite.game import SnakeGame
from snakelite.types import Direction, Point


def test_step_moves_and_collides_wall() -> None:
    g = SnakeGame(
        width=4,
        height=3,
        snake=[Point(2, 1), Point(1, 1)],
        direction=Direction.RIGHT,
        food=None,
    )

    s1 = g.step(Direction.RIGHT)
    assert s1.alive is True
    assert s1.snake[0] == Point(3, 1)

    s2 = g.step(Direction.RIGHT)
    assert s2.alive is False
    # On death, snake does not move for that step.
    assert s2.snake[0] == Point(3, 1)


def test_reverse_is_ignored_when_no_reverse() -> None:
    g = SnakeGame(
        width=6,
        height=3,
        snake=[Point(2, 1), Point(1, 1), Point(0, 1)],
        direction=Direction.RIGHT,
        food=None,
        no_reverse=True,
    )
    s1 = g.step(Direction.LEFT)
    assert s1.direction == Direction.RIGHT
    assert s1.snake[0] == Point(3, 1)


def test_eat_food_grows_and_spawns_next_food_from_queue() -> None:
    g = SnakeGame(
        width=6,
        height=4,
        snake=[Point(2, 1), Point(1, 1), Point(0, 1)],
        direction=Direction.RIGHT,
        food=Point(3, 1),
        food_queue=[Point(5, 3)],
    )
    s1 = g.step(Direction.RIGHT)
    assert s1.alive is True
    assert s1.score == 1
    assert len(s1.snake) == 4
    assert s1.food == Point(5, 3)


def test_self_collision_ends_game_without_moving() -> None:
    g = SnakeGame(
        width=5,
        height=5,
        snake=[Point(2, 2), Point(2, 3), Point(1, 3), Point(1, 2), Point(1, 1), Point(2, 1)],
        direction=Direction.UP,
        food=None,
    )
    # Moving up would hit (2,1) which is body.
    s1 = g.step(Direction.UP)
    assert s1.alive is False
    assert s1.snake[0] == Point(2, 2)
