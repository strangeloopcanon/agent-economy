from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest


@dataclass
class FakeClock:
    t: float = 1000.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def now(clock: FakeClock) -> Callable[[], float]:
    return clock.now
