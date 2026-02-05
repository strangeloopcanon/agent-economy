from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    text: str
    line: int
    col: int


def tokenize(src: str) -> list[Token]:
    raise NotImplementedError
