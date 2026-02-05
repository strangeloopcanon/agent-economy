from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Instruction:
    op: str
    arg: int | str | None = None


@dataclass(frozen=True, slots=True)
class BytecodeProgram:
    instructions: list[Instruction]

    def run(self) -> list[str]:
        raise NotImplementedError
