from __future__ import annotations


class CompileError(Exception):
    def __init__(self, message: str, *, line: int | None = None, col: int | None = None) -> None:
        self.line = line
        self.col = col
        prefix = ""
        if line is not None and col is not None:
            prefix = f"line {line} col {col}: "
        super().__init__(prefix + str(message))


class VMError(Exception):
    pass
