from __future__ import annotations

from compilerlite.ast import Program
from compilerlite.bytecode import BytecodeProgram


def compile_program(*, program: Program, optimize: bool = False) -> BytecodeProgram:
    _ = optimize
    raise NotImplementedError
