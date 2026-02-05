from __future__ import annotations

from compilerlite.bytecode import BytecodeProgram


def run_program(program: BytecodeProgram) -> list[str]:
    return program.run()
