from __future__ import annotations

from compilerlite.bytecode import BytecodeProgram
from compilerlite.compiler import compile_program
from compilerlite.parser import parse_program


def compile_source(*, src: str, optimize: bool = False) -> BytecodeProgram:
    program = parse_program(src)
    return compile_program(program=program, optimize=optimize)


def run_source(*, src: str, optimize: bool = False) -> list[str]:
    program = compile_source(src=src, optimize=optimize)
    return program.run()
