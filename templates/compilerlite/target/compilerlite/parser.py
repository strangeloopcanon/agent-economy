from __future__ import annotations

from compilerlite.ast import Program


def parse_program(src: str) -> Program:
    raise NotImplementedError


def parse_source(src: str) -> Program:
    return parse_program(src)
