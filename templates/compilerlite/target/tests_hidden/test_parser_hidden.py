from __future__ import annotations

from compilerlite.parser import parse_program
from compilerlite.sexpr import to_sexpr


def test_nested_parens_and_precedence():
    program = parse_program("print (1 + 2) * (3 + 4);\n")
    assert to_sexpr(program) == "(program (print (mul (add 1 2) (add 3 4))))"
