from __future__ import annotations

import pytest

from compilerlite.errors import CompileError
from compilerlite.parser import parse_program
from compilerlite.sexpr import to_sexpr


def test_parse_let_print_expr_precedence():
    program = parse_program("let x = 1 + 2 * 3;\nprint x;\n")
    assert to_sexpr(program) == "(program (let x (add 1 (mul 2 3))) (print (var x)))"


def test_parse_unary_minus_and_parens():
    program = parse_program("print -(1 + 2);\n")
    assert to_sexpr(program) == "(program (print (neg (add 1 2))))"


def test_parse_error():
    with pytest.raises(CompileError):
        parse_program("let x = ;")
