from __future__ import annotations

import pytest

from compilerlite.errors import CompileError
from compilerlite.lexer import tokenize


def _kinds(src: str) -> list[tuple[str, str]]:
    return [(t.kind, t.text) for t in tokenize(src)]


def test_tokens_basic():
    toks = _kinds("let x = 1+2;\nprint x;\n")
    assert toks[:5] == [
        ("KW_LET", "let"),
        ("IDENT", "x"),
        ("EQ", "="),
        ("INT", "1"),
        ("PLUS", "+"),
    ]
    assert toks[-3:] == [("IDENT", "x"), ("SEMICOLON", ";"), ("EOF", "")]


def test_tokens_locations():
    toks = tokenize("let x=1;\nprint x;\n")
    kw_print = next(t for t in toks if t.kind == "KW_PRINT")
    assert (kw_print.line, kw_print.col) == (2, 1)


def test_unknown_char_raises():
    with pytest.raises(CompileError) as e:
        tokenize("let x = 1 @ 2;")
    assert "line" in str(e.value)
