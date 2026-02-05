from __future__ import annotations

import pytest

from compilerlite.errors import CompileError
from compilerlite.lexer import tokenize


def test_comments_are_ignored():
    toks = tokenize("let foo_bar = 12; # comment\nprint foo_bar;\n")
    kinds = [t.kind for t in toks]
    assert "KW_LET" in kinds
    assert "KW_PRINT" in kinds
    assert kinds[-1] == "EOF"


def test_error_includes_line_and_col():
    with pytest.raises(CompileError) as exc:
        tokenize("let x = 1 @ 2;")
    assert exc.value.line is not None
    assert exc.value.col is not None
