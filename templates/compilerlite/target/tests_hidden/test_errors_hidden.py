from __future__ import annotations

import pytest

from compilerlite.errors import CompileError
from compilerlite.parser import parse_program


def test_unexpected_token_has_location():
    with pytest.raises(CompileError) as exc:
        parse_program("print (1 + 2;\n")
    assert exc.value.line is not None
    assert exc.value.col is not None
