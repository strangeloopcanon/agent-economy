from __future__ import annotations

import pytest

from compilerlite.api import run_source
from compilerlite.errors import CompileError, VMError


def test_compile_error_has_location():
    with pytest.raises(CompileError) as exc:
        run_source(src="let x = 1")
    assert exc.value.line is not None
    assert exc.value.col is not None
    s = str(exc.value)
    assert "line" in s and "col" in s


def test_runtime_div_by_zero():
    with pytest.raises(VMError):
        run_source(src="print 1 / 0;")
