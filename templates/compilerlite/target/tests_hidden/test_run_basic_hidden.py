from __future__ import annotations

import pytest

from compilerlite.api import run_source
from compilerlite.errors import VMError


def test_large_expression():
    out = run_source(src="print 1+2+3+4+5;\n")
    assert out == ["15"]


def test_mod_by_zero_raises():
    with pytest.raises(VMError):
        run_source(src="print 1 % 0;")
