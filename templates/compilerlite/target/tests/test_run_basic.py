from __future__ import annotations

import pytest

from compilerlite.errors import VMError
from compilerlite.api import run_source


def test_run_print_and_vars():
    out = run_source(src="let x = 1 + 2 * 3; print x; print x + 1;")
    assert out == ["7", "8"]


def test_division_and_mod_trunc_toward_zero():
    out = run_source(src="print -7 / 2; print -7 % 2;")
    assert out == ["-3", "-1"]


def test_undefined_var_raises():
    with pytest.raises(VMError):
        run_source(src="print x;")
