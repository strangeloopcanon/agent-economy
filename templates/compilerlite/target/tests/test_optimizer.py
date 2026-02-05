from __future__ import annotations

from compilerlite.api import compile_source, run_source


def test_constant_folding_reduces_instructions():
    src = "print 1 + 2 * 3;"
    prog0 = compile_source(src=src, optimize=False)
    prog1 = compile_source(src=src, optimize=True)
    assert len(prog1.instructions) < len(prog0.instructions)
    assert run_source(src=src, optimize=True) == ["7"]
