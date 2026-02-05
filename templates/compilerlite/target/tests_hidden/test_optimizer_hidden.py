from __future__ import annotations

from compilerlite.api import compile_source, run_source


def test_optimizer_folds_comparisons_and_preserves_output():
    src = "print 1 < 2; print 2 < 1;"
    prog0 = compile_source(src=src, optimize=False)
    prog1 = compile_source(src=src, optimize=True)
    assert len(prog1.instructions) < len(prog0.instructions)
    assert run_source(src=src, optimize=True) == ["1", "0"]
