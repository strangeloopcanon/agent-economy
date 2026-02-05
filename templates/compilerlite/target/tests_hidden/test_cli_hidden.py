from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, "-m", "compilerlite", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_compile_error_exit_code(tmp_path: Path):
    p = tmp_path / "bad.cl"
    p.write_text("let x = ;\n", encoding="utf-8")
    cwd = Path.cwd() / "target"

    proc = _run(["run", str(p)], cwd=cwd)
    assert proc.returncode == 2


def test_cli_runtime_error_exit_code(tmp_path: Path):
    p = tmp_path / "bad.cl"
    p.write_text("print 1 / 0;\n", encoding="utf-8")
    cwd = Path.cwd() / "target"

    proc = _run(["run", str(p)], cwd=cwd)
    assert proc.returncode == 3
