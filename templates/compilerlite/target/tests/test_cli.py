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


def test_cli_run(tmp_path: Path):
    src = "print 1 + 2 * 3;\n"
    p = tmp_path / "prog.cl"
    p.write_text(src, encoding="utf-8")
    cwd = Path.cwd() / "target"

    proc = _run(["run", str(p)], cwd=cwd)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines() == ["7"]


def test_cli_opt(tmp_path: Path):
    src = "print 1 + 2 * 3;\n"
    p = tmp_path / "prog.cl"
    p.write_text(src, encoding="utf-8")
    cwd = Path.cwd() / "target"

    proc = _run(["run", str(p), "--opt"], cwd=cwd)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines() == ["7"]
