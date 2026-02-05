from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, "-m", "kvlite", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_roundtrip(tmp_path: Path):
    db = tmp_path / "db.json"
    cwd = Path.cwd() / "target"

    p = _run(["set", "a", "1", "--db", str(db)], cwd=cwd)
    assert p.returncode == 0, p.stderr

    p = _run(["get", "a", "--db", str(db)], cwd=cwd)
    assert p.returncode == 0
    assert p.stdout.strip() == "1"

    p = _run(["keys", "--db", str(db)], cwd=cwd)
    assert p.returncode == 0
    assert p.stdout.splitlines() == ["a"]

    p = _run(["delete", "a", "--db", str(db)], cwd=cwd)
    assert p.returncode == 0

    p = _run(["get", "a", "--db", str(db)], cwd=cwd)
    assert p.returncode == 1
