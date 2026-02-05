from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_replay_smoke() -> None:
    env = dict(os.environ)
    cwd = Path.cwd() / "target"
    replay = cwd / "tests" / "replays" / "smoke.json"

    proc = subprocess.run(
        [sys.executable, "-m", "snakelite", "replay", str(replay)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    first = proc.stdout.splitlines()[0]
    data = json.loads(first)
    assert set(data.keys()) >= {"alive", "score", "steps", "head"}
