from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_train_emits_metrics_json() -> None:
    env = dict(os.environ)
    cwd = Path.cwd() / "target"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanogptlite",
            "train",
            "--steps",
            "50",
            "--lr",
            "0.5",
            "--seed",
            "0",
        ],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "initial_loss" in data
    assert "final_loss" in data
