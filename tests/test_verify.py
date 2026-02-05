from __future__ import annotations

import os

from institution_service import verify


def test_base_env_prepends_sys_executable_parent(tmp_path, monkeypatch) -> None:
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    exe = venv_bin / "python"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(verify.sys, "executable", str(exe))
    monkeypatch.setenv("PATH", "/usr/bin")
    env = verify._base_env(scrub_secrets=False)
    first = env["PATH"].split(os.pathsep, 1)[0]
    assert first == str(venv_bin)
