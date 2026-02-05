from __future__ import annotations

import os
from pathlib import Path

from institution_service.sandbox import (
    apply_unified_diff,
    apply_unified_diff_path,
    build_patch_from_dirs,
    extract_file_blocks,
    extract_git_diff,
    parse_patch_changes,
)


def test_apply_unified_diff_works_with_relative_cwd(tmp_path) -> None:
    start = Path.cwd()
    try:
        os.chdir(tmp_path)
        sandbox = Path("sandbox")
        sandbox.mkdir(parents=True, exist_ok=True)
        (sandbox / "a.txt").write_text("hello\n", encoding="utf-8")

        patch_text = (
            "diff --git a/a.txt b/a.txt\n"
            "index e69de29..4b825dc 100644\n"
            "--- a/a.txt\n"
            "+++ b/a.txt\n"
            "@@ -1 +1 @@\n"
            "-hello\n"
            "+hello world\n"
        )
        apply_unified_diff(patch_text=patch_text, cwd=sandbox)
        assert (sandbox / "a.txt").read_text(encoding="utf-8") == "hello world\n"
    finally:
        os.chdir(start)


def test_apply_unified_diff_normalizes_corrupt_hunk_counts(tmp_path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "a.txt").write_text("hello\n", encoding="utf-8")

    patch_text = (
        "diff --git a/a.txt b/a.txt\n"
        "index e69de29..4b825dc 100644\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-hello\n"
        "+hello world\n"
    )
    patch_path = apply_unified_diff(patch_text=patch_text, cwd=sandbox)
    assert (sandbox / "a.txt").read_text(encoding="utf-8") == "hello world\n"
    assert "@@ -1,1 +1,1 @@" in patch_path.read_text(encoding="utf-8")


def test_build_patch_from_dirs_is_relativized_and_applyable(tmp_path) -> None:
    base_dir = tmp_path / "base"
    work_dir = tmp_path / "work"
    base_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)

    (base_dir / "a.txt").write_text("hello\n", encoding="utf-8")
    (work_dir / "a.txt").write_text("hello world\n", encoding="utf-8")
    (work_dir / "b.txt").write_text("new\n", encoding="utf-8")

    patch = build_patch_from_dirs(base_dir=base_dir, work_dir=work_dir)
    assert patch.touched_paths == ["a.txt", "b.txt"]
    assert str(tmp_path) not in patch.patch_text
    assert "diff --git a/a.txt b/a.txt" in patch.patch_text

    patch_path = tmp_path / "patch.diff"
    patch_path.write_text(patch.patch_text, encoding="utf-8")
    apply_unified_diff_path(patch_path=patch_path, cwd=base_dir)

    assert (base_dir / "a.txt").read_text(encoding="utf-8") == "hello world\n"
    assert (base_dir / "b.txt").read_text(encoding="utf-8") == "new\n"


def test_extract_file_blocks_strips_workspace_prefix() -> None:
    text = (
        "BEGIN_FILE runs/bench_snake_market/sandboxes/r0_T3/workspace/target/snakelite/render.py\n"
        "print('hi')\n"
        "END_FILE\n"
    )
    files = extract_file_blocks(text)
    assert sorted(files.keys()) == ["target/snakelite/render.py"]
    assert files["target/snakelite/render.py"] == "print('hi')\n"


def test_extract_git_diff_strips_workspace_prefix() -> None:
    raw = (
        "diff --git a/runs/bench_snake_market/sandboxes/r0_T3/workspace/target/snakelite/render.py "
        "b/runs/bench_snake_market/sandboxes/r0_T3/workspace/target/snakelite/render.py\n"
        "index e69de29..4b825dc 100644\n"
        "--- a/runs/bench_snake_market/sandboxes/r0_T3/workspace/target/snakelite/render.py\n"
        "+++ b/runs/bench_snake_market/sandboxes/r0_T3/workspace/target/snakelite/render.py\n"
        "@@ -0,0 +1 @@\n"
        "+print('hi')\n"
    )
    patch = extract_git_diff(raw)
    changes = parse_patch_changes(patch)
    assert [c.old_path for c in changes] == ["target/snakelite/render.py"]
    assert [c.new_path for c in changes] == ["target/snakelite/render.py"]


def test_build_patch_from_dirs_strips_cwd_relative_prefixes(tmp_path) -> None:
    start = Path.cwd()
    try:
        os.chdir(tmp_path)
        base_dir = tmp_path / "runs" / "bench" / "workspace"
        work_dir = tmp_path / "runs" / "bench" / "sandboxes" / "r0" / "workspace"
        (base_dir / "target").mkdir(parents=True)
        (work_dir / "target").mkdir(parents=True)
        (base_dir / "target" / "x.txt").write_text("hello\n", encoding="utf-8")
        (work_dir / "target" / "x.txt").write_text("hello world\n", encoding="utf-8")

        patch = build_patch_from_dirs(base_dir=base_dir, work_dir=work_dir)
        assert patch.touched_paths == ["target/x.txt"]
        assert "diff --git a/target/x.txt b/target/x.txt" in patch.patch_text
        assert "runs/" not in patch.patch_text
    finally:
        os.chdir(start)
