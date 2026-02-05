from __future__ import annotations

from institution_service.openai_executor import _read_hint_files


def test_read_hint_files_redacts_dotenv(tmp_path) -> None:
    (tmp_path / ".env").write_text('OPENAI_API_KEY="secret"\n', encoding="utf-8")
    files = _read_hint_files(root=tmp_path, rel_paths=[".env"])
    assert files[".env"] == "<redacted>"
