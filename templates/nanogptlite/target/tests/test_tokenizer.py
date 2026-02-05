from __future__ import annotations

import pytest

from nanogptlite.tokenizer import CharTokenizer


def test_roundtrip() -> None:
    tok = CharTokenizer.from_text("abca")
    ids = tok.encode("abca")
    assert tok.decode(ids) == "abca"
    assert tok.vocab_size >= 3


def test_unknown_char_raises() -> None:
    tok = CharTokenizer.from_text("ab")
    with pytest.raises(ValueError):
        tok.encode("abc")
