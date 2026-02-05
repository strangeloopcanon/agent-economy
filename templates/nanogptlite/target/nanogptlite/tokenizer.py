from __future__ import annotations


class CharTokenizer:
    def __init__(self, *, vocab: list[str]):
        raise NotImplementedError

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError
