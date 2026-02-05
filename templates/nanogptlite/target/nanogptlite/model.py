from __future__ import annotations


class BigramLM:
    def __init__(self, *, vocab_size: int):
        raise NotImplementedError

    def loss(self, ids: list[int]) -> float:
        raise NotImplementedError

    def train_step(self, *, x: int, y: int, lr: float) -> None:
        raise NotImplementedError
