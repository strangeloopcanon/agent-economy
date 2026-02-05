from __future__ import annotations

from nanogptlite.model import BigramLM


def test_loss_is_finite() -> None:
    m = BigramLM(vocab_size=3)
    loss = m.loss([0, 1, 2, 0, 1])
    assert loss > 0.0
    assert loss < 100.0
