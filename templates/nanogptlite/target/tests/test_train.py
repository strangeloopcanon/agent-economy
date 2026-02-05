from __future__ import annotations

from nanogptlite.train import train


def test_training_decreases_loss() -> None:
    _tok, _model, metrics = train(text="abababababababab", steps=200, lr=0.5, seed=0)
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["final_loss"] < 0.4
