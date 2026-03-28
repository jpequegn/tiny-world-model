"""Tests for the evaluation suite."""

import numpy as np
import torch

from src.env.cartpole import Transition
from src.env.replay_buffer import ReplayBuffer
from src.eval.metrics import baseline_mse, latent_structure_score, rollout_mse
from src.models.world_model import WorldModel


def _make_buf(n: int = 300, seed: int = 0) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=n)
    rng = np.random.default_rng(seed)
    for i in range(n):
        buf.add(Transition(
            state=rng.random(4).astype("float32"),
            action=int(rng.integers(0, 2)),
            next_state=rng.random(4).astype("float32"),
            reward=1.0,
            done=(i % 20 == 19),
        ))
    return buf


# ── rollout_mse ───────────────────────────────────────────────────────────────

def test_rollout_mse_returns_correct_horizons():
    model = WorldModel()
    buf = _make_buf()
    result = rollout_mse(model, buf, horizons=[1, 5], n_episodes=10, device=torch.device("cpu"))
    assert set(result.keys()) == {1, 5}


def test_rollout_mse_nonnegative():
    model = WorldModel()
    buf = _make_buf()
    result = rollout_mse(model, buf, horizons=[1, 5], n_episodes=10, device=torch.device("cpu"))
    for h, mse in result.items():
        assert mse >= 0.0, f"Negative MSE at horizon {h}"


def test_rollout_mse_default_horizons():
    """rollout_mse works with default horizons when episodes are long enough."""
    buf = ReplayBuffer(capacity=500)
    rng = np.random.default_rng(5)
    for i in range(500):
        buf.add(Transition(
            state=rng.random(4).astype("float32"),
            action=int(rng.integers(0, 2)),
            next_state=rng.random(4).astype("float32"),
            reward=1.0,
            done=(i % 100 == 99),  # long episodes so 20-step windows exist
        ))
    model = WorldModel()
    result = rollout_mse(model, buf, horizons=[1, 5, 20], n_episodes=10, device=torch.device("cpu"))
    assert set(result.keys()) == {1, 5, 20}
    for mse in result.values():
        assert mse >= 0.0


# ── baseline_mse ──────────────────────────────────────────────────────────────

def test_baseline_mse_keys():
    buf = _make_buf()
    result = baseline_mse(buf)
    assert "random" in result
    assert "repeat_last" in result


def test_baseline_mse_nonnegative():
    buf = _make_buf()
    result = baseline_mse(buf)
    assert result["random"] >= 0.0
    assert result["repeat_last"] >= 0.0


# ── latent_structure_score ────────────────────────────────────────────────────

def test_latent_structure_score_range():
    model = WorldModel()
    buf = _make_buf()
    score = latent_structure_score(model, buf, n_samples=100, device=torch.device("cpu"))
    assert -1.0 <= score <= 1.0
