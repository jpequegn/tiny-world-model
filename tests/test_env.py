"""Tests for CartPole wrapper, ReplayBuffer, and data collection."""

import numpy as np
import pytest

from src.env.cartpole import CartPoleEnv, Transition
from src.env.collect import collect_random
from src.env.replay_buffer import ReplayBuffer


# ── CartPoleEnv ──────────────────────────────────────────────────────────────

def test_cartpole_reset_shape():
    env = CartPoleEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    env.close()


def test_cartpole_step():
    env = CartPoleEnv(seed=0)
    env.reset()
    obs, reward, done = env.step(0)
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    env.close()


def test_cartpole_sample_action():
    env = CartPoleEnv(seed=0)
    for _ in range(20):
        a = env.sample_action()
        assert a in (0, 1)
    env.close()


# ── ReplayBuffer ─────────────────────────────────────────────────────────────

def _make_transition(i: int) -> Transition:
    return Transition(
        state=np.zeros(4, dtype=np.float32),
        action=i % 2,
        next_state=np.ones(4, dtype=np.float32),
        reward=1.0,
        done=(i % 10 == 9),
    )


def test_replay_buffer_add_and_len():
    buf = ReplayBuffer(capacity=100)
    for i in range(50):
        buf.add(_make_transition(i))
    assert len(buf) == 50


def test_replay_buffer_ring_wrap():
    buf = ReplayBuffer(capacity=10)
    for i in range(15):
        buf.add(_make_transition(i))
    assert len(buf) == 10
    assert buf.is_full()


def test_replay_buffer_sample_shape():
    buf = ReplayBuffer(capacity=100)
    for i in range(50):
        buf.add(_make_transition(i))
    batch = buf.sample(16)
    assert batch["states"].shape == (16, 4)
    assert batch["actions"].shape == (16,)
    assert batch["next_states"].shape == (16, 4)
    assert batch["rewards"].shape == (16,)
    assert batch["dones"].shape == (16,)


# ── collect_random ────────────────────────────────────────────────────────────

def test_collect_random_fills_buffer():
    buf = collect_random(n_transitions=500, seed=7)
    assert len(buf) == 500


def test_collect_random_transitions_valid():
    buf = collect_random(n_transitions=200, seed=99)
    # rewards should all be 1.0 in CartPole
    assert np.all(buf.rewards[: len(buf)] == 1.0)
    # actions should be 0 or 1
    assert set(np.unique(buf.actions[: len(buf)])).issubset({0, 1})
