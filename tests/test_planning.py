"""Tests for planning utilities."""

import torch
import pytest

from src.eval.planner import (
    imagine_rollout,
    measure_prediction_accuracy,
    random_shooting,
)
from src.models.world_model import WorldModel
from src.training.reward_head import RewardHead

LATENT_DIM = 32
OBS_DIM = 4
ACTION_DIM = 2


def _model() -> WorldModel:
    return WorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM)


def _reward_head() -> RewardHead:
    return RewardHead(latent_dim=LATENT_DIM)


# ── imagine_rollout ───────────────────────────────────────────────────────────

def test_imagine_rollout_lengths():
    model = _model()
    z0 = torch.randn(LATENT_DIM)
    latents, actions = imagine_rollout(model, z0, policy=lambda z: 0, steps=10)
    assert len(latents) == 11  # z0 + 10 steps
    assert len(actions) == 10


def test_imagine_rollout_latent_shape():
    model = _model()
    z0 = torch.randn(LATENT_DIM)
    latents, _ = imagine_rollout(model, z0, policy=lambda z: 1, steps=5)
    for z in latents:
        assert z.shape == (1, LATENT_DIM)


def test_imagine_rollout_different_policies():
    model = _model()
    z0 = torch.randn(LATENT_DIM)
    latents_0, _ = imagine_rollout(model, z0, policy=lambda z: 0, steps=5)
    latents_1, _ = imagine_rollout(model, z0, policy=lambda z: 1, steps=5)
    # Different actions → different trajectories
    assert not torch.allclose(latents_0[-1], latents_1[-1])


# ── random_shooting ───────────────────────────────────────────────────────────

def test_random_shooting_returns_valid_action():
    model = _model()
    head = _reward_head()
    z0 = torch.randn(LATENT_DIM)
    action, ret = random_shooting(model, head, z0, horizon=5, n_candidates=32)
    assert action.item() in (0, 1)
    assert isinstance(ret, float)


def test_random_shooting_batch_z0():
    model = _model()
    head = _reward_head()
    z0 = torch.randn(1, LATENT_DIM)
    action, _ = random_shooting(model, head, z0, horizon=5, n_candidates=16)
    assert action.item() in (0, 1)


# ── measure_prediction_accuracy ───────────────────────────────────────────────

def test_measure_prediction_accuracy_length():
    model = _model()
    T = 10
    obs_seq = torch.randn(T + 1, OBS_DIM)
    act_seq = torch.randint(0, ACTION_DIM, (T,))
    mse = measure_prediction_accuracy(model, obs_seq, act_seq)
    assert len(mse) == T


def test_measure_prediction_accuracy_nonnegative():
    model = _model()
    T = 5
    obs_seq = torch.randn(T + 1, OBS_DIM)
    act_seq = torch.randint(0, ACTION_DIM, (T,))
    mse = measure_prediction_accuracy(model, obs_seq, act_seq)
    assert all(v >= 0.0 for v in mse)


# ── RewardHead ────────────────────────────────────────────────────────────────

def test_reward_head_shape():
    head = _reward_head()
    z = torch.randn(8, LATENT_DIM)
    r = head(z)
    assert r.shape == (8, 1)
