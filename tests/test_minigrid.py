"""Tests for MiniGrid env wrapper, CNN encoder, and MiniGridWorldModel."""

import numpy as np
import torch

from src.env.collect_minigrid import ImageReplayBuffer, collect_minigrid
from src.env.minigrid import MiniGridEnv
from src.models.cnn_encoder import CNNEncoder
from src.models.minigrid_world_model import MiniGridWorldModel

B = 4
C, H, W = 3, 7, 7
LATENT_DIM = 64
ACTION_DIM = 3


# ── MiniGridEnv ───────────────────────────────────────────────────────────────

def test_minigrid_reset_shape():
    env = MiniGridEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (C, H, W)
    assert obs.dtype == np.float32
    env.close()


def test_minigrid_obs_range():
    env = MiniGridEnv(seed=0)
    obs = env.reset()
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    env.close()


def test_minigrid_step():
    env = MiniGridEnv(seed=0)
    env.reset()
    obs, reward, done = env.step(0)
    assert obs.shape == (C, H, W)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    env.close()


def test_minigrid_sample_action():
    env = MiniGridEnv(seed=0)
    for _ in range(20):
        a = env.sample_action()
        assert a in range(ACTION_DIM)
    env.close()


# ── CNNEncoder ────────────────────────────────────────────────────────────────

def test_cnn_encoder_output_shape():
    enc = CNNEncoder(in_channels=C, obs_h=H, obs_w=W, latent_dim=LATENT_DIM)
    obs = torch.rand(B, C, H, W)
    z = enc(obs)
    assert z.shape == (B, LATENT_DIM)


def test_cnn_encoder_single_obs():
    enc = CNNEncoder(in_channels=C, obs_h=H, obs_w=W, latent_dim=LATENT_DIM)
    obs = torch.rand(C, H, W)
    z = enc(obs)
    assert z.shape == (LATENT_DIM,)


# ── MiniGridWorldModel ────────────────────────────────────────────────────────

def test_minigrid_world_model_forward():
    model = MiniGridWorldModel(action_dim=ACTION_DIM, latent_dim=LATENT_DIM)
    obs = torch.rand(B, C, H, W)
    action = torch.randint(0, ACTION_DIM, (B,))
    z, z_next, obs_recon = model(obs, action)
    assert z.shape == (B, LATENT_DIM)
    assert z_next.shape == (B, LATENT_DIM)
    assert obs_recon.shape == (B, C, H, W)


def test_minigrid_world_model_recon_range():
    """Decoder output should be in [0, 1] (Sigmoid activation)."""
    model = MiniGridWorldModel(action_dim=ACTION_DIM, latent_dim=LATENT_DIM)
    obs = torch.rand(B, C, H, W)
    action = torch.randint(0, ACTION_DIM, (B,))
    _, _, obs_recon = model(obs, action)
    assert obs_recon.min() >= 0.0 and obs_recon.max() <= 1.0


def test_minigrid_world_model_imagine():
    model = MiniGridWorldModel(action_dim=ACTION_DIM, latent_dim=LATENT_DIM)
    obs = torch.rand(C, H, W)
    T = 5
    actions = torch.randint(0, ACTION_DIM, (T,))
    preds = model.imagine(obs, actions)
    assert preds.shape == (T, C, H, W)


def test_minigrid_world_model_param_count():
    model = MiniGridWorldModel(action_dim=ACTION_DIM, latent_dim=LATENT_DIM)
    total = model.log_parameter_count()
    assert 0 < total < 500_000


# ── collect_minigrid ──────────────────────────────────────────────────────────

def test_collect_minigrid_fills_buffer():
    buf = collect_minigrid(n_transitions=100, seed=0)
    assert len(buf) == 100


def test_collect_minigrid_state_shapes():
    buf = collect_minigrid(n_transitions=50, seed=1)
    assert buf.states.shape == (50, C, H, W)
    assert buf.next_states.shape == (50, C, H, W)


def test_image_replay_buffer_sample():
    buf = ImageReplayBuffer(capacity=100)
    from src.env.minigrid import Transition
    rng = np.random.default_rng(0)
    for i in range(60):
        buf.add(Transition(
            state=rng.random((C, H, W)).astype("float32"),
            action=int(rng.integers(0, ACTION_DIM)),
            next_state=rng.random((C, H, W)).astype("float32"),
            reward=0.0,
            done=(i % 10 == 9),
        ))
    batch = buf.sample(16)
    assert batch["states"].shape == (16, C, H, W)
