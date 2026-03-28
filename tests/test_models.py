"""Tests for Encoder, TransitionModel, Decoder, and WorldModel."""

import torch
import pytest

from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.transition import TransitionModel
from src.models.world_model import WorldModel

B = 8
OBS_DIM = 4
LATENT_DIM = 32
ACTION_DIM = 2


# ── Encoder ───────────────────────────────────────────────────────────────────

def test_encoder_output_shape():
    enc = Encoder(obs_dim=OBS_DIM, latent_dim=LATENT_DIM)
    obs = torch.randn(B, OBS_DIM)
    z = enc(obs)
    assert z.shape == (B, LATENT_DIM)


def test_encoder_single_obs():
    enc = Encoder(obs_dim=OBS_DIM, latent_dim=LATENT_DIM)
    obs = torch.randn(OBS_DIM)
    z = enc(obs)
    assert z.shape == (LATENT_DIM,)


# ── TransitionModel ───────────────────────────────────────────────────────────

def test_transition_output_shape():
    trans = TransitionModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM)
    z = torch.randn(B, LATENT_DIM)
    action = torch.randint(0, ACTION_DIM, (B,))
    z_next = trans(z, action)
    assert z_next.shape == (B, LATENT_DIM)


def test_transition_different_actions_differ():
    trans = TransitionModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM)
    z = torch.randn(1, LATENT_DIM)
    z0 = trans(z, torch.tensor([0]))
    z1 = trans(z, torch.tensor([1]))
    assert not torch.allclose(z0, z1)


# ── Decoder ───────────────────────────────────────────────────────────────────

def test_decoder_output_shape():
    dec = Decoder(latent_dim=LATENT_DIM, obs_dim=OBS_DIM)
    z = torch.randn(B, LATENT_DIM)
    obs_pred = dec(z)
    assert obs_pred.shape == (B, OBS_DIM)


# ── WorldModel ────────────────────────────────────────────────────────────────

def test_world_model_forward_shapes():
    model = WorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM)
    obs = torch.randn(B, OBS_DIM)
    action = torch.randint(0, ACTION_DIM, (B,))
    z, z_next, obs_pred = model(obs, action)
    assert z.shape == (B, LATENT_DIM)
    assert z_next.shape == (B, LATENT_DIM)
    assert obs_pred.shape == (B, OBS_DIM)


def test_world_model_imagine():
    model = WorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM)
    obs = torch.randn(OBS_DIM)
    T = 10
    actions = torch.randint(0, ACTION_DIM, (T,))
    preds = model.imagine(obs, actions)
    assert preds.shape == (T, OBS_DIM)


def test_world_model_parameter_count():
    model = WorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM)
    total = model.log_parameter_count()
    assert total > 0
    # Sanity: should be a small model (< 100k params)
    assert total < 100_000


def test_world_model_gradients_flow():
    model = WorldModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM)
    obs = torch.randn(B, OBS_DIM)
    action = torch.randint(0, ACTION_DIM, (B,))
    z, z_next, obs_pred = model(obs, action)
    # Combine losses so all sub-networks receive gradients:
    # obs_pred touches encoder+decoder; z_next touches encoder+transition.
    loss = obs_pred.mean() + z_next.mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
