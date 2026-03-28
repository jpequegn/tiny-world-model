"""Tests for TransitionDataset and Trainer."""

import torch

from src.env.cartpole import Transition
from src.env.replay_buffer import ReplayBuffer
from src.models.world_model import WorldModel
from src.training.dataset import TransitionDataset
from src.training.trainer import TrainConfig, Trainer


def _make_buffer(n: int = 200) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=n)
    import numpy as np
    rng = np.random.default_rng(0)
    for i in range(n):
        buf.add(Transition(
            state=rng.random(4).astype("float32"),
            action=int(rng.integers(0, 2)),
            next_state=rng.random(4).astype("float32"),
            reward=1.0,
            done=(i % 20 == 19),
        ))
    return buf


# ── TransitionDataset ─────────────────────────────────────────────────────────

def test_dataset_len():
    buf = _make_buffer(100)
    ds = TransitionDataset(buf)
    assert len(ds) == 100


def test_dataset_item_shapes():
    buf = _make_buffer(50)
    ds = TransitionDataset(buf)
    obs, action, next_obs = ds[0]
    assert obs.shape == (4,)
    assert action.shape == ()
    assert next_obs.shape == (4,)


def test_dataset_dtype():
    buf = _make_buffer(50)
    ds = TransitionDataset(buf)
    obs, action, _ = ds[0]
    assert obs.dtype == torch.float32
    assert action.dtype == torch.int64


# ── Trainer ───────────────────────────────────────────────────────────────────

def _make_trainer(epochs: int = 3) -> tuple[Trainer, WorldModel]:
    buf = _make_buffer(200)
    ds = TransitionDataset(buf)
    model = WorldModel()
    cfg = TrainConfig(
        epochs=epochs,
        batch_size=64,
        lr=1e-3,
        checkpoint_every=999,   # disable checkpointing
        log_dir="runs/test_run",
        device="cpu",
    )
    trainer = Trainer(model, ds, cfg)
    return trainer, model


def test_trainer_runs():
    trainer, _ = _make_trainer(epochs=2)
    history = trainer.train()
    assert len(history) == 2


def test_trainer_loss_keys():
    trainer, _ = _make_trainer(epochs=1)
    history = trainer.train()
    assert "train" in history[0]
    assert "val" in history[0]
    assert "loss" in history[0]["train"]
    assert "recon" in history[0]["train"]
    assert "pred" in history[0]["train"]


def test_trainer_loss_decreases():
    """Loss should decrease over 10 epochs on a small dataset."""
    trainer, _ = _make_trainer(epochs=10)
    history = trainer.train()
    first = history[0]["train"]["loss"]
    last = history[-1]["train"]["loss"]
    assert last < first, f"Expected loss to decrease: {first:.5f} → {last:.5f}"


def test_trainer_checkpoint(tmp_path):
    buf = _make_buffer(200)
    ds = TransitionDataset(buf)
    model = WorldModel()
    cfg = TrainConfig(
        epochs=2,
        batch_size=64,
        checkpoint_every=1,
        checkpoint_dir=str(tmp_path),
        log_dir=str(tmp_path / "runs"),
        device="cpu",
    )
    Trainer(model, ds, cfg).train()
    checkpoints = list(tmp_path.glob("*.pt"))
    assert len(checkpoints) == 2
    ckpt = torch.load(checkpoints[0], weights_only=True)
    assert "model_state" in ckpt
    assert "epoch" in ckpt
