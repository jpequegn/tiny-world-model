"""Evaluation metrics for world model quality."""

from __future__ import annotations

import numpy as np
import torch

from src.env.replay_buffer import ReplayBuffer
from src.models.world_model import WorldModel


def rollout_mse(
    model: WorldModel,
    buf: ReplayBuffer,
    horizons: list[int] | None = None,
    n_episodes: int = 50,
    device: torch.device | None = None,
) -> dict[int, float]:
    """Compute mean MSE at each prediction horizon by re-rolling from episode starts.

    For each starting state, rolls the model forward h steps and compares the
    predicted observation to the real one stored in the buffer.

    Args:
        model: Trained WorldModel.
        buf: ReplayBuffer with collected transitions.
        horizons: Prediction steps to evaluate.
        n_episodes: Number of episode starts to sample.
        device: Target device; defaults to model's device.

    Returns:
        Dict mapping horizon → mean MSE.
    """
    if horizons is None:
        horizons = [1, 5, 20]
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    max_h = max(horizons)

    # Find valid episode start indices (enough room for max_h steps, not crossing done)
    starts = _find_episode_starts(buf, max_h, n_episodes)

    mse_by_horizon: dict[int, list[float]] = {h: [] for h in horizons}

    with torch.no_grad():
        for idx in starts:
            obs0 = torch.tensor(buf.states[idx], device=device).unsqueeze(0)
            z = model.encoder(obs0)
            for step in range(1, max_h + 1):
                a = torch.tensor([buf.actions[idx + step - 1]], device=device)
                z = model.transition(z, a)
                if step in mse_by_horizon:
                    obs_pred = model.decoder(z)
                    obs_real = torch.tensor(buf.next_states[idx + step - 1], device=device).unsqueeze(0)
                    mse = torch.nn.functional.mse_loss(obs_pred, obs_real).item()
                    mse_by_horizon[step].append(mse)

    return {h: float(np.mean(v)) for h, v in mse_by_horizon.items()}


def _find_episode_starts(buf: ReplayBuffer, max_h: int, n: int) -> list[int]:
    size = len(buf)
    valid = []
    for i in range(size - max_h):
        # Ensure no episode boundary within the horizon window
        if not any(buf.dones[i:i + max_h]):
            valid.append(i)
    rng = np.random.default_rng(0)
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return chosen.tolist()


def baseline_mse(
    buf: ReplayBuffer,
    n_samples: int = 1000,
) -> dict[str, float]:
    """Compute MSE for simple baselines (random and linear/repeat-last).

    Args:
        buf: ReplayBuffer with collected transitions.
        n_samples: Number of transitions to sample.

    Returns:
        Dict with keys 'random' and 'repeat_last'.
    """
    rng = np.random.default_rng(42)
    n = len(buf)
    idx = rng.integers(0, n, size=n_samples)

    obs = buf.states[idx]
    obs_next = buf.next_states[idx]

    # Random: predict a random observation drawn from the data distribution
    rand_idx = rng.integers(0, n, size=n_samples)
    random_pred = buf.states[rand_idx]
    random_mse = float(np.mean((random_pred - obs_next) ** 2))

    # Repeat-last: predict obs_next ≈ obs (zero-change baseline)
    repeat_mse = float(np.mean((obs - obs_next) ** 2))

    return {"random": random_mse, "repeat_last": repeat_mse}


def latent_structure_score(
    model: WorldModel,
    buf: ReplayBuffer,
    n_samples: int = 500,
    device: torch.device | None = None,
) -> float:
    """Measure whether similar states map to similar latents (Spearman correlation).

    Computes pairwise observation distances and pairwise latent distances for
    a random sample, then returns their Spearman rank correlation.
    A high score (→1) means the latent space preserves neighbourhood structure.

    Args:
        model: Trained WorldModel.
        buf: ReplayBuffer.
        n_samples: Number of states to sample.
        device: Target device.

    Returns:
        Spearman correlation coefficient in [-1, 1].
    """
    from scipy.stats import spearmanr

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    rng = np.random.default_rng(1)
    n = len(buf)
    idx = rng.integers(0, n, size=n_samples)
    obs = torch.tensor(buf.states[idx], device=device)

    with torch.no_grad():
        z = model.encoder(obs).cpu().numpy()

    obs_np = buf.states[idx]

    # Flatten pairwise distances (upper triangle)
    from scipy.spatial.distance import pdist
    obs_dists = pdist(obs_np)
    z_dists = pdist(z)

    corr, _ = spearmanr(obs_dists, z_dists)
    return float(corr)


def latent_dim_ablation(
    buf: ReplayBuffer,
    latent_dims: list[int] | None = None,
    epochs: int = 20,
    device_str: str = "cpu",
) -> dict[int, dict[str, float]]:
    """Train a WorldModel for each latent_dim and report 1-step and 5-step MSE.

    Args:
        buf: ReplayBuffer with collected transitions.
        latent_dims: Latent dimensions to sweep.
        epochs: Training epochs per model.
        device_str: Device string.

    Returns:
        Dict mapping latent_dim → {horizon: mse, ...}.
    """
    from src.training.dataset import TransitionDataset
    from src.training.trainer import TrainConfig, Trainer

    if latent_dims is None:
        latent_dims = [8, 16, 32, 64]

    results = {}
    device = torch.device(device_str)

    for dim in latent_dims:
        model = WorldModel(latent_dim=dim)
        ds = TransitionDataset(buf)
        cfg = TrainConfig(
            epochs=epochs,
            batch_size=256,
            checkpoint_every=9999,
            log_dir=f"runs/ablation_dim{dim}",
            device=device_str,
        )
        Trainer(model, ds, cfg).train()
        mse = rollout_mse(model, buf, horizons=[1, 5], n_episodes=50, device=device)
        results[dim] = mse

    return results
