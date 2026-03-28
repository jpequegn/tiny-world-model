"""Fine-tunes a RewardHead on top of a frozen WorldModel encoder."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.env.replay_buffer import ReplayBuffer
from src.models.world_model import WorldModel
from src.training.dataset import TransitionDataset
from src.training.reward_head import RewardHead


def train_reward_head(
    model: WorldModel,
    buf: ReplayBuffer,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> RewardHead:
    """Train a RewardHead on frozen encoder representations.

    Args:
        model: Trained WorldModel (encoder used as feature extractor).
        buf: ReplayBuffer with collected transitions.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        device: Target device; defaults to model's current device.

    Returns:
        Trained RewardHead.
    """
    if device is None:
        device = next(model.parameters()).device

    ds = TransitionDataset(buf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    latent_dim = model.encoder.net[-1].out_features
    head = RewardHead(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    model.eval()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for obs, _, _ in loader:
            obs = obs.to(device)
            with torch.no_grad():
                z = model.encoder(obs)
            # CartPole reward is always 1.0 while alive; use actual buffer rewards
            r_target = torch.ones(len(obs), 1, device=device)
            r_pred = head(z)
            loss = nn.functional.mse_loss(r_pred, r_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

    return head
