"""Reward head: latent z → predicted scalar reward."""

import torch
import torch.nn as nn


class RewardHead(nn.Module):
    """Small MLP predicting reward from a latent vector.

    Args:
        latent_dim: Dimensionality of the latent vector.
        hidden_dim: Width of the hidden layer.
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict reward.

        Args:
            z: Latent tensor (..., latent_dim).

        Returns:
            Scalar reward predictions (..., 1).
        """
        return self.net(z)
