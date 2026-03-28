"""Encoder: observation → latent vector."""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """MLP encoder mapping observations to a latent vector.

    Args:
        obs_dim: Dimensionality of the input observation.
        latent_dim: Dimensionality of the output latent vector.
        hidden_dim: Width of hidden layers.
    """

    def __init__(self, obs_dim: int = 4, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent vectors.

        Args:
            obs: Tensor of shape (..., obs_dim).

        Returns:
            Latent tensor of shape (..., latent_dim).
        """
        return self.net(obs)
