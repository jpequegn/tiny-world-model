"""Decoder: latent vector → predicted observation."""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """MLP decoder reconstructing observations from latent vectors.

    Args:
        latent_dim: Dimensionality of the latent vector.
        obs_dim: Dimensionality of the output observation.
        hidden_dim: Width of hidden layers.
    """

    def __init__(self, latent_dim: int = 32, obs_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to observations.

        Args:
            z: Latent tensor of shape (..., latent_dim).

        Returns:
            Predicted observation of shape (..., obs_dim).
        """
        return self.net(z)
