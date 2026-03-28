"""Transition model: (z, action) → z_next."""

import torch
import torch.nn as nn


class TransitionModel(nn.Module):
    """MLP transition model predicting the next latent state.

    Actions are one-hot encoded and concatenated with the latent vector.

    Args:
        latent_dim: Dimensionality of latent vectors.
        action_dim: Number of discrete actions.
        hidden_dim: Width of hidden layers.
    """

    def __init__(self, latent_dim: int = 32, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent state.

        Args:
            z: Current latent state (..., latent_dim).
            action: Discrete action indices (...,) as int64.

        Returns:
            Predicted next latent state (..., latent_dim).
        """
        action_onehot = nn.functional.one_hot(action, self.action_dim).float()
        x = torch.cat([z, action_onehot], dim=-1)
        return self.net(x)
