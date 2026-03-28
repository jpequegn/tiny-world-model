"""WorldModel: combines encoder, transition model, and decoder."""

import logging

import torch
import torch.nn as nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.transition import TransitionModel

logger = logging.getLogger(__name__)


class WorldModel(nn.Module):
    """Latent-space dynamics model for CartPole.

    Forward pass:
        obs → encoder → z → transition(z, a) → z' → decoder → obs_pred

    Args:
        obs_dim: Observation dimensionality.
        action_dim: Number of discrete actions.
        latent_dim: Latent space dimensionality.
        hidden_dim: Hidden layer width shared across all sub-networks.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        action_dim: int = 2,
        latent_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = Encoder(obs_dim, latent_dim, hidden_dim)
        self.transition = TransitionModel(latent_dim, action_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, obs_dim, hidden_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a full forward pass.

        Args:
            obs: Current observations (B, obs_dim).
            action: Discrete actions (B,) as int64.

        Returns:
            z: Encoded current state (B, latent_dim).
            z_next: Predicted next latent state (B, latent_dim).
            obs_pred: Reconstructed observation from z (B, obs_dim).
        """
        z = self.encoder(obs)
        z_next = self.transition(z, action)
        obs_pred = self.decoder(z)
        return z, z_next, obs_pred

    def imagine(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll out latent dynamics over a sequence of actions.

        Args:
            obs: Initial observation (obs_dim,) or (1, obs_dim).
            actions: Action sequence (T,) as int64.

        Returns:
            Predicted observations at each step (T, obs_dim).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        z = self.encoder(obs)
        preds = []
        for t in range(actions.shape[0]):
            z = self.transition(z, actions[t].unsqueeze(0))
            preds.append(self.decoder(z))
        return torch.cat(preds, dim=0)

    def log_parameter_count(self) -> int:
        """Log and return total trainable parameter count."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        trans = sum(p.numel() for p in self.transition.parameters() if p.requires_grad)
        dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        logger.info(
            "WorldModel parameters — encoder: %d | transition: %d | decoder: %d | total: %d",
            enc, trans, dec, total,
        )
        return total
