"""WorldModel variant for MiniGrid with CNN encoder and pixel decoder."""

import logging

import torch
import torch.nn as nn

from src.models.cnn_encoder import CNNEncoder
from src.models.transition import TransitionModel

logger = logging.getLogger(__name__)

# MiniGrid partial-view image: (3, 7, 7) CHW
_C, _H, _W = 3, 7, 7
_FLAT = _C * _H * _W  # 147


class MiniGridWorldModel(nn.Module):
    """Latent-space world model for MiniGrid image observations.

    Encoder:    CNN  (3,7,7) → latent_dim
    Transition: MLP  (latent_dim + one-hot action) → latent_dim
    Decoder:    MLP  latent_dim → (3×7×7), reshaped to (3,7,7)

    The decoder reconstructs the flattened pixel image; MSE in pixel space
    is used as the reconstruction loss.

    Args:
        action_dim: Number of discrete navigation actions (default 3).
        latent_dim: Latent vector dimensionality.
        hidden_dim: Hidden layer width for transition and decoder.
    """

    def __init__(
        self,
        action_dim: int = 3,
        latent_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = CNNEncoder(
            in_channels=_C, obs_h=_H, obs_w=_W, latent_dim=latent_dim
        )
        self.transition = TransitionModel(
            latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, _FLAT),
            nn.Sigmoid(),  # pixel values in [0, 1]
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: Image observations (B, 3, 7, 7) float32 in [0, 1].
            action: Discrete navigation actions (B,) int64.

        Returns:
            z: Encoded latent (B, latent_dim).
            z_next: Predicted next latent (B, latent_dim).
            obs_recon: Reconstructed image (B, 3, 7, 7).
        """
        z = self.encoder(obs)
        z_next = self.transition(z, action)
        obs_recon = self.decoder(z).view(-1, _C, _H, _W)
        return z, z_next, obs_recon

    def imagine(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll out latent dynamics and decode each predicted state.

        Args:
            obs: Initial observation (3,7,7) or (1,3,7,7).
            actions: Action sequence (T,) int64.

        Returns:
            Predicted observations (T, 3, 7, 7).
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        z = self.encoder(obs)
        preds = []
        with torch.no_grad():
            for t in range(actions.shape[0]):
                z = self.transition(z, actions[t].unsqueeze(0))
                preds.append(self.decoder(z).view(1, _C, _H, _W))
        return torch.cat(preds, dim=0)

    def log_parameter_count(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        trans = sum(p.numel() for p in self.transition.parameters() if p.requires_grad)
        dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        logger.info(
            "MiniGridWorldModel — encoder: %d | transition: %d | decoder: %d | total: %d",
            enc, trans, dec, total,
        )
        return total
