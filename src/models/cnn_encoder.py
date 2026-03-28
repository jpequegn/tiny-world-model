"""CNN encoder for image observations (e.g. MiniGrid 7×7×3 partial view)."""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Small CNN mapping a CHW image to a flat latent vector.

    Default input: (3, 7, 7) float32 in [0, 1] (MiniGrid partial view).

    Architecture:
        Conv 3→16, 3×3, stride 1 → ELU
        Conv 16→32, 3×3, stride 1 → ELU
        Flatten → Linear → latent_dim

    Args:
        in_channels: Number of input channels (3 for MiniGrid RGB).
        obs_h: Observation height in pixels.
        obs_w: Observation width in pixels.
        latent_dim: Dimensionality of the output latent vector.
    """

    def __init__(
        self,
        in_channels: int = 3,
        obs_h: int = 7,
        obs_w: int = 7,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
        )
        # Compute flattened size after convolutions
        conv_h = obs_h - 2  # padding=0 on second conv: 7 → 5
        conv_w = obs_w - 2
        flat_dim = 32 * conv_h * conv_w

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode image observation to latent vector.

        Args:
            obs: Float tensor (..., C, H, W) in [0, 1].

        Returns:
            Latent tensor (..., latent_dim).
        """
        batch_shape = obs.shape[:-3]
        x = obs.view(-1, *obs.shape[-3:])
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x.view(*batch_shape, -1)
