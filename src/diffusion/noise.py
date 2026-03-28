"""Masked diffusion noise schedule.

Forward process: independently mask each token with probability alpha(t),
where t is uniformly sampled from {1, ..., T} and alpha increases with t.

This follows the MDLM (Masked Diffusion Language Model) formulation:
  - t=0: original sequence (no masking)
  - t=T: fully masked sequence
  - alpha(t) = t / T  (linear schedule)
"""

from __future__ import annotations

import torch


def mask_rate(t: torch.Tensor, T: int) -> torch.Tensor:
    """Compute per-sample masking probability alpha(t) = t/T.

    Args:
        t: Integer diffusion timestep tensor (...,).
        T: Total number of diffusion steps.

    Returns:
        Float tensor of same shape as t, values in (0, 1].
    """
    return t.float() / T


def forward_diffuse(
    x: torch.Tensor,
    t: torch.Tensor,
    T: int,
    mask_id: int,
) -> torch.Tensor:
    """Apply forward diffusion: mask tokens with probability alpha(t).

    Args:
        x: Token IDs (B, L) int64.
        t: Diffusion timestep per sample (B,) int64.
        T: Total diffusion steps.
        mask_id: Token ID for the MASK token.

    Returns:
        Noisy token IDs (B, L) int64, with some tokens replaced by mask_id.
    """
    alpha = mask_rate(t, T)           # (B,)
    # Sample a Bernoulli mask: 1 = replace with MASK
    noise = torch.rand_like(x, dtype=torch.float)
    should_mask = noise < alpha.unsqueeze(1)   # (B, L)
    return torch.where(should_mask, torch.full_like(x, mask_id), x)


def sample_timesteps(batch_size: int, T: int, device: torch.device) -> torch.Tensor:
    """Sample uniformly from {1, ..., T}."""
    return torch.randint(1, T + 1, (batch_size,), device=device)
