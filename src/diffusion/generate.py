"""Reverse diffusion: iteratively denoise a fully-masked sequence.

Decoding strategy: at each step t, the model predicts the clean token for
every position. We greedily commit tokens that are unmasked (argmax), and
keep the rest masked for the next step — similar to the "absorbing" schedule
in MDLM / Austin et al. 2021.
"""

from __future__ import annotations

import torch

from src.diffusion.model import DiffusionTransformer
from src.diffusion.vocab import MASK_ID, decode


def generate(
    model: DiffusionTransformer,
    seq_len: int = 64,
    steps: int = 20,
    device: torch.device | None = None,
    T: int = 100,
) -> tuple[str, list[str]]:
    """Generate a sequence via iterative masked diffusion.

    Starts from a fully-masked sequence and unmasks tokens over `steps` rounds.
    At each round t (from T down to 1), the model predicts clean tokens;
    positions with high confidence are committed (unmasked), the rest remain
    masked for the next round.

    Args:
        model: Trained DiffusionTransformer.
        seq_len: Length of the sequence to generate.
        steps: Number of denoising steps (≤ T).
        device: Target device.
        T: Total diffusion steps the model was trained with.

    Returns:
        final_text: Decoded final string.
        trajectory: List of decoded strings at each step (for visualisation).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = torch.full((1, seq_len), MASK_ID, dtype=torch.long, device=device)
    trajectory: list[str] = []

    timesteps = torch.linspace(T, 1, steps, dtype=torch.long, device=device)

    with torch.no_grad():
        for step_idx, t_val in enumerate(timesteps):
            t = t_val.unsqueeze(0)   # (1,)
            logits = model(x, t)     # (1, L, V)
            probs = logits.softmax(dim=-1)

            # Greedy pick for masked positions
            predicted = logits.argmax(dim=-1)   # (1, L)

            # Fraction of tokens to reveal at this step (schedule: reveal
            # proportionally more as t decreases)
            frac_revealed = 1.0 - (t_val.item() / T)
            n_reveal = max(1, int(frac_revealed * seq_len))

            # Among still-masked positions, pick the n_reveal most confident
            still_masked = (x == MASK_ID).squeeze(0)  # (L,)
            if still_masked.any():
                confidence = probs[0].max(dim=-1).values  # (L,)
                confidence[~still_masked] = -1.0          # ignore unmasked
                top_k = confidence.topk(min(n_reveal, still_masked.sum().item())).indices
                x[0, top_k] = predicted[0, top_k]

            trajectory.append(decode(x[0].tolist()))

    return decode(x[0].tolist()), trajectory


def generate_autoregressive(
    model: DiffusionTransformer,
    seq_len: int = 64,
    device: torch.device | None = None,
    T: int = 100,
) -> str:
    """Baseline: token-by-token left-to-right generation.

    Uses the diffusion model at t=1 (minimal noise) as a proxy AR model:
    fills positions left-to-right using the model's argmax prediction,
    conditioning on already-filled tokens.

    Args:
        model: Trained DiffusionTransformer.
        seq_len: Sequence length to generate.
        device: Target device.
        T: Diffusion steps the model was trained with.

    Returns:
        Generated string.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = torch.full((1, seq_len), MASK_ID, dtype=torch.long, device=device)
    t = torch.tensor([1], device=device)

    with torch.no_grad():
        for pos in range(seq_len):
            logits = model(x, t)
            x[0, pos] = logits[0, pos].argmax()

    return decode(x[0].tolist())
