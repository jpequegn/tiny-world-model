"""Tiny Transformer denoiser for masked diffusion.

Given a partially-masked sequence x_t and diffusion step t, predict the
original token at every position (including non-masked ones — the model
learns to reconstruct the full sequence at every step).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.diffusion.vocab import VOCAB_SIZE


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for the discrete diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep t.

        Args:
            t: Integer timesteps (B,).

        Returns:
            Embeddings (B, dim).
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)    # (B, dim)


class DiffusionTransformer(nn.Module):
    """Small Transformer denoiser.

    Architecture:
        - Token embedding (vocab_size → d_model)
        - Sinusoidal time embedding projected to d_model, added to each position
        - N Transformer encoder layers (bidirectional — no causal mask)
        - Linear head: d_model → vocab_size (logits over clean tokens)

    Args:
        vocab_size: Vocabulary size (default from vocab.py).
        d_model: Token embedding / hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.time_proj = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict clean token logits from noisy sequence.

        Args:
            x_t: Noisy token IDs (B, L) int64.
            t: Diffusion timestep (B,) int64.

        Returns:
            Logits (B, L, vocab_size).
        """
        B, L = x_t.shape
        positions = torch.arange(L, device=x_t.device).unsqueeze(0)  # (1, L)

        # Token + positional embeddings
        h = self.token_emb(x_t) + self.pos_emb(positions)

        # Time conditioning: broadcast over sequence positions
        t_emb = self.time_proj(self.time_emb(t))   # (B, d_model)
        h = h + t_emb.unsqueeze(1)                 # (B, L, d_model)

        h = self.transformer(h)
        return self.head(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
