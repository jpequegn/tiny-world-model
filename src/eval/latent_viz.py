"""PCA / t-SNE visualisation of the latent space."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.env.replay_buffer import ReplayBuffer
from src.models.world_model import WorldModel


def plot_latent_pca(
    model: WorldModel,
    buf: ReplayBuffer,
    n_samples: int = 1000,
    save_path: str | None = None,
    device: torch.device | None = None,
) -> None:
    """Plot first two PCA components of the latent space, coloured by cart position.

    Args:
        model: Trained WorldModel.
        buf: ReplayBuffer.
        n_samples: Number of states to embed.
        save_path: File path to save; if None calls plt.show().
        device: Target device.
    """
    from sklearn.decomposition import PCA

    zs, cart_pos = _encode_sample(model, buf, n_samples, device)
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(zs)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=cart_pos, cmap="coolwarm", s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Cart position")
    ax.set_title("Latent space — PCA (coloured by cart position)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_latent_tsne(
    model: WorldModel,
    buf: ReplayBuffer,
    n_samples: int = 500,
    save_path: str | None = None,
    device: torch.device | None = None,
) -> None:
    """Plot t-SNE embedding of the latent space, coloured by cart position.

    Args:
        model: Trained WorldModel.
        buf: ReplayBuffer.
        n_samples: Number of states to embed (keep small for speed).
        save_path: File path to save; if None calls plt.show().
        device: Target device.
    """
    from sklearn.manifold import TSNE

    zs, cart_pos = _encode_sample(model, buf, n_samples, device)
    z2 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(zs)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=cart_pos, cmap="coolwarm", s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Cart position")
    ax.set_title("Latent space — t-SNE (coloured by cart position)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _encode_sample(
    model: WorldModel,
    buf: ReplayBuffer,
    n_samples: int,
    device: torch.device | None,
) -> tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    n = min(n_samples, len(buf))
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(buf), size=n)
    obs = torch.tensor(buf.states[idx], device=device)
    with torch.no_grad():
        z = model.encoder(obs).cpu().numpy()
    cart_pos = buf.states[idx, 0]  # first state dim = cart position
    return z, cart_pos


def _save_or_show(fig: "plt.Figure", save_path: str | None) -> None:
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
