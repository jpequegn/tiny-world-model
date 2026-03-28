"""Visualise predicted vs. real observations over a rollout."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from src.models.world_model import WorldModel


def plot_predicted_vs_real(
    model: WorldModel,
    obs_sequence: torch.Tensor,
    action_sequence: torch.Tensor,
    save_path: str | None = None,
) -> None:
    """Plot predicted vs. real observations for each state dimension.

    Args:
        model: Trained WorldModel.
        obs_sequence: Real observations (T+1, obs_dim).
        action_sequence: Actions taken (T,).
        save_path: File path to save figure; if None, calls plt.show().
    """
    model.eval()
    T = action_sequence.shape[0]
    obs_dim = obs_sequence.shape[1]

    predicted = []
    with torch.no_grad():
        z = model.encoder(obs_sequence[0].unsqueeze(0))
        for t in range(T):
            z = model.transition(z, action_sequence[t].unsqueeze(0))
            predicted.append(model.decoder(z).squeeze(0).cpu())

    predicted_arr = torch.stack(predicted).numpy()
    real_arr = obs_sequence[1:].cpu().numpy()

    labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for dim in range(obs_dim):
        ax = axes[dim]
        ax.plot(real_arr[:, dim], label="Real", linewidth=1.5)
        ax.plot(predicted_arr[:, dim], label="Predicted", linestyle="--", linewidth=1.5)
        ax.set_title(labels[dim])
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("World model: predicted vs. real observations", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
