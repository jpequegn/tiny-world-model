"""Visualize sample rollouts from the replay buffer."""

import matplotlib.pyplot as plt
import numpy as np

from src.env.replay_buffer import ReplayBuffer


def plot_rollouts(buf: ReplayBuffer, n_episodes: int = 5, save_path: str | None = None) -> None:
    """Plot state trajectories for the first n_episodes in the buffer.

    Reconstructs episode boundaries using the `done` flags, then plots
    each of the 4 CartPole state dimensions over time.
    """
    episodes = _extract_episodes(buf, n_episodes)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    axes = axes.flatten()

    for ep_idx, ep_states in enumerate(episodes):
        ep_arr = np.array(ep_states)
        for dim in range(4):
            axes[dim].plot(ep_arr[:, dim], label=f"ep {ep_idx + 1}", alpha=0.7)

    for dim, ax in enumerate(axes):
        ax.set_title(labels[dim])
        ax.set_xlabel("Step")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"CartPole-v1 — {n_episodes} sample rollouts (random policy)", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved rollout plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _extract_episodes(buf: ReplayBuffer, n_episodes: int) -> list[list[np.ndarray]]:
    episodes: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []

    for i in range(len(buf)):
        current.append(buf.states[i])
        if buf.dones[i]:
            episodes.append(current)
            current = []
            if len(episodes) == n_episodes:
                break

    return episodes
