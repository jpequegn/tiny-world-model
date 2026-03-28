"""Planning via latent-space rollouts and random shooting."""

from __future__ import annotations

from typing import Callable

import torch

from src.models.world_model import WorldModel
from src.training.reward_head import RewardHead


def imagine_rollout(
    model: WorldModel,
    z0: torch.Tensor,
    policy: Callable[[torch.Tensor], int],
    steps: int = 20,
) -> tuple[list[torch.Tensor], list[int]]:
    """Roll forward in latent space using a policy.

    Args:
        model: Trained WorldModel.
        z0: Initial latent state (latent_dim,) or (1, latent_dim).
        policy: Callable mapping latent tensor (1, latent_dim) → int action.
        steps: Number of steps to roll out.

    Returns:
        latents: List of latent states at each step (length=steps+1).
        actions: List of actions taken (length=steps).
    """
    model.eval()
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)

    latents = [z0]
    actions = []
    z = z0

    with torch.no_grad():
        for _ in range(steps):
            a = policy(z)
            a_t = torch.tensor([a], device=z.device)
            z = model.transition(z, a_t)
            latents.append(z)
            actions.append(a)

    return latents, actions


def random_shooting(
    model: WorldModel,
    reward_head: RewardHead,
    z0: torch.Tensor,
    horizon: int = 20,
    n_candidates: int = 512,
    action_dim: int = 2,
) -> tuple[torch.Tensor, float]:
    """Select the best action sequence by sampling random plans.

    Rolls out `n_candidates` random action sequences in latent space,
    sums predicted rewards, and returns the first action of the best plan.

    Args:
        model: Trained WorldModel.
        reward_head: Trained RewardHead predicting scalar reward from z.
        z0: Initial latent state (latent_dim,) or (1, latent_dim).
        horizon: Planning horizon (number of steps).
        n_candidates: Number of random action sequences to evaluate.
        action_dim: Number of discrete actions.

    Returns:
        best_action: First action of the best plan as a scalar tensor.
        best_return: Predicted cumulative return of that plan.
    """
    model.eval()
    reward_head.eval()
    device = z0.device

    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)

    # Expand z0 to (n_candidates, latent_dim)
    z = z0.expand(n_candidates, -1).clone()

    # Sample random action sequences: (n_candidates, horizon)
    actions = torch.randint(0, action_dim, (n_candidates, horizon), device=device)

    cumulative_reward = torch.zeros(n_candidates, device=device)

    with torch.no_grad():
        for t in range(horizon):
            z = model.transition(z, actions[:, t])
            r = reward_head(z).squeeze(-1)
            cumulative_reward += r

    best_idx = cumulative_reward.argmax()
    best_action = actions[best_idx, 0]
    best_return = cumulative_reward[best_idx].item()
    return best_action, best_return


def measure_prediction_accuracy(
    model: WorldModel,
    obs_sequence: torch.Tensor,
    action_sequence: torch.Tensor,
) -> list[float]:
    """Measure per-step MSE between imagined and real observations.

    Args:
        model: Trained WorldModel.
        obs_sequence: Real observations (T+1, obs_dim).
        action_sequence: Actions taken (T,).

    Returns:
        mse_per_step: MSE at each prediction step (length=T).
    """
    model.eval()
    T = action_sequence.shape[0]

    with torch.no_grad():
        z = model.encoder(obs_sequence[0].unsqueeze(0))
        mse_per_step = []
        for t in range(T):
            z = model.transition(z, action_sequence[t].unsqueeze(0))
            obs_pred = model.decoder(z)
            mse = torch.nn.functional.mse_loss(obs_pred, obs_sequence[t + 1].unsqueeze(0))
            mse_per_step.append(mse.item())

    return mse_per_step
