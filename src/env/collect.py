"""Data collection: random policy rollouts into a ReplayBuffer."""

import logging

import numpy as np

from src.env.cartpole import CartPoleEnv, Transition
from src.env.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


def collect_random(
    n_transitions: int = 10_000,
    capacity: int | None = None,
    seed: int = 42,
) -> ReplayBuffer:
    """Fill a ReplayBuffer with random-policy transitions.

    Args:
        n_transitions: Total transitions to collect.
        capacity: Buffer capacity (defaults to n_transitions).
        seed: RNG seed for reproducibility.

    Returns:
        Filled ReplayBuffer.
    """
    if capacity is None:
        capacity = n_transitions

    env = CartPoleEnv(seed=seed)
    buf = ReplayBuffer(capacity=capacity)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    collected = 0
    ep_reward = 0.0
    ep_len = 0
    state = env.reset()

    while collected < n_transitions:
        action = env.sample_action()
        next_state, reward, done = env.step(action)

        buf.add(Transition(state, action, next_state, reward, done))
        collected += 1
        ep_reward += reward
        ep_len += 1
        state = next_state

        if done:
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            ep_reward = 0.0
            ep_len = 0
            state = env.reset()

    env.close()
    _log_stats(episode_rewards, episode_lengths)
    return buf


def _log_stats(episode_rewards: list[float], episode_lengths: list[int]) -> None:
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    logger.info(
        "Collected %d episodes | "
        "reward: mean=%.1f std=%.1f min=%.0f max=%.0f | "
        "length: mean=%.1f min=%d max=%d",
        len(rewards),
        rewards.mean(), rewards.std(), rewards.min(), rewards.max(),
        lengths.mean(), lengths.min(), lengths.max(),
    )
