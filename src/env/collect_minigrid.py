"""Random-policy data collection for MiniGrid."""

import logging

import numpy as np

from src.env.minigrid import MiniGridEnv, Transition
from src.env.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class ImageReplayBuffer(ReplayBuffer):
    """ReplayBuffer for image observations stored in CHW float32 format."""

    def __init__(self, capacity: int):
        # Override parent arrays with image-shaped storage
        import numpy as np
        self.capacity = capacity
        self._ptr = 0
        self._size = 0
        self.states = np.zeros((capacity, 3, 7, 7), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.next_states = np.zeros((capacity, 3, 7, 7), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)


def collect_minigrid(
    n_transitions: int = 5_000,
    capacity: int | None = None,
    seed: int = 42,
) -> ImageReplayBuffer:
    """Fill an ImageReplayBuffer with random-policy MiniGrid transitions.

    Args:
        n_transitions: Total transitions to collect.
        capacity: Buffer capacity (defaults to n_transitions).
        seed: RNG seed.

    Returns:
        Filled ImageReplayBuffer.
    """
    if capacity is None:
        capacity = n_transitions

    env = MiniGridEnv(seed=seed)
    buf = ImageReplayBuffer(capacity=capacity)

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

    rewards = np.array(episode_rewards) if episode_rewards else np.array([0.0])
    lengths = np.array(episode_lengths) if episode_lengths else np.array([0])
    logger.info(
        "MiniGrid collected %d episodes | "
        "reward: mean=%.3f max=%.3f | length: mean=%.1f max=%d",
        len(rewards), rewards.mean(), rewards.max(), lengths.mean(), int(lengths.max()),
    )
    return buf
