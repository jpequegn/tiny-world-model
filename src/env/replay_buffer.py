"""Ring-buffer replay buffer for environment transitions."""

import numpy as np

from src.env.cartpole import Transition


class ReplayBuffer:
    """Fixed-capacity ring buffer storing (s, a, s', r, done) transitions."""

    def __init__(self, capacity: int, obs_dim: int = 4):
        self.capacity = capacity
        self._ptr = 0
        self._size = 0

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

    def add(self, t: Transition) -> None:
        self.states[self._ptr] = t.state
        self.actions[self._ptr] = t.action
        self.next_states[self._ptr] = t.next_state
        self.rewards[self._ptr] = t.reward
        self.dones[self._ptr] = t.done
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "next_states": self.next_states[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        return self._size

    def is_full(self) -> bool:
        return self._size == self.capacity
