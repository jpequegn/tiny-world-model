"""CartPole-v1 environment wrapper."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool


class CartPoleEnv:
    """Clean wrapper around gymnasium CartPole-v1."""

    OBS_DIM = 4
    ACT_DIM = 2

    def __init__(self, seed: int = 0):
        self._env = gym.make("CartPole-v1")
        self._seed = seed

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset(seed=self._seed)
        return obs.astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        obs, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        return obs.astype(np.float32), float(reward), done

    def sample_action(self) -> int:
        return self._env.action_space.sample()

    def close(self) -> None:
        self._env.close()
