"""MiniGrid-Empty-8x8-v0 environment wrapper with the same Env interface."""

from dataclasses import dataclass

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
import numpy as np


# MiniGrid actions: 0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done
# For Empty grid navigation we only need turn left/right + forward (0,1,2)
NAVIGATION_ACTIONS = (0, 1, 2)


@dataclass
class Transition:
    state: np.ndarray   # (7, 7, 3) uint8
    action: int
    next_state: np.ndarray
    reward: float
    done: bool


class MiniGridEnv:
    """Wrapper around MiniGrid-Empty-8x8-v0.

    Observations are the 7×7×3 partial-view image, normalised to float32 in [0, 1].
    Only navigation actions (left/right/forward) are exposed to keep the action
    space small and consistent with the CartPole regime.
    """

    ENV_ID = "MiniGrid-Empty-8x8-v0"
    OBS_SHAPE = (3, 7, 7)   # CHW for the CNN encoder
    ACT_DIM = len(NAVIGATION_ACTIONS)

    def __init__(self, seed: int = 0):
        self._env = gym.make(self.ENV_ID)
        self._seed = seed

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset(seed=self._seed)
        return self._process(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Step with a navigation action index (0=left,1=right,2=forward)."""
        raw_action = NAVIGATION_ACTIONS[action]
        obs, reward, terminated, truncated, _ = self._env.step(raw_action)
        done = terminated or truncated
        return self._process(obs), float(reward), done

    def sample_action(self) -> int:
        return int(np.random.randint(0, self.ACT_DIM))

    def close(self) -> None:
        self._env.close()

    @staticmethod
    def _process(obs: dict) -> np.ndarray:
        """Extract image, normalise to [0,1] float32, and convert to CHW."""
        img = obs["image"].astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))   # HWC → CHW
