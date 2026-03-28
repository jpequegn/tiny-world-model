"""PyTorch Dataset wrapping a ReplayBuffer."""

import torch
from torch.utils.data import Dataset

from src.env.replay_buffer import ReplayBuffer


class TransitionDataset(Dataset):
    """Dataset of (state, action, next_state) tuples from a ReplayBuffer."""

    def __init__(self, buf: ReplayBuffer):
        n = len(buf)
        self.states = torch.from_numpy(buf.states[:n].copy())
        self.actions = torch.from_numpy(buf.actions[:n].copy())
        self.next_states = torch.from_numpy(buf.next_states[:n].copy())

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx], self.next_states[idx]
