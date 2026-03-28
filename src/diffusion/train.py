"""Training loop for the masked diffusion language model."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.diffusion.model import DiffusionTransformer
from src.diffusion.noise import forward_diffuse, sample_timesteps
from src.diffusion.vocab import MASK_ID, PAD_ID, encode

logger = logging.getLogger(__name__)

# ── Toy dataset ───────────────────────────────────────────────────────────────

# Short sentences that stress long-range structure (verb-object agreement,
# rhyme, repeated patterns). The model needs to denoise the full sequence
# holistically — impossible for a purely left-to-right AR model without
# attending right-of-cursor.
_SENTENCES = [
    "the cat sat on the mat",
    "the dog lay on the log",
    "a fox in a box by the docks",
    "the quick brown fox jumps over the lazy dog",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "the sun also rises in the east",
    "every good boy deserves fudge",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck",
    "peter piper picked a peck of pickled peppers",
    "the rain in spain stays mainly in the plain",
    "red lorry yellow lorry red lorry yellow lorry",
    "unique new york unique new york you know you need unique new york",
    "if a black bug bleeds black blood what color blood bleeds a blue bug",
]


class CharDataset(Dataset):
    """Toy character-level dataset. Sequences are padded to seq_len."""

    def __init__(self, sentences: list[str], seq_len: int = 64):
        self.seq_len = seq_len
        self.data: list[torch.Tensor] = []
        for s in sentences:
            ids = encode(s)[:seq_len]
            # Pad to seq_len
            ids += [PAD_ID] * (seq_len - len(ids))
            self.data.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# ── Trainer ───────────────────────────────────────────────────────────────────

@dataclass
class DiffusionTrainConfig:
    T: int = 100               # diffusion steps
    epochs: int = 200
    batch_size: int = 8
    lr: float = 3e-4
    seq_len: int = 64          # must match model's max_len
    device: str = "cpu"


def train_diffusion(
    model: DiffusionTransformer,
    cfg: DiffusionTrainConfig,
    sentences: list[str] | None = None,
) -> list[float]:
    """Train the DiffusionTransformer on the toy sentence dataset.

    Loss: cross-entropy over ALL positions (not just masked ones).
    Following MDLM, the model learns to reconstruct the full clean sequence
    from the noisy input at every timestep.

    Args:
        model: DiffusionTransformer to train.
        cfg: Training configuration.
        sentences: Optional list of training sentences (defaults to built-in set).

    Returns:
        List of per-epoch mean losses.
    """
    if sentences is None:
        sentences = _SENTENCES

    device = torch.device(cfg.device)
    model.to(device).train()

    ds = CharDataset(sentences, seq_len=cfg.seq_len)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        for x0 in loader:
            x0 = x0.to(device)
            t = sample_timesteps(x0.shape[0], cfg.T, device)
            x_t = forward_diffuse(x0, t, cfg.T, MASK_ID)

            logits = model(x_t, t)   # (B, L, V)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                x0.view(-1),
                ignore_index=PAD_ID,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(loader)
        history.append(mean_loss)
        if epoch % 50 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d  loss=%.4f", epoch, cfg.epochs, mean_loss)

    return history
