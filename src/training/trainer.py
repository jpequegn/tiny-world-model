"""Trainer: trains the WorldModel on transition data."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from src.models.world_model import WorldModel
from src.training.dataset import TransitionDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    val_fraction: float = 0.1
    checkpoint_every: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs/world_model"
    device: str = "auto"


class Trainer:
    """Trains a WorldModel to minimise next-state reconstruction loss.

    Loss:
        L = MSE(decoder(encoder(obs)), obs)           # reconstruction
          + MSE(decoder(transition(encoder(obs), a)), obs_next)  # prediction
    """

    def __init__(self, model: WorldModel, dataset: TransitionDataset, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg

        if cfg.device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(cfg.device)

        self.model.to(self.device)
        logger.info("Training on device: %s", self.device)

        # Train / val split
        n_val = max(1, int(len(dataset) * cfg.val_fraction))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        self.train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        obs_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device)
        action = action.to(self.device)
        obs_next = obs_next.to(self.device)

        z, z_next, obs_recon = self.model(obs, action)
        obs_pred = self.model.decoder(z_next)

        recon_loss = nn.functional.mse_loss(obs_recon, obs)
        pred_loss = nn.functional.mse_loss(obs_pred, obs_next)
        total = recon_loss + pred_loss
        return total, recon_loss, pred_loss

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        totals = {"loss": 0.0, "recon": 0.0, "pred": 0.0}
        with torch.set_grad_enabled(train):
            for obs, action, obs_next in loader:
                loss, recon, pred = self._loss(obs, action, obs_next)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                totals["loss"] += loss.item()
                totals["recon"] += recon.item()
                totals["pred"] += pred.item()
        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    def train(self) -> list[dict]:
        history = []
        self.model.log_parameter_count()

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

            # TensorBoard
            self.writer.add_scalar("loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("recon/train", train_metrics["recon"], epoch)
            self.writer.add_scalar("pred/train", train_metrics["pred"], epoch)

            # Latent space norm (log once per epoch)
            with torch.no_grad():
                sample_obs = next(iter(self.train_loader))[0][:64].to(self.device)
                z = self.model.encoder(sample_obs)
                self.writer.add_scalar("latent/norm_mean", z.norm(dim=-1).mean().item(), epoch)

            logger.info(
                "Epoch %3d/%d | train_loss=%.5f recon=%.5f pred=%.5f | val_loss=%.5f",
                epoch, self.cfg.epochs,
                train_metrics["loss"], train_metrics["recon"], train_metrics["pred"],
                val_metrics["loss"],
            )

            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

            if epoch % self.cfg.checkpoint_every == 0:
                path = Path(self.cfg.checkpoint_dir) / f"world_model_epoch{epoch:04d}.pt"
                torch.save({"epoch": epoch, "model_state": self.model.state_dict()}, path)
                logger.info("Checkpoint saved: %s", path)

        self.writer.close()
        return history
