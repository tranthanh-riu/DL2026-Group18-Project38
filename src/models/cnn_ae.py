"""1D convolutional autoencoder. Anomaly score = mean squared reconstruction error.

Shapes (window 1024): 1x1024 -> 16x512 -> 32x256 -> 64x128 -> latent 4x64 (256 values)
and back. The latent is 4x smaller than the input, so the network cannot simply copy
the signal: it must learn the structure of NORMAL vibration, and windows it cannot
reconstruct well (faults, unseen conditions) get a high score.
"""
import numpy as np
import torch
from torch import nn

from src.utils import load_config, resolve


class ConvAE(nn.Module):
    def __init__(self, channels=(16, 32, 64), latent_channels=4):
        super().__init__()
        c1, c2, c3 = channels
        self.encoder = nn.Sequential(
            nn.Conv1d(1, c1, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(c1, c2, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(c2, c3, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(c3, latent_channels, 3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, c3, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(c3, c2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(c2, c1, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(c1, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_model(cfg: dict) -> ConvAE:
    return ConvAE(tuple(cfg["channels"]), cfg["latent_channels"])


class CNNAEDetector:
    name = "cnn_ae"

    def __init__(self, model: ConvAE, batch_size: int = 256):
        self.model = model.eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        out = [self.model(torch.from_numpy(X[i:i + self.batch_size])).numpy()
               for i in range(0, len(X), self.batch_size)]
        return np.concatenate(out)

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        return ((self.reconstruct(X) - X) ** 2).mean(axis=(1, 2))


def checkpoint_path(seed: int, config: str = "configs/default.yaml"):
    cfg = load_config(config)["cnn_ae"]
    return resolve(cfg["checkpoint_dir"]) / f"cnn_ae_s{seed}.pt"


def load_detector(seed: int, config: str = "configs/default.yaml") -> CNNAEDetector:
    p = checkpoint_path(seed, config)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run `python -m src.train --seeds {seed}` first")
    ckpt = torch.load(p, map_location="cpu")
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return CNNAEDetector(model)
