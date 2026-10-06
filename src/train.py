"""Train the CNN autoencoder on NORMAL 0 HP windows only.

Usage:
    python -m src.train --seeds 0 1 2
Outputs:
    checkpoints/cnn_ae_s{seed}.pt               best model (lowest val loss)
    results/tables/cnn_ae_history.csv           train/val loss per epoch and seed
    results/figures/cnn_ae_loss.png             loss curves
"""
import argparse
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.dataset import load_split
from src.models.cnn_ae import build_model, checkpoint_path
from src.utils import load_config, resolve


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def run(seed: int, cfg: dict) -> pd.DataFrame:
    set_seed(seed)
    Xtr = torch.from_numpy(load_split(0, "train")["X"])
    Xva = torch.from_numpy(load_split(0, "val")["X"])
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)

    best, wait, hist = float("inf"), 0, []
    path = checkpoint_path(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tr = 0.0
        for i in range(0, len(Xtr), cfg["batch_size"]):
            xb = Xtr[perm[i:i + cfg["batch_size"]]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tr += loss.item() * len(xb)
        tr /= len(Xtr)
        model.eval()
        with torch.no_grad():
            va = loss_fn(model(Xva), Xva).item()
        hist.append({"seed": seed, "epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best - 1e-6:
            best, wait = va, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch,
                        "val_loss": va, "seed": seed}, path)
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    n_params = sum(p.numel() for p in model.parameters())
    print(f"seed {seed}: {epoch} epochs, best val MSE {best:.4f}, {n_params} params, "
          f"{time.time() - t0:.0f}s -> {path}")
    return pd.DataFrame(hist)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)["cnn_ae"]

    hist = pd.concat([run(s, cfg) for s in args.seeds])
    tab, fig_dir = resolve("results/tables"), resolve("results/figures")
    tab.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    hist.round(5).to_csv(tab / "cnn_ae_history.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    for s, h in hist.groupby("seed"):
        ax.plot(h.epoch, h.train_loss, label=f"train s{s}")
        ax.plot(h.epoch, h.val_loss, "--", label=f"val s{s}")
    ax.set(xlabel="epoch", ylabel="MSE (normalised units)", title="CNN-AE training on normal 0 HP")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "cnn_ae_loss.png", dpi=150)


if __name__ == "__main__":
    main()
