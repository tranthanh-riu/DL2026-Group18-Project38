import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn_ae import CNNAE
from src.utils import load_config, set_seed

PROCESSED_FOLDER = "data/processed"
CHECKPOINT_FOLDER = "checkpoints"
LOG_FOLDER = "results/logs"


def load_windows_as_tensor(name):
    # (N, 1024) -> (N, 1, 1024): Conv1d cần thêm chiều kênh
    X = np.load(os.path.join(PROCESSED_FOLDER, f"{name}.npz"))["X"]
    return torch.from_numpy(X).unsqueeze(1)


def compute_val_loss(model, val_loader, criterion):
    # Val loss trung bình theo window, không tính gradient
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for (xb,) in val_loader:
            loss = criterion(model(xb), xb)
            total_loss += loss.item() * len(xb)
            total_count += len(xb)
    return total_loss / total_count


def train_one_epoch(model, train_loader, optimizer, criterion):
    # Target của loss chính là input (autoencoder học dựng lại tín hiệu)
    model.train()
    total_loss = 0.0
    total_count = 0
    for (xb,) in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), xb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        total_count += len(xb)
    return total_loss / total_count


def train_one_seed(seed):
    config = load_config()
    ae_config = config["ae"]
    set_seed(seed)
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
    os.makedirs(LOG_FOLDER, exist_ok=True)

    X_train = load_windows_as_tensor("load0_train")
    X_val = load_windows_as_tensor("load0_val")
    print(f"seed {seed}: train {tuple(X_train.shape)}, val {tuple(X_val.shape)}")
    train_loader = DataLoader(TensorDataset(X_train), batch_size=ae_config["batch"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=ae_config["batch"], shuffle=False)

    model = CNNAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_config["lr"])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    log_rows = []
    start_time = time.time()
    for epoch in range(1, ae_config["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = compute_val_loss(model, val_loader, criterion)
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:>2} | train loss {train_loss:.5f} | val loss {val_loss:.5f}")

        # Early stopping: chỉ lưu checkpoint khi val loss tốt hơn tốt nhất cũ
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_FOLDER, f"cnn_ae_seed{seed}.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= ae_config["patience"]:
                print(f"early stopping tai epoch {epoch}")
                break

    train_seconds = time.time() - start_time
    pd.DataFrame(log_rows).to_csv(os.path.join(LOG_FOLDER, f"train_seed{seed}.csv"), index=False)
    print(f"seed {seed}: best val loss {best_val_loss:.5f}, thoi gian train {train_seconds:.1f}s")
    with open(os.path.join(LOG_FOLDER, f"train_time_seed{seed}.txt"), "w") as f:
        f.write(f"{train_seconds:.1f}\n")


def main():
    # python -m src.train 0   -> chỉ chạy seed 0;  python -m src.train -> chạy mọi seed trong config
    if len(sys.argv) > 1:
        seeds = [int(sys.argv[1])]
    else:
        seeds = load_config()["seeds"]
    for seed in seeds:
        train_one_seed(seed)


if __name__ == "__main__":
    main()
