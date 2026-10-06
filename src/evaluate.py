import os

import numpy as np
import torch

from src.metrics import threshold_free
from src.models.cnn_ae import CNNAE
from src.utils import load_config

PROCESSED_FOLDER = "data/processed"
SCORE_FOLDER = "results/scores"
CHECKPOINT_FOLDER = "checkpoints"


def reconstruction_score(model, X, batch_size=256):
    # Score mỗi window = MSE giữa tín hiệu gốc và tín hiệu dựng lại; X: (N, 1024) -> (N,)
    model.eval()
    X_tensor = torch.from_numpy(X).unsqueeze(1)  # (N, 1, 1024)
    all_scores = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            xb = X_tensor[start:start + batch_size]
            x_hat = model(xb)
            batch_score = ((xb - x_hat) ** 2).mean(dim=(1, 2))
            all_scores.append(batch_score.numpy())
    return np.concatenate(all_scores)


def load_model(seed):
    model = CNNAE()
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_FOLDER, f"cnn_ae_seed{seed}.pt")))
    model.eval()
    return model


def load_X(name):
    return np.load(os.path.join(PROCESSED_FOLDER, f"{name}.npz"))


def main():
    config = load_config()
    os.makedirs(SCORE_FOLDER, exist_ok=True)
    for seed in config["seeds"]:
        model = load_model(seed)
        val_score = reconstruction_score(model, load_X("load0_val")["X"])
        np.savez(os.path.join(SCORE_FOLDER, f"cnn_ae_seed{seed}_val.npz"), score=val_score)
        for load in [1, 2, 3]:
            calib_score = reconstruction_score(model, load_X(f"load{load}_calib")["X"])
            np.savez(os.path.join(SCORE_FOLDER, f"cnn_ae_seed{seed}_calib{load}.npz"), score=calib_score)
        for load in [0, 1, 2, 3]:
            data = load_X(f"load{load}_test")
            test_score = reconstruction_score(model, data["X"])
            np.savez(os.path.join(SCORE_FOLDER, f"cnn_ae_seed{seed}_load{load}.npz"),
                     score=test_score, y=data["y"], fault_type=data["fault_type"])
            if load == 0:
                result = threshold_free(data["y"], test_score)
                normal_mean = test_score[data["y"] == 0].mean()
                fault_mean = test_score[data["y"] == 1].mean()
                print(f"seed {seed} | 0 HP | AUC-ROC = {result['auc_roc']:.4f} | AUC-PR = {result['auc_pr']:.4f} | "
                      f"score TB normal = {normal_mean:.4f} | fault = {fault_mean:.4f}")


if __name__ == "__main__":
    main()
