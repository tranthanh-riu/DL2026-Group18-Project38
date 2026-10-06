import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.features import extract_features
from src.metrics import threshold_free
from src.utils import load_config

PROCESSED_FOLDER = "data/processed"
SCORE_FOLDER = "results/scores"
MODEL_FOLDER = "checkpoints"


def fit(features_train, seed, n_estimators):
    # Isolation Forest học trên đặc trưng của window normal 0 HP
    model = IsolationForest(n_estimators=n_estimators, random_state=seed)
    model.fit(features_train)
    return model


def score(model, features):
    # score_samples càng cao càng "bình thường" -> đổi dấu để score cao = bất thường
    # (cùng chiều với MSE của CNN-AE)
    return -model.score_samples(features)


def load_X(name):
    return np.load(os.path.join(PROCESSED_FOLDER, f"{name}.npz"))


def main():
    config = load_config()
    os.makedirs(SCORE_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    features_train = extract_features(load_X("load0_train")["X"])  # (N, 8)
    print("features_train:", features_train.shape)

    for seed in config["seeds"]:
        model = fit(features_train, seed, config["iforest"]["n_estimators"])
        # Lưu model để Setup3 (thêm nhiễu) dùng lại mà không cần fit lại
        joblib.dump(model, os.path.join(MODEL_FOLDER, f"iforest_seed{seed}.joblib"))

        val_score = score(model, extract_features(load_X("load0_val")["X"]))
        np.savez(os.path.join(SCORE_FOLDER, f"iforest_seed{seed}_val.npz"), score=val_score)
        for load in [1, 2, 3]:
            calib_score = score(model, extract_features(load_X(f"load{load}_calib")["X"]))
            np.savez(os.path.join(SCORE_FOLDER, f"iforest_seed{seed}_calib{load}.npz"), score=calib_score)
        for load in [0, 1, 2, 3]:
            data = load_X(f"load{load}_test")
            test_score = score(model, extract_features(data["X"]))
            np.savez(os.path.join(SCORE_FOLDER, f"iforest_seed{seed}_load{load}.npz"),
                     score=test_score, y=data["y"], fault_type=data["fault_type"])
            if load == 0:
                result = threshold_free(data["y"], test_score)
                print(f"seed {seed} | 0 HP | AUC-ROC = {result['auc_roc']:.4f} | AUC-PR = {result['auc_pr']:.4f}")


if __name__ == "__main__":
    main()
