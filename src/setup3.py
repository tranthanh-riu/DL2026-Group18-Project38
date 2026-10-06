import json
import os

import joblib
import numpy as np
import pandas as pd

from src.evaluate import load_model, reconstruction_score
from src.features import extract_features
from src.make_tables import MODELS, summarize
from src.metrics import at_threshold, threshold_free
from src.models.iforest import score as iforest_score
from src.noise import add_noise
from src.thresholds import pot_details, thr_mean3s, thr_oracle, thr_p99, thr_per_condition
from src.utils import load_config

PROCESSED_FOLDER = "data/processed"
SCORE_FOLDER = "results/scores"
TABLE_FOLDER = "results/tables"
CHECKPOINT_FOLDER = "checkpoints"


def load_norm():
    with open(os.path.join(PROCESSED_FOLDER, "norm_global.json")) as f:
        norm = json.load(f)
    return norm["mean"], norm["std"]


def score_any(model_name, seed, X):
    # Score cho cả 2 model từ X đã chuẩn hóa (N, 1024): score cao = bất thường
    if model_name == "iforest":
        model = joblib.load(os.path.join(CHECKPOINT_FOLDER, f"iforest_seed{seed}.joblib"))
        return iforest_score(model, extract_features(X))
    return reconstruction_score(load_model(seed), X)


def load_saved_scores(model_name, seed, name):
    return np.load(os.path.join(SCORE_FOLDER, f"{model_name}_seed{seed}_{name}.npz"))["score"]


def run_noise(config):
    # Nhiễu chỉ thêm vào test; ngưỡng p99 lấy từ val SẠCH
    mean, std = load_norm()
    rows = []
    for model_name in MODELS:
        for seed in config["seeds"]:
            threshold = thr_p99(load_saved_scores(model_name, seed, "val"))
            for snr_db in config["snr_db"]:
                for load in [0, 1, 2, 3]:
                    data = np.load(os.path.join(PROCESSED_FOLDER, f"load{load}_test.npz"))
                    noisy_raw = add_noise(data["X_raw"], snr_db, seed)
                    # Chuẩn hóa SAU khi thêm nhiễu (nhiễu là một phần của tín hiệu đo được)
                    noisy_X = ((noisy_raw - mean) / std).astype(np.float32)
                    test_score = score_any(model_name, seed, noisy_X)
                    row = {"model": model_name, "seed": seed, "snr_db": snr_db, "load": load}
                    row.update(threshold_free(data["y"], test_score))
                    row.update(at_threshold(data["y"], test_score, threshold))
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLE_FOLDER, "setup3_noise_per_seed.csv"), index=False, encoding="utf-8-sig")
    table = summarize(df, ["model", "snr_db", "load"])
    table.to_csv(os.path.join(TABLE_FOLDER, "setup3_noise.csv"), index=False, encoding="utf-8-sig")
    print("\nSetup3 noise (AUC-PR, F1, FPR):")
    print(table[["model", "snr_db", "load", "auc_pr", "f1", "fpr"]].to_string(index=False))
    return df


def get_threshold_scores(model_name, seed, load):
    # Score dùng để đặt ngưỡng per-condition: val cho 0 HP, calib cho 1-3 HP
    if load == 0:
        return load_saved_scores(model_name, seed, "val")
    return load_saved_scores(model_name, seed, f"calib{load}")


def run_thresholds(config):
    rows = []
    pot_config = config["pot"]
    for model_name in MODELS:
        for seed in config["seeds"]:
            val_score = load_saved_scores(model_name, seed, "val")
            pot_info = pot_details(val_score, pot_config["init_quantile"], pot_config["q"])
            print(f"POT {model_name} seed {seed}: t={pot_info['t']:.5f} N_t={pot_info['N_t']} "
                  f"xi={pot_info['xi']:.3f} sigma={pot_info['sigma']:.5f} z_q={pot_info['z_q']:.5f} "
                  f"| p99={thr_p99(val_score):.5f}")
            for load in [0, 1, 2, 3]:
                data = np.load(os.path.join(SCORE_FOLDER, f"{model_name}_seed{seed}_load{load}.npz"))
                y, test_score = data["y"], data["score"]
                thresholds = {
                    "p99": thr_p99(val_score),
                    "mean3s": thr_mean3s(val_score),
                    "pot": pot_info["z_q"],
                    "per_condition": thr_per_condition(get_threshold_scores(model_name, seed, load)),
                    "oracle": thr_oracle(y, test_score),
                }
                for strategy, threshold in thresholds.items():
                    row = {"model": model_name, "seed": seed, "strategy": strategy, "load": load,
                           "threshold": threshold}
                    row.update(at_threshold(y, test_score, threshold))
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLE_FOLDER, "setup3_threshold_per_seed.csv"), index=False, encoding="utf-8-sig")
    table = summarize(df, ["model", "strategy", "load"], ["precision", "recall", "f1", "fpr"])
    table.to_csv(os.path.join(TABLE_FOLDER, "setup3_threshold.csv"), index=False, encoding="utf-8-sig")
    print("\nSetup3 threshold:")
    print(table.to_string(index=False))
    return df


def local_norm_scores(seed, load):
    # Chuẩn hóa theo từng tải: mean/std từ X_raw của calib tải đó (ablation Bước 5c), model CNN-AE
    calib = np.load(os.path.join(PROCESSED_FOLDER, f"load{load}_calib.npz"))
    test = np.load(os.path.join(PROCESSED_FOLDER, f"load{load}_test.npz"))
    mean_load = float(calib["X_raw"].mean())
    std_load = float(calib["X_raw"].std())
    calib_X = ((calib["X_raw"] - mean_load) / std_load).astype(np.float32)
    test_X = ((test["X_raw"] - mean_load) / std_load).astype(np.float32)
    return score_any("cnn_ae", seed, calib_X), score_any("cnn_ae", seed, test_X), test["y"]


def run_ablation(config, threshold_df):
    # So chuẩn hóa toàn cục (ngưỡng p99 từ val) với chuẩn hóa theo tải (ngưỡng p99 từ calib tải đó)
    rows = []
    for seed in config["seeds"]:
        for load in [1, 2, 3]:
            calib_score, test_score, y = local_norm_scores(seed, load)
            row = {"seed": seed, "load": load}
            row.update(threshold_free(y, test_score))
            row.update(at_threshold(y, test_score, thr_p99(calib_score)))
            rows.append(row)
    local_df = pd.DataFrame(rows)

    # Phía toàn cục: AUC-PR từ per_seed_results (Bước 4), F1/FPR là ngưỡng p99 của val 0 HP
    base = pd.read_csv(os.path.join(TABLE_FOLDER, "per_seed_results.csv"))
    global_df = base[(base["model"] == "cnn_ae") & (base["load"] != 0)]

    columns = ["auc_pr", "f1", "fpr"]
    global_table = summarize(global_df, ["load"], columns)
    local_table = summarize(local_df, ["load"], columns)
    table = global_table[["load"]].copy()
    for col in columns:
        table[f"{col}_global_norm"] = global_table[col]
        table[f"{col}_per_load_norm"] = local_table[col]
    table.to_csv(os.path.join(TABLE_FOLDER, "setup3_ablation.csv"), index=False, encoding="utf-8-sig")
    print("\nSetup3 ablation (CNN-AE):")
    print(table.to_string(index=False))


def self_check(noise_df, threshold_df):
    # Ba mục Tự kiểm tra của Bước 5
    print("\n--- Tu kiem tra Buoc 5 ---")
    print("AUC-PR trung binh theo model va SNR (ky vong giam khi SNR thap):")
    print(noise_df.groupby(["model", "snr_db"])["auc_pr"].mean().round(4).to_string())
    pot = threshold_df[threshold_df["strategy"] == "pot"]
    print("Nguong POT: min =", pot["threshold"].min(), "| co NaN:", bool(pot["threshold"].isna().any()))
    oracle = threshold_df[threshold_df["strategy"] == "oracle"].set_index(["model", "seed", "load"])["f1"]
    others = threshold_df[threshold_df["strategy"] != "oracle"]
    violations = 0
    for _, row in others.iterrows():
        if row["f1"] > oracle[(row["model"], row["seed"], row["load"])] + 1e-9:
            violations += 1
    print("So truong hop F1 cua nguong khac > Oracle (ky vong 0):", violations)


def main():
    config = load_config()
    os.makedirs(TABLE_FOLDER, exist_ok=True)
    noise_df = run_noise(config)
    threshold_df = run_thresholds(config)
    run_ablation(config, threshold_df)
    self_check(noise_df, threshold_df)


if __name__ == "__main__":
    main()
