import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.evaluate import load_model
from src.setup3 import local_norm_scores
from src.thresholds import thr_p99
from src.utils import load_config

FIGURE_FOLDER = "results/figures"
TABLE_FOLDER = "results/tables"
SCORE_FOLDER = "results/scores"
PROCESSED_FOLDER = "data/processed"
LOG_FOLDER = "results/logs"
MODELS = ["iforest", "cnn_ae"]
MODEL_LABELS = {"iforest": "Isolation Forest", "cnn_ae": "CNN-AE"}


def save_figure(name):
    plt.savefig(os.path.join(FIGURE_FOLDER, name), dpi=200, bbox_inches="tight")
    plt.close()
    print("da luu", os.path.join(FIGURE_FOLDER, name))


def load_scores(model, seed, name):
    return np.load(os.path.join(SCORE_FOLDER, f"{model}_seed{seed}_{name}.npz"))


def plot_loss():
    # Hình 1: train/val loss theo epoch của seed 0 để xem model có hội tụ không
    log = pd.read_csv(os.path.join(LOG_FOLDER, "train_seed0.csv"))
    plt.figure(figsize=(6, 4))
    plt.plot(log["epoch"], log["train_loss"], label="train loss")
    plt.plot(log["epoch"], log["val_loss"], label="val loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (normalized units, log scale)")
    plt.title("CNN-AE training curve (seed 0)")
    plt.legend()
    save_figure("fig1_loss.png")


def plot_score_distribution():
    # Hình 2: histogram score của window normal ở từng tải, kèm ngưỡng p99 của val 0 HP
    threshold = thr_p99(load_scores("cnn_ae", 0, "val")["score"])
    bins = np.logspace(-3, 0, 40)
    plt.figure(figsize=(7, 4))
    for load in [0, 1, 2, 3]:
        data = load_scores("cnn_ae", 0, f"load{load}")
        normal_scores = data["score"][data["y"] == 0]
        plt.hist(normal_scores, bins=bins, alpha=0.5, label=f"{load} HP (n={len(normal_scores)})")
    plt.axvline(threshold, color="black", linestyle="--", label="p99 threshold (val, 0 HP)")
    plt.xscale("log")
    plt.xlabel("Reconstruction error (MSE, normalized units, log scale)")
    plt.ylabel("Number of normal windows")
    plt.title("Normal-window scores by load (CNN-AE, seed 0)")
    plt.legend()
    save_figure("fig2_score_distribution.png")


def plot_fpr_by_load(per_seed):
    # Hình 3: FPR theo tải, mean ± std qua 3 seed
    plt.figure(figsize=(6, 4))
    for model in MODELS:
        model_df = per_seed[per_seed["model"] == model].groupby("load")["fpr"]
        plt.errorbar(model_df.mean().index, model_df.mean().values, yerr=model_df.std().values,
                     marker="o", capsize=3, label=MODEL_LABELS[model])
    plt.xticks([0, 1, 2, 3])
    plt.xlabel("Load (HP)")
    plt.ylabel("False positive rate (fraction of normal windows)")
    plt.title("FPR vs load (p99 threshold from 0 HP val)")
    plt.legend()
    save_figure("fig3_fpr_by_load.png")


def plot_f1_by_snr(per_seed, noise_per_seed):
    # Hình 4: F1 theo SNR; "clean" lấy từ Bước 4. Mỗi điểm = trung bình 4 tải, thanh sai số = std qua seed
    labels = ["clean", "20", "10", "5"]
    plt.figure(figsize=(6, 4))
    for model in MODELS:
        means = []
        stds = []
        clean = per_seed[per_seed["model"] == model].groupby("seed")["f1"].mean()
        means.append(clean.mean())
        stds.append(clean.std())
        for snr_db in [20, 10, 5]:
            part = noise_per_seed[(noise_per_seed["model"] == model) & (noise_per_seed["snr_db"] == snr_db)]
            by_seed = part.groupby("seed")["f1"].mean()
            means.append(by_seed.mean())
            stds.append(by_seed.std())
        plt.errorbar(range(4), means, yerr=stds, marker="o", capsize=3, label=MODEL_LABELS[model])
    plt.xticks(range(4), labels)
    plt.xlabel("SNR (dB)")
    plt.ylabel("F1 (mean over 4 loads)")
    plt.title("Robustness to noise (p99 threshold from clean val)")
    plt.legend()
    save_figure("fig4_f1_by_snr.png")


def plot_f1_by_threshold(threshold_per_seed):
    # Hình 5: cột nhóm theo tải, mỗi cột một chiến lược ngưỡng (CNN-AE); oracle chỉ là cận trên
    strategies = ["p99", "mean3s", "pot", "per_condition", "oracle"]
    part = threshold_per_seed[threshold_per_seed["model"] == "cnn_ae"]
    width = 0.15
    plt.figure(figsize=(8, 4))
    for index, strategy in enumerate(strategies):
        grouped = part[part["strategy"] == strategy].groupby("load")["f1"]
        positions = np.arange(4) + (index - 2) * width
        plt.bar(positions, grouped.mean().values, width=width, yerr=grouped.std().values, capsize=2,
                label=strategy + (" (upper bound)" if strategy == "oracle" else ""))
    plt.xticks(range(4), ["0 HP", "1 HP", "2 HP", "3 HP"])
    plt.ylim(0.95, 1.005)
    plt.xlabel("Load")
    plt.ylabel("F1 (y-axis starts at 0.95)")
    plt.title("F1 by threshold strategy (CNN-AE)")
    plt.legend(loc="lower left", fontsize=8)
    save_figure("fig5_f1_by_threshold.png")


def reconstruct(model, window):
    # window: (1024,) -> tín hiệu dựng lại (1024,)
    x = torch.from_numpy(window).float().reshape(1, 1, -1)
    with torch.no_grad():
        return model(x).numpy().reshape(-1)


def plot_original_vs_reconstruction():
    # Hình 6: mỗi tải một ô con, tín hiệu gốc và tín hiệu dựng lại của một window normal
    model = load_model(0)
    figure, axes = plt.subplots(2, 2, figsize=(9, 5), sharey=True)
    for load in [0, 1, 2, 3]:
        data = np.load(os.path.join(PROCESSED_FOLDER, f"load{load}_test.npz"))
        window = data["X"][np.where(data["y"] == 0)[0][0]]
        axis = axes[load // 2][load % 2]
        axis.plot(window, label="original", linewidth=0.8)
        axis.plot(reconstruct(model, window), label="reconstruction", linewidth=0.8)
        axis.set_title(f"{load} HP (normal window)")
        axis.set_xlabel("Sample index (12 kHz)")
        axis.set_ylabel("Normalized amplitude")
    axes[0][0].legend(fontsize=8)
    figure.tight_layout()
    save_figure("fig6_original_vs_reconstruction.png")


def pick_error_cases(model, seed=0):
    # Chọn ca lỗi cho CNN-AE seed 0, ngưỡng p99 của val: (tên ca, tải, chỉ số window)
    threshold = thr_p99(load_scores("cnn_ae", seed, "val")["score"])
    cases = []
    for load in [0, 1, 2, 3]:
        data = load_scores("cnn_ae", seed, f"load{load}")
        missed = np.where((data["y"] == 1) & (data["score"] <= threshold))[0]
        print(f"{load} HP: fault bi bo sot = {len(missed)}")
        for index in missed[:2]:
            cases.append(("missed fault", load, index))
    # Không có fault bị bỏ sót -> lấy 2 fault có score thấp nhất (gần ngưỡng nhất) để phân tích
    if len(cases) == 0:
        data = load_scores("cnn_ae", seed, "load0")
        fault_indices = np.where(data["y"] == 1)[0]
        lowest = fault_indices[np.argsort(data["score"][fault_indices])[:2]]
        for index in lowest:
            cases.append(("weakest fault (not missed)", 0, index))
    data3 = load_scores("cnn_ae", seed, "load3")
    false_alarms = np.where((data3["y"] == 0) & (data3["score"] > threshold))[0]
    worst = false_alarms[np.argsort(-data3["score"][false_alarms])[:2]]
    for index in worst:
        cases.append(("false alarm at 3 HP", 3, index))
    cases.extend(pick_rescued_cases(seed, threshold))
    return cases, threshold


def pick_rescued_cases(seed, threshold):
    # Ca báo nhầm khi chuẩn hóa toàn cục nhưng đúng khi chuẩn hóa theo tải
    cases = []
    for load in [1, 2, 3]:
        calib_score, local_test_score, y = local_norm_scores(seed, load)
        global_score = load_scores("cnn_ae", seed, f"load{load}")["score"]
        global_alarm = (y == 0) & (global_score > threshold)
        local_ok = local_test_score <= thr_p99(calib_score)
        rescued = np.where(global_alarm & local_ok)[0]
        print(f"{load} HP: normal bao nham (global) duoc cuu nho chuan hoa theo tai = {len(rescued)}")
        if len(rescued) > 0 and len(cases) < 2:
            cases.append(("rescued by per-load norm", load, rescued[0]))
    return cases


def plot_error_cases():
    model = load_model(0)
    cases, threshold = pick_error_cases(model)
    rows = []
    figure, axes = plt.subplots(len(cases), 2, figsize=(11, 2.6 * len(cases)))
    for row_index, (case_name, load, index) in enumerate(cases):
        data = np.load(os.path.join(PROCESSED_FOLDER, f"load{load}_test.npz"))
        window = data["X"][index]
        reconstruction = reconstruct(model, window)
        error = (window - reconstruction) ** 2
        score = error.mean()
        axes[row_index][0].plot(window, label="original", linewidth=0.8)
        axes[row_index][0].plot(reconstruction, label="reconstruction", linewidth=0.8)
        axes[row_index][0].set_title(f"{case_name} | {load} HP | {data['fault_type'][index]} | score={score:.4f}", fontsize=9)
        axes[row_index][0].set_ylabel("Normalized amplitude")
        axes[row_index][1].plot(error, color="tab:red", linewidth=0.8)
        axes[row_index][1].set_ylabel("Squared error")
        rows.append({"case": case_name, "load": load, "window_index": int(index),
                     "fault_type": str(data["fault_type"][index]), "score": float(score), "p99_threshold": float(threshold)})
    axes[-1][0].set_xlabel("Sample index (12 kHz)")
    axes[-1][1].set_xlabel("Sample index (12 kHz)")
    axes[0][0].legend(fontsize=8)
    figure.tight_layout()
    save_figure("fig7_error_cases.png")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLE_FOLDER, "error_cases.csv"), index=False)
    print(table.to_string(index=False))


def main():
    load_config()
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    per_seed = pd.read_csv(os.path.join(TABLE_FOLDER, "per_seed_results.csv"))
    noise_per_seed = pd.read_csv(os.path.join(TABLE_FOLDER, "setup3_noise_per_seed.csv"))
    threshold_per_seed = pd.read_csv(os.path.join(TABLE_FOLDER, "setup3_threshold_per_seed.csv"))
    plot_loss()
    plot_score_distribution()
    plot_fpr_by_load(per_seed)
    plot_f1_by_snr(per_seed, noise_per_seed)
    plot_f1_by_threshold(threshold_per_seed)
    plot_original_vs_reconstruction()
    plot_error_cases()


if __name__ == "__main__":
    main()
