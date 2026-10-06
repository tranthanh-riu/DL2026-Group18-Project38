import os

import numpy as np
import pandas as pd

from src.metrics import at_threshold, threshold_free
from src.utils import load_config

SCORE_FOLDER = "results/scores"
TABLE_FOLDER = "results/tables"
MODELS = ["iforest", "cnn_ae"]
METRIC_COLUMNS = ["auc_roc", "auc_pr", "precision", "recall", "f1", "fpr"]


def format_mean_std(mean, std):
    # Ví dụ "0.912 ± 0.004"
    return f"{mean:.3f} ± {std:.3f}"


def summarize(df, group_cols, value_cols=METRIC_COLUMNS):
    # Gộp các seed: mỗi ô là chuỗi "mean ± std". Dùng chung cho Setup1/2/3
    # std dùng ddof=1 (mặc định của pandas); std = 0 nếu chỉ có 1 giá trị
    rows = []
    for group_values, group_df in df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values))
        for col in value_cols:
            if col in group_df.columns:
                row[col] = format_mean_std(group_df[col].mean(), group_df[col].std(ddof=1))
        row["n_seeds"] = len(group_df)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_one(model, seed, load):
    # Ngưỡng = percentile 99 của score val (normal 0 HP) -> áp cho test của tải `load`
    val_score = np.load(os.path.join(SCORE_FOLDER, f"{model}_seed{seed}_val.npz"))["score"]
    threshold = np.percentile(val_score, 99)
    data = np.load(os.path.join(SCORE_FOLDER, f"{model}_seed{seed}_load{load}.npz"))
    result = {"model": model, "seed": seed, "load": load}
    result.update(threshold_free(data["y"], data["score"]))
    result.update(at_threshold(data["y"], data["score"], threshold))
    return result


def main():
    config = load_config()
    os.makedirs(TABLE_FOLDER, exist_ok=True)
    rows = []
    for model in MODELS:
        for seed in config["seeds"]:
            for load in [0, 1, 2, 3]:
                rows.append(evaluate_one(model, seed, load))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLE_FOLDER, "per_seed_results.csv"), index=False, encoding="utf-8-sig")

    setup1 = summarize(df[df["load"] == 0], ["model", "load"])
    setup2 = summarize(df[df["load"] != 0], ["model", "load"])
    setup1.to_csv(os.path.join(TABLE_FOLDER, "setup1.csv"), index=False, encoding="utf-8-sig")
    setup2.to_csv(os.path.join(TABLE_FOLDER, "setup2.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("Setup1 (0 HP):")
    print(setup1.to_string(index=False))
    print("\nSetup2 (1-3 HP, nguong p99 tu val 0 HP):")
    print(setup2.to_string(index=False))


if __name__ == "__main__":
    main()
