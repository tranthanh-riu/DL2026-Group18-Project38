"""Metrics tables for Setup1 (same load) and Setup2 (load shift).

Threshold: 99th percentile of the model's scores on normal 0 HP validation windows
(fitted per seed). Results are mean +- std over seeds.

Usage:
    python -m src.summarize --models iforest
    python -m src.summarize --models iforest cnn_ae
Outputs: results/tables/setup1.csv, setup2.csv, metrics_per_seed.csv
"""
import argparse

import pandas as pd

from src import thresholds
from src.data.dataset import LOADS
from src.metrics import evaluate
from src.scores import load_scores
from src.utils import resolve

COLS = ["auc_roc", "auc_pr", "precision", "recall", "f1", "fpr",
        "recall_inner_race", "recall_ball", "recall_outer_race"]


def per_seed(models, seeds, norm="global", noise="clean", q=99.0) -> pd.DataFrame:
    rows = []
    for m in models:
        for s in seeds:
            thr = thresholds.percentile(load_scores(m, s, norm, "val", 0)["score"], q)
            for L in LOADS:
                t = load_scores(m, s, norm, "test", L, noise)
                rows.append({"model": m, "seed": s, "norm": norm, "noise": noise, "test_load": L,
                             **evaluate(t["score"], t["y"], t["fault_type"], thr)})
    return pd.DataFrame(rows)


def mean_std(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "test_load"])[COLS]
    out = g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    out = resolve("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    df = per_seed(args.models, args.seeds)
    df.round(4).to_csv(out / "metrics_per_seed.csv", index=False)
    table = mean_std(df)
    table[table.test_load == 0].to_csv(out / "setup1.csv", index=False)
    table[table.test_load != 0].to_csv(out / "setup2.csv", index=False)

    pd.set_option("display.width", 200)
    print("Setup1 - train 0 HP, test 0 HP\n", table[table.test_load == 0].to_string(index=False))
    print("\nSetup2 - train 0 HP, test 1/2/3 HP\n", table[table.test_load != 0].to_string(index=False))


if __name__ == "__main__":
    main()
