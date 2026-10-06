"""Setup3 robustness tables: noise, threshold strategy, normalisation ablation.

Inference only - models are the ones trained on normal 0 HP. Thresholds are fitted on
normal scores only (val0, or the test load's calib set for condition-aware). The oracle
best-F1 threshold is picked on the test set and is reported as an UPPER BOUND only.

Usage (after scoring clean + noisy sets with src.evaluate):
    python -m src.robustness --models iforest cnn_ae
Outputs (results/tables/):
    setup3_noise.csv            F1/FPR/AUC vs SNR (global norm, p99 val0 threshold)
    setup3_threshold.csv        threshold strategies (global norm, clean)
    setup3_threshold_sweep.csv  F1/FPR vs val0 percentile 90-99.9 (for plots)
    setup3_ablation.csv         global vs condition normalisation x val0 / calib threshold
    setup3_per_seed.csv         every row above, per seed (numeric)
"""
import argparse

import numpy as np
import pandas as pd

from src import thresholds
from src.data.dataset import LOADS
from src.metrics import at_threshold, evaluate, oracle_best_f1
from src.scores import load_scores
from src.utils import resolve

NOISES = ["clean", "snr20", "snr10", "snr5"]
SWEEP_Q = [90.0, 92.5, 95.0, 97.0, 98.0, 99.0, 99.5, 99.9]
COLS = ["auc_roc", "auc_pr", "precision", "recall", "f1", "fpr", "threshold"]


def thr_for(method: str, m: str, s: int, norm: str, L: int, test: dict) -> float:
    val0 = load_scores(m, s, norm, "val", 0)["score"]
    if method == "p99_val0":
        return thresholds.percentile(val0, 99.0)
    if method == "mean_3std_val0":
        return thresholds.mean_kstd(val0, 3.0)
    if method == "pot_val0":
        return thresholds.pot(val0, init_q=98.0, risk=1e-3)
    if method == "condition_aware":
        calib = val0 if L == 0 else load_scores(m, s, norm, "calib", L)["score"]
        return thresholds.condition_aware(calib, 99.0)
    if method == "oracle_best_f1 (upper bound)":
        return oracle_best_f1(test["score"], test["y"])
    raise ValueError(method)


def row(exp, m, s, L, norm, noise, method, t, thr) -> dict:
    return {"experiment": exp, "model": m, "seed": s, "test_load": L, "norm": norm,
            "noise": noise, "threshold_method": method,
            **evaluate(t["score"], t["y"], t["fault_type"], thr)}


def collect(models, seeds) -> pd.DataFrame:
    rows = []
    for m in models:
        for s in seeds:
            for L in LOADS:
                for noise in NOISES:
                    t = load_scores(m, s, "global", "test", L, noise)
                    rows.append(row("noise", m, s, L, "global", noise, "p99_val0", t,
                                    thr_for("p99_val0", m, s, "global", L, t)))
                t = load_scores(m, s, "global", "test", L)
                for method in ["p99_val0", "mean_3std_val0", "pot_val0", "condition_aware",
                               "oracle_best_f1 (upper bound)"]:
                    rows.append(row("threshold", m, s, L, "global", "clean", method, t,
                                    thr_for(method, m, s, "global", L, t)))
                val0 = load_scores(m, s, "global", "val", 0)["score"]
                for q in SWEEP_Q:
                    rows.append({"experiment": "sweep", "model": m, "seed": s, "test_load": L,
                                 "norm": "global", "noise": "clean", "threshold_method": f"p{q}_val0",
                                 "q": q, **at_threshold(t["score"], t["y"], np.percentile(val0, q))})
                for norm in ["global", "condition"]:
                    t = load_scores(m, s, norm, "test", L)
                    for method in ["p99_val0", "condition_aware"]:
                        rows.append(row("ablation", m, s, L, norm, "clean", method, t,
                                        thr_for(method, m, s, norm, L, t)))
    return pd.DataFrame(rows)


def mean_std(df: pd.DataFrame, keys: list, cols: list) -> pd.DataFrame:
    g = df.groupby(keys, sort=False)[cols]
    out = g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["iforest", "cnn_ae"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    out = resolve("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    df = collect(args.models, args.seeds)
    df.round(4).to_csv(out / "setup3_per_seed.csv", index=False)

    tables = {
        "setup3_noise": mean_std(df[df.experiment == "noise"],
                                 ["model", "noise", "test_load"], COLS[:-1]),
        "setup3_threshold": mean_std(df[df.experiment == "threshold"],
                                     ["model", "threshold_method", "test_load"],
                                     ["precision", "recall", "f1", "fpr", "threshold"]),
        "setup3_ablation": mean_std(df[df.experiment == "ablation"],
                                    ["model", "norm", "threshold_method", "test_load"],
                                    ["auc_roc", "auc_pr", "f1", "fpr"]),
    }
    sweep = df[df.experiment == "sweep"].groupby(["model", "q", "test_load"])[["f1", "fpr"]]
    sweep = sweep.agg(["mean", "std"]).round(4)
    sweep.columns = [f"{a}_{b}" for a, b in sweep.columns]
    tables["setup3_threshold_sweep"] = sweep.reset_index()

    pd.set_option("display.width", 220)
    for name, tab in tables.items():
        tab.to_csv(out / f"{name}.csv", index=False)
        print(f"\n{name}\n{tab.to_string(index=False)}")


if __name__ == "__main__":
    main()
