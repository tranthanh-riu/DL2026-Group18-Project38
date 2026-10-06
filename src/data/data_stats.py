"""Describe how the normal signal changes with motor load (task D-10).

Outputs
  results/tables/data_stats_by_load.csv
  results/figures/data_psd_by_load.png        normal PSD at 0/1/2/3 HP
  results/figures/data_examples.png           one raw window per load / fault type

Usage:
    python -m src.data.data_stats
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis

from src.data.dataset import LOADS, load_split
from src.utils import load_config, resolve

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    fs = load_config(args.config)["data"]["fs"]
    tab_dir, fig_dir = resolve("results/tables"), resolve("results/figures")
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows, psd = [], {}
    for L in LOADS:
        d = load_split(L, "test", norm="none")
        for ft in ["normal", "inner_race", "ball", "outer_race"]:
            X = d["X"][d["fault_type"] == ft, 0, :]
            rms = np.sqrt((X ** 2).mean(axis=1))
            f, P = welch(X, fs=fs, nperseg=X.shape[1], axis=1)
            Pm = P.mean(axis=0)
            if ft == "normal":
                psd[L] = (f, Pm)
            rows.append({"load_hp": L, "fault_type": ft, "windows": len(X),
                         "rms_mean": rms.mean(), "rms_std": rms.std(),
                         "kurtosis_mean": kurtosis(X, axis=1, fisher=False).mean(),
                         "peak_mean": np.abs(X).max(axis=1).mean(),
                         "dominant_freq_hz": float(f[1:][np.argmax(Pm[1:])])})

    df = pd.DataFrame(rows)
    base = df[(df.fault_type == "normal") & (df.load_hp == 0)].iloc[0]
    df["rms_change_vs_normal0_pct"] = 100 * (df.rms_mean / base.rms_mean - 1)
    df.round(5).to_csv(tab_dir / "data_stats_by_load.csv", index=False)
    print(df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    for L in LOADS:
        f, P = psd[L]
        ax.semilogy(f, P, color=COLORS[L], lw=1, label=f"normal {L} HP")
    ax.set(xlim=(0, 6000), ylim=(1e-9, None), xlabel="Frequency (Hz)", ylabel="PSD",
           title="Normal drive-end vibration spectrum changes with motor load")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "data_psd_by_load.png", dpi=150)

    fig, axes = plt.subplots(4, 4, figsize=(12, 7), sharex=True)
    t = np.arange(1024) / fs * 1000
    for L in LOADS:
        d = load_split(L, "test", norm="none")
        for j, ft in enumerate(["normal", "inner_race", "ball", "outer_race"]):
            x = d["X"][d["fault_type"] == ft][0, 0]
            axes[L, j].plot(t, x, lw=0.6, color=COLORS[L])
            if L == 0:
                axes[L, j].set_title(ft)
            if j == 0:
                axes[L, j].set_ylabel(f"{L} HP")
    for a in axes[-1]:
        a.set_xlabel("ms")
    fig.suptitle("One raw test window per load and condition")
    fig.tight_layout()
    fig.savefig(fig_dir / "data_examples.png", dpi=130)
    print(f"figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
