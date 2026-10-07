"""Analysis figures for the report (results/figures/).

Reads the score files (results/scores) and the Setup3 per-seed table; the CNN-AE
reconstruction figures load checkpoints/cnn_ae_s{seed}.pt. No metric is tuned here.

Usage (after src.evaluate, src.robustness):
    python -m src.plots
Outputs:
    score_dist_by_load.png       score histograms per load and class, with thresholds
    fpr_by_load.png              FPR per load for each threshold strategy
    f1_vs_snr.png                F1 and FPR vs SNR per load
    threshold_sweep.png          F1 and FPR vs val0 percentile threshold
    reconstruction_by_load.png   CNN-AE: normal window and its reconstruction at 0-3 HP
    error_cases.png              6 failure cases (missed ball faults, false alarms)
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset import LOADS, load_split
from src.scores import load_scores
from src.utils import resolve

# Validated categorical palette (light surface); identity is also given by legend/markers.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
NORMAL_GRAY = "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"
MODELS = {"iforest": "Isolation Forest", "cnn_ae": "CNN-AE"}
CLASSES = [("normal", NORMAL_GRAY), ("inner_race", SERIES[0]), ("ball", SERIES[1]),
           ("outer_race", SERIES[2])]
METHODS = [("p99_val0", "p99 of val0"), ("pot_val0", "POT on val0"),
           ("condition_aware", "condition-aware (calib p99)")]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlecolor": INK, "axes.grid": True, "grid.color": "#ecebe7",
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    out = resolve("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out / name)


def val0_thr(model: str, seed: int, q: float = 99.0) -> float:
    return float(np.percentile(load_scores(model, seed, "global", "val", 0)["score"], q))


def calib_thr(model: str, seed: int, L: int, q: float = 99.0) -> float:
    if L == 0:
        return val0_thr(model, seed, q)
    return float(np.percentile(load_scores(model, seed, "global", "calib", L)["score"], q))


# ---------------------------------------------------------------- score distributions
def score_dist(seed: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2), sharey="row")
    for r, model in enumerate(MODELS):
        log = model == "cnn_ae"
        allsc = np.concatenate([load_scores(model, seed, "global", "test", L)["score"] for L in LOADS])
        lo, hi = (allsc[allsc > 0].min(), allsc.max()) if log else (allsc.min(), allsc.max())
        bins = np.geomspace(lo, hi, 60) if log else np.linspace(lo, hi, 60)
        for L in LOADS:
            ax = axes[r, L]
            t = load_scores(model, seed, "global", "test", L)
            for cls, col in CLASSES:
                s = t["score"][t["fault_type"] == cls]
                ax.hist(s, bins=bins, histtype="step", lw=1.6, color=col,
                        label=cls.replace("_", " "), density=True)
            ax.axvline(val0_thr(model, seed), color=INK, ls="--", lw=1.2, label="p99 of val0")
            if L:
                ax.axvline(calib_thr(model, seed, L), color=INK, ls=":", lw=1.4,
                           label="condition-aware")
            if log:
                ax.set_xscale("log")
            ax.set_title(f"{MODELS[model]} - test {L} HP")
            ax.set_xlabel("anomaly score" + (" (log)" if log else ""))
        axes[r, 0].set_ylabel("density")
    h, lab = axes[1, 1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Anomaly scores by load and class (seed {seed}, global norm, clean)", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "score_dist_by_load.png")


# ---------------------------------------------------------------- FPR per load
def fpr_by_load(df: pd.DataFrame) -> None:
    d = df[(df.experiment == "threshold")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    w = 0.26
    for ax, model in zip(axes, MODELS):
        for i, (m, label) in enumerate(METHODS):
            g = d[(d.model == model) & (d.threshold_method == m)].groupby("test_load")["fpr"]
            mean, std = g.mean() * 100, g.std().fillna(0) * 100
            x = np.arange(len(LOADS)) + (i - 1) * w
            bars = ax.bar(x, mean.values, w - 0.03, yerr=std.values, color=SERIES[i],
                          label=label, capsize=2, error_kw={"lw": 0.8, "ecolor": INK2})
            for b, v, e in zip(bars, mean.values, std.values):
                ax.text(b.get_x() + b.get_width() / 2, v + e + 1.5, f"{v:.0f}", ha="center",
                        va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(range(len(LOADS)), [f"{L} HP" for L in LOADS])
        ax.set(title=MODELS[model], xlabel="test load", ylim=(0, 112))
    axes[0].set_ylabel("false-positive rate on normal windows (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("False alarms under load shift, by threshold strategy (mean ± std, 3 seeds)",
                 color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fpr_by_load.png")


# ---------------------------------------------------------------- noise
def note_overlap(ax, y: float) -> None:
    ax.text(0.98, y, "1, 2 and 3 HP lines coincide", transform=ax.transAxes, ha="right",
            va="center", fontsize=8, color=INK2)


def f1_vs_snr(df: pd.DataFrame) -> None:
    d = df[df.experiment == "noise"]
    order = ["clean", "snr20", "snr10", "snr5"]
    xt = ["clean", "20 dB", "10 dB", "5 dB"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for c, model in enumerate(MODELS):
        for r, metric in enumerate(["f1", "fpr"]):
            ax = axes[r, c]
            for L in LOADS:
                g = d[(d.model == model) & (d.test_load == L)].groupby("noise")[metric]
                mean = g.mean().reindex(order).values * (100 if metric == "fpr" else 1)
                ax.plot(range(4), mean, color=SERIES[L], marker=MARKERS[L], ms=6,
                        label=f"{L} HP")
            ax.set_xticks(range(4), xt)
            if metric == "f1":
                ax.set(title=MODELS[model], ylim=(0.55, 1.02))
            else:
                ax.set(ylim=(-4, 104), xlabel="noise level (SNR vs normal 0 HP power)")
        axes[0, c].set_ylabel("F1" if c == 0 else "")
        axes[1, c].set_ylabel("FPR (%)" if c == 0 else "")
        if model == "cnn_ae":
            note_overlap(axes[0, c], 0.22)
            note_overlap(axes[1, c], 0.88)
    axes[0, 0].legend(title="test load", loc="lower left")
    fig.suptitle("Additive Gaussian noise: threshold = p99 of clean val0 (mean of 3 seeds)",
                 color=INK)
    fig.tight_layout()
    save(fig, "f1_vs_snr.png")


# ---------------------------------------------------------------- threshold sweep
def threshold_sweep(df: pd.DataFrame) -> None:
    d = df[df.experiment == "sweep"]
    qs = sorted(d.q.unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for c, model in enumerate(MODELS):
        for r, metric in enumerate(["f1", "fpr"]):
            ax = axes[r, c]
            for L in LOADS:
                g = d[(d.model == model) & (d.test_load == L)].groupby("q")[metric].mean()
                y = g.reindex(qs).values * (100 if metric == "fpr" else 1)
                ax.plot(range(len(qs)), y, color=SERIES[L], marker=MARKERS[L], ms=5,
                        label=f"{L} HP")
            ax.set_xticks(range(len(qs)), [f"{q:g}" for q in qs])
            if metric == "f1":
                ax.set(title=MODELS[model], ylim=(0.55, 1.02))
            else:
                ax.set(ylim=(-4, 104), xlabel="threshold = percentile of normal 0 HP val scores")
        axes[0, c].set_ylabel("F1" if c == 0 else "")
        axes[1, c].set_ylabel("FPR (%)" if c == 0 else "")
        if model == "cnn_ae":
            note_overlap(axes[0, c], 0.22)
            note_overlap(axes[1, c], 0.88)
    axes[0, 1].legend(title="test load", loc="center right")
    fig.suptitle("Threshold sensitivity (global norm, clean, mean of 3 seeds)", color=INK)
    fig.tight_layout()
    save(fig, "threshold_sweep.png")


# ---------------------------------------------------------------- CNN-AE reconstructions
def _detector(seed: int):
    from src.models.cnn_ae import load_detector  # needs torch
    return load_detector(seed)


def _plot_recon(ax, x, xr, n=512) -> None:
    t = np.arange(n)
    ax.plot(t, x[:n], color=NORMAL_GRAY, lw=1.0, label="original")
    ax.plot(t, xr[:n], color=SERIES[0], lw=1.0, label="CNN-AE reconstruction")
    ax.set_xlim(0, n)


def reconstruction_by_load(seed: int) -> None:
    det = _detector(seed)
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
    for L, ax in zip(LOADS, axes):
        d = load_split(L, "test")
        idx = np.flatnonzero(d["fault_type"] == "normal")
        s = load_scores("cnn_ae", seed, "global", "test", L)["score"]
        i = idx[np.argsort(s[idx])[len(idx) // 2]]  # median-score normal window
        x = d["X"][i:i + 1]
        xr = det.reconstruct(x)[0, 0]
        _plot_recon(ax, x[0, 0], xr)
        ax.set_title(f"normal {L} HP (median window): reconstruction MSE {s[i]:.3f}  "
                     f"| p99 val0 threshold {val0_thr('cnn_ae', seed):.3f}", loc="left")
        ax.set_ylabel("amplitude (z)")
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("sample (first 512 of 1024, 48 kHz)")
    fig.suptitle(f"CNN-AE trained on 0 HP reconstructs 0 HP well, other loads worse (seed {seed})",
                 color=INK)
    fig.tight_layout()
    save(fig, "reconstruction_by_load.png")


def error_cases(seed: int) -> None:
    """Failure cases, chosen by rule (median score of the failing group), not by hand."""
    det = _detector(seed)
    cases = []
    for L in (1, 2, 3):  # CNN-AE misses ball faults even with the condition-aware threshold
        thr = calib_thr("cnn_ae", seed, L)
        cases.append(("cnn_ae", L, "ball", lambda s, thr=thr: s <= thr, thr,
                      f"MISSED ball fault, {L} HP (CNN-AE, condition-aware thr)"))
    thr = val0_thr("cnn_ae", seed)
    cases.append(("cnn_ae", 3, "normal", lambda s, thr=thr: s > thr, thr,
                  "FALSE ALARM normal 3 HP (CNN-AE, p99 val0 thr)"))
    thr = val0_thr("iforest", seed)
    for L in (1, 2):
        cases.append(("iforest", L, "normal", lambda s, thr=thr: s > thr, thr,
                      f"FALSE ALARM normal {L} HP (IF, p99 val0 thr)"))

    fig, axes = plt.subplots(3, 2, figsize=(12, 7.5), sharex=True)
    for ax, (model, L, cls, fails, thr, title) in zip(axes.T.ravel(), cases):
        d = load_split(L, "test")
        s = load_scores(model, seed, "global", "test", L)["score"]
        idx = np.flatnonzero((d["fault_type"] == cls) & fails(s))
        n_cls = int((d["fault_type"] == cls).sum())
        i = idx[np.argsort(s[idx])[len(idx) // 2]]
        x = d["X"][i:i + 1]
        if model == "cnn_ae":
            _plot_recon(ax, x[0, 0], det.reconstruct(x)[0, 0])
        else:  # IF has no reconstruction: show the window only
            ax.plot(np.arange(512), x[0, 0, :512], color=NORMAL_GRAY, lw=1.0)
            ax.set_xlim(0, 512)
        ax.set_title(f"{title}\nscore {s[i]:.3f} vs thr {thr:.3f}; "
                     f"{len(idx)}/{n_cls} {cls} windows fail", loc="left", fontsize=9)
    axes[0, 0].legend(loc="upper right", ncol=2, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("sample (first 512 of 1024)")
    for ax in axes[:, 0]:
        ax.set_ylabel("amplitude (z)")
    fig.suptitle(f"Error cases (median window of each failing group, seed {seed})", color=INK)
    fig.tight_layout()
    save(fig, "error_cases.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="seed for per-window figures")
    args = ap.parse_args()
    df = pd.read_csv(resolve("results/tables/setup3_per_seed.csv"))
    score_dist(args.seed)
    fpr_by_load(df)
    f1_vs_snr(df)
    threshold_sweep(df)
    reconstruction_by_load(args.seed)
    error_cases(args.seed)


if __name__ == "__main__":
    main()
