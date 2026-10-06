"""Turn raw CWRU recordings into windowed, split datasets.

Split protocol (contiguous in time, no shuffling, so overlapping windows can
never leak across splits):
  * normal 0 HP  -> train / val / test  (70 / 15 / 15 % of the recording)
  * normal 1/2/3 HP -> calib (first 10 %) / test (remaining 90 %)
  * every fault recording -> test of its own load
Windows are stored as RAW amplitudes (float32). Normalisation statistics are
saved to stats.json and applied at load time by src/data/dataset.py:
  * "global"    : mean/std of the 0 HP training segment (main pipeline)
  * "condition" : mean/std of each load's calibration segment (ablation)

Output: data/processed/load{L}_{split}.npz, stats.json, summary.csv

Usage:
    python -m src.data.prepare
"""
import argparse
import json

import numpy as np
import pandas as pd
import scipy.io as sio

from src.utils import load_config, resolve


def read_signal(raw_dir, file_id: int, key: str) -> np.ndarray:
    # Use the exact key: 99.mat also contains X098_* variables.
    return sio.loadmat(raw_dir / f"{file_id}.mat", variable_names=[key])[key].ravel().astype(np.float32)


def windows(sig: np.ndarray, start: int, stop: int, win: int, stride: int):
    """All windows lying entirely inside sig[start:stop]."""
    starts = np.arange(start, stop - win + 1, stride, dtype=np.int64)
    if len(starts) == 0:
        return np.empty((0, win), np.float32), starts
    return np.stack([sig[s:s + win] for s in starts]), starts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)["data"]

    raw_dir, out_dir = resolve(cfg["raw_dir"]), resolve(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(cfg["manifest"]))
    win = cfg["window"]

    parts: dict[tuple[int, str], list[dict]] = {}

    def add(load, split, X, starts, row):
        parts.setdefault((load, split), []).append({
            "X": X, "start": starts,
            "y": np.full(len(X), row.label, np.int8),
            "fault_type": np.full(len(X), row.fault_type),
            "fault_size": np.full(len(X), row.fault_size_in, np.float32),
            "file_id": np.full(len(X), row.file_id, np.int32),
        })

    seg_stats = {}  # mean/std per (load, segment) on raw samples
    for row in manifest.itertuples():
        sig = read_signal(raw_dir, row.file_id, row.signal_key)
        n = len(sig)
        if row.label == 1:  # fault -> test of its load
            X, s = windows(sig, 0, n, win, cfg["fault_stride"])
            add(row.load_hp, "test", X, s, row)
            continue
        if row.load_hp == 0:
            a, b, _ = cfg["split_load0"]
            cut1, cut2 = int(n * a), int(n * (a + b))
            seg = {"train": (0, cut1, cfg["train_stride"]),
                   "val": (cut1, cut2, cfg["eval_stride"]),
                   "test": (cut2, n, cfg["eval_stride"])}
            seg_stats[0] = sig[:cut1]
        else:
            cut = int(n * cfg["calib_frac"])
            seg = {"calib": (0, cut, cfg["eval_stride"]),
                   "test": (cut, n, cfg["eval_stride"])}
            seg_stats[row.load_hp] = sig[:cut]
        for split, (lo, hi, stride) in seg.items():
            X, s = windows(sig, lo, hi, win, stride)
            add(row.load_hp, split, X, s, row)

    stats = {
        "global": {"mean": float(seg_stats[0].mean()), "std": float(seg_stats[0].std()),
                   "source": "normal 0 HP train segment"},
        "condition": {str(L): {"mean": float(x.mean()), "std": float(x.std()),
                               "source": "normal 0 HP train segment" if L == 0 else f"normal {L} HP calib segment"}
                      for L, x in sorted(seg_stats.items())},
        "window": win, "fs": cfg["fs"],
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    summary = []
    for (load, split), lst in sorted(parts.items()):
        d = {k: np.concatenate([p[k] for p in lst]) for k in lst[0]}
        np.savez_compressed(out_dir / f"load{load}_{split}.npz", **d)
        for ft in ["normal", "inner_race", "ball", "outer_race"]:
            cnt = int((d["fault_type"] == ft).sum())
            if cnt:
                summary.append({"load_hp": load, "split": split, "fault_type": ft, "windows": cnt})

    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "summary.csv", index=False)
    pivot = df.pivot_table(index=["load_hp", "split"], columns="fault_type",
                           values="windows", fill_value=0, aggfunc="sum")
    print(pivot.to_string())
    print(f"\nglobal stats (0 HP train): mean={stats['global']['mean']:.5f} std={stats['global']['std']:.5f}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
