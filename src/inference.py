"""Demo / inference: score a raw CWRU .mat recording window by window and flag anomalies.

The recording is cut into 1024-sample windows, z-scored with the normal 0 HP train
statistics (global norm, as in training), scored by a trained detector, and compared
with a threshold fitted on NORMAL data only:
  default       99th percentile of the detector's scores on normal 0 HP val windows
  --calib-mat   condition-aware: 99th percentile of the scores on the first 10% of a
                NORMAL recording from the new operating condition

Usage:
    python -m src.inference --mat data/raw/123.mat                        # CNN-AE, seed 0
    python -m src.inference --mat data/raw/99.mat --calib-mat data/raw/99.mat
    python -m src.inference --mat path/to/file.mat --model iforest --out results/inference/x.csv
Output: CSV with one row per window (start sample, time, score, threshold, flag).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from src.data.dataset import _stats, load_split
from src.evaluate import get_detector
from src.utils import load_config, resolve


def find_key(mat_path: Path, key: str | None) -> str:
    """Drive-end signal key. Prefer the one matching the file number (99.mat also holds X098_*)."""
    if key:
        return key
    keys = [k for k in sio.whosmat(mat_path) if k[0].endswith("_DE_time")]
    names = [k[0] for k in keys]
    if not names:
        raise KeyError(f"no *_DE_time variable in {mat_path}")
    num = re.sub(r"\D", "", mat_path.stem)
    match = [n for n in names if num and n.startswith(f"X{int(num):03d}_")]
    return (match or names)[0]


def read_windows(mat_path: Path, key: str | None, cfg: dict, stride: int,
                 frac: float = 1.0) -> tuple[np.ndarray, np.ndarray, str]:
    key = find_key(mat_path, key)
    sig = sio.loadmat(mat_path, variable_names=[key])[key].ravel().astype(np.float32)
    sig = sig[: int(len(sig) * frac)]
    win = cfg["window"]
    starts = np.arange(0, len(sig) - win + 1, stride)
    if len(starts) == 0:
        raise ValueError(f"{mat_path}: shorter than one window ({win} samples)")
    X = np.stack([sig[s:s + win] for s in starts])
    g = _stats(cfg["processed_dir"])["global"]
    return ((X - g["mean"]) / g["std"])[:, None, :].astype(np.float32), starts, key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True, help="CWRU .mat recording to score")
    ap.add_argument("--key", default=None, help="signal variable (default: auto *_DE_time)")
    ap.add_argument("--model", default="cnn_ae", choices=["cnn_ae", "iforest"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=None, help="default: window (no overlap)")
    ap.add_argument("--calib-mat", default=None,
                    help="NORMAL recording from the new condition for a condition-aware threshold")
    ap.add_argument("--q", type=float, default=99.0, help="threshold percentile")
    ap.add_argument("--out", default=None, help="CSV path (default results/inference/<name>.csv)")
    args = ap.parse_args()

    cfg = load_config()["data"]
    mat = resolve(args.mat)
    stride = args.stride or cfg["window"]
    det = get_detector(args.model, args.seed)

    if args.calib_mat:
        Xc, _, ckey = read_windows(resolve(args.calib_mat), None, cfg, cfg["eval_stride"],
                                   frac=cfg["calib_frac"])
        thr = float(np.percentile(det.score(Xc), args.q))
        thr_src = f"p{args.q:g} of {len(Xc)} calib windows ({Path(args.calib_mat).name}, {ckey})"
    else:
        thr = float(np.percentile(det.score(load_split(0, "val")["X"]), args.q))
        thr_src = f"p{args.q:g} of normal 0 HP val windows"

    X, starts, key = read_windows(mat, args.key, cfg, stride)
    score = det.score(X)
    flag = score > thr
    df = pd.DataFrame({"window": np.arange(len(X)), "start_sample": starts,
                       "start_s": np.round(starts / cfg["fs"], 4), "score": score,
                       "threshold": thr, "anomaly": flag.astype(int)})
    out = resolve(args.out) if args.out else resolve("results/inference") / f"{mat.stem}_{args.model}_s{args.seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"{mat.name} [{key}] -> {len(X)} windows of {cfg['window']} samples, model {args.model} seed {args.seed}")
    print(f"threshold {thr:.4f} ({thr_src})")
    print(f"flagged anomalous: {flag.sum()}/{len(X)} ({flag.mean():.1%}), "
          f"median score {np.median(score):.4f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
