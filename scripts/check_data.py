"""Sanity checks for data/processed. Run after prepare:  python scripts/check_data.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import LOADS, available_splits, load_split  # noqa: E402
from src.utils import load_config  # noqa: E402

win = load_config()["data"]["window"]
errors = []

def check(cond, msg):
    if not cond:
        errors.append(msg)

for L in LOADS:
    spans = {}
    for split in available_splits(L):
        d = load_split(L, split)
        X = d["X"]
        check(X.ndim == 3 and X.shape[1:] == (1, win), f"load{L}_{split}: bad shape {X.shape}")
        check(np.isfinite(X).all(), f"load{L}_{split}: non-finite values")
        if split != "test":
            check((d["y"] == 0).all(), f"load{L}_{split}: contains fault windows")
        else:
            check(set(np.unique(d["y"])) == {0, 1}, f"load{L}_test: needs both normal and fault")
        normal = d["fault_type"] == "normal"
        if normal.any():
            s = d["start"][normal]
            spans[split] = (s.min(), s.max() + win)
    # normal segments of the same recording must not overlap across splits
    items = sorted(spans.items(), key=lambda kv: kv[1][0])
    for (a, (_, a_end)), (b, (b_start, _)) in zip(items, items[1:]):
        check(a_end <= b_start, f"load{L}: '{a}' overlaps '{b}'")

d = load_split(0, "train")
check(abs(d["X"].mean()) < 0.05 and abs(d["X"].std() - 1) < 0.05,
      "0 HP train is not ~zero-mean/unit-std under global normalisation")

if errors:
    print("FAILED:\n  " + "\n  ".join(errors))
    sys.exit(1)
print("all data checks passed")
