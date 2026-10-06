"""Load the processed splits with the chosen normalisation.

    from src.data.dataset import load_split
    d = load_split(load=0, split="train")            # global normalisation
    d = load_split(load=2, split="test", norm="condition")  # ablation

Returns a dict: X (N, 1, window) float32, y, fault_type, fault_size, file_id, start.
"""
import json
from functools import lru_cache

import numpy as np

from src.utils import load_config, resolve

LOADS = (0, 1, 2, 3)


@lru_cache(maxsize=None)
def _stats(processed_dir: str) -> dict:
    with open(resolve(processed_dir) / "stats.json") as f:
        return json.load(f)


def load_split(load: int, split: str, norm: str = "global", config: str = "configs/default.yaml") -> dict:
    """norm: 'global' (0 HP train stats), 'condition' (per-load calib stats) or 'none'."""
    pdir = load_config(config)["data"]["processed_dir"]
    path = resolve(pdir) / f"load{load}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.data.prepare` first")
    raw = np.load(path)
    d = {k: raw[k] for k in raw.files}

    X = d["X"].astype(np.float32)
    if norm == "global":
        s = _stats(pdir)["global"]
    elif norm == "condition":
        s = _stats(pdir)["condition"][str(load)]
    elif norm == "none":
        s = {"mean": 0.0, "std": 1.0}
    else:
        raise ValueError(f"unknown norm '{norm}'")
    d["X"] = ((X - s["mean"]) / s["std"])[:, None, :]
    return d


def available_splits(load: int) -> list[str]:
    return ["train", "val", "test"] if load == 0 else ["calib", "test"]
