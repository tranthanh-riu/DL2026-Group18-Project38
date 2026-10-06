"""Standard location/format of anomaly scores, shared by all models and analyses.

File: results/scores/{model}_s{seed}_{norm}_{split}{load}_{noise}.npz
      e.g. iforest_s0_global_test2_clean.npz, cnn_ae_s1_global_val0_clean.npz
Arrays: score (N,), y (N,), fault_type (N,), file_id (N,), start (N,)
"""
import numpy as np

from src.utils import resolve

SCORE_DIR = resolve("results/scores")


def score_path(model, seed, norm, split, load, noise="clean"):
    return SCORE_DIR / f"{model}_s{seed}_{norm}_{split}{load}_{noise}.npz"


def save_scores(model, seed, norm, split, load, score, d, noise="clean"):
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(score_path(model, seed, norm, split, load, noise),
                        score=score.astype(np.float32), y=d["y"], fault_type=d["fault_type"],
                        file_id=d["file_id"], start=d["start"])


def load_scores(model, seed, norm, split, load, noise="clean") -> dict:
    p = score_path(model, seed, norm, split, load, noise)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run the scoring step first")
    z = np.load(p)
    return {k: z[k] for k in z.files}
