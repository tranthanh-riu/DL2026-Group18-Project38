"""Score every split/load with a trained detector and save the scores.

The model is always trained on normal 0 HP data only. Scored sets:
  val0 (threshold fitting), test0-3 (evaluation), calib1-3 (condition-aware threshold)
for each normalisation mode (global = main pipeline, condition = ablation).

Usage:
    python -m src.evaluate --model iforest --seeds 0 1 2
    python -m src.evaluate --model cnn_ae  --seeds 0 1 2      # after src.train
"""
import argparse
import time

from src.data.dataset import LOADS, load_split
from src.scores import save_scores

SETS = [("val", 0)] + [("test", L) for L in LOADS] + [("calib", L) for L in LOADS if L != 0]


def get_detector(model: str, seed: int):
    if model == "iforest":
        from src.models.iforest import IForestDetector
        return IForestDetector(seed=seed).fit(load_split(0, "train")["X"])
    if model == "cnn_ae":
        from src.models.cnn_ae import load_detector  # added in the CNN-AE step
        return load_detector(seed)
    raise ValueError(f"unknown model '{model}'")


def apply_noise(X, noise: str, seed: int):
    if noise == "clean":
        return X
    from src.noise import add_noise  # added in the robustness step (Setup3)
    return add_noise(X, noise, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["iforest", "cnn_ae"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--norms", nargs="+", default=["global", "condition"])
    ap.add_argument("--noise", nargs="+", default=["clean"])
    args = ap.parse_args()

    for seed in args.seeds:
        t0 = time.time()
        det = get_detector(args.model, seed)
        for norm in args.norms:
            for noise in args.noise:
                for split, L in SETS:
                    if noise != "clean" and split != "test":
                        continue  # noise is injected into test data only
                    d = load_split(L, split, norm=norm)
                    X = apply_noise(d["X"], noise, seed)
                    save_scores(args.model, seed, norm, split, L, det.score(X), d, noise)
        print(f"{args.model} seed {seed}: scored in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
