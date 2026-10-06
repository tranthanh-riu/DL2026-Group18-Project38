"""Detection-threshold strategies. All fitted on NORMAL scores only.

Implemented now: percentile, mean_kstd. (POT and condition-aware are added in the
robustness step, Setup3.)
"""
import numpy as np


def percentile(normal_scores: np.ndarray, q: float = 99.0) -> float:
    return float(np.percentile(normal_scores, q))


def mean_kstd(normal_scores: np.ndarray, k: float = 3.0) -> float:
    return float(normal_scores.mean() + k * normal_scores.std())

