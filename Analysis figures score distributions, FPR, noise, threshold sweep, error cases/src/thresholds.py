"""Detection-threshold strategies. All fitted on NORMAL scores only, never on test data.

- percentile:      q-th percentile of normal 0 HP val scores (main pipeline, q = 99)
- mean_kstd:       mean + k * std of normal 0 HP val scores
- pot:             Peaks-Over-Threshold (extreme value theory, Siffer et al. 2017)
- condition-aware: percentile of the normal calibration scores of the TEST load
                   (uses a small amount of unlabelled-normal data from the new condition)
"""
import numpy as np
from scipy.stats import genpareto


def percentile(normal_scores: np.ndarray, q: float = 99.0) -> float:
    return float(np.percentile(normal_scores, q))


def mean_kstd(normal_scores: np.ndarray, k: float = 3.0) -> float:
    return float(normal_scores.mean() + k * normal_scores.std())


def pot(normal_scores: np.ndarray, init_q: float = 98.0, risk: float = 1e-3) -> float:
    """Fit a Generalised Pareto Distribution to the excesses over the init_q-th percentile
    and return the score whose exceedance probability is `risk`."""
    s = np.asarray(normal_scores, dtype=float)
    t = np.percentile(s, init_q)
    excess = s[s > t] - t
    if len(excess) < 2:
        return float(s.max())
    c, _, scale = genpareto.fit(excess, floc=0)
    r = risk * len(s) / len(excess)
    if abs(c) < 1e-8:
        return float(t - scale * np.log(r))
    return float(t + scale / c * (r ** (-c) - 1))


def condition_aware(calib_scores: np.ndarray, q: float = 99.0) -> float:
    """Percentile of the normal calibration scores of the load being tested."""
    return percentile(calib_scores, q)
