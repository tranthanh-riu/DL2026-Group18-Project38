"""Hand-crafted statistical features of a vibration window (baseline input).

    from src.features import extract_features, FEATURE_NAMES
    F = extract_features(X)   # X: (N, 1, L) or (N, L)  ->  F: (N, 8)
"""
import numpy as np
from scipy.stats import kurtosis, skew

FEATURE_NAMES = ["rms", "std", "mean_abs", "peak", "peak_to_peak",
                 "crest_factor", "skewness", "kurtosis"]


def extract_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        X = X[:, 0, :]
    rms = np.sqrt((X ** 2).mean(axis=1))
    peak = np.abs(X).max(axis=1)
    feats = [
        rms,
        X.std(axis=1),
        np.abs(X).mean(axis=1),
        peak,
        X.max(axis=1) - X.min(axis=1),
        peak / np.maximum(rms, 1e-12),
        skew(X, axis=1),
        kurtosis(X, axis=1, fisher=False),
    ]
    return np.stack(feats, axis=1).astype(np.float32)
