"""Isolation Forest baseline on statistical window features.

Score convention used by every model in this repo: HIGHER score = MORE anomalous.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.features import extract_features


class IForestDetector:
    name = "iforest"

    def __init__(self, n_estimators: int = 200, max_samples="auto", seed: int = 0):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                     random_state=seed, n_jobs=-1)

    def fit(self, X: np.ndarray) -> "IForestDetector":
        F = self.scaler.fit_transform(extract_features(X))
        self.model.fit(F)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        F = self.scaler.transform(extract_features(X))
        return -self.model.score_samples(F)  # sklearn: lower = more abnormal -> flip sign
