"""Evaluation metrics. Scores: higher = more anomalous. Labels: 1 = fault, 0 = normal."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_free(score: np.ndarray, y: np.ndarray) -> dict:
    return {"auc_roc": float(roc_auc_score(y, score)),
            "auc_pr": float(average_precision_score(y, score))}


def at_threshold(score: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pred = score > thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,   # false alarms on normal windows
            "threshold": float(thr)}


def recall_by_type(score: np.ndarray, fault_type: np.ndarray, thr: float) -> dict:
    out = {}
    for ft in ["inner_race", "ball", "outer_race"]:
        m = fault_type == ft
        if m.any():
            out[f"recall_{ft}"] = float((score[m] > thr).mean())
    return out


def evaluate(score, y, fault_type, thr) -> dict:
    return {**threshold_free(score, y), **at_threshold(score, y, thr),
            **recall_by_type(score, fault_type, thr)}


def oracle_best_f1(score: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximising F1 ON THE TEST SET. Upper bound only - never a real result."""
    cands = np.unique(score)
    if len(cands) > 2000:
        cands = np.quantile(score, np.linspace(0, 1, 2001))
    f1 = [at_threshold(score, y, t)["f1"] for t in cands]
    return float(cands[int(np.argmax(f1))])
