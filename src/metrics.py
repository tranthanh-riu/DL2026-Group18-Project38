import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_scores, threshold=None):
    """
    Calculate anomaly detection metrics.
    Args:
        y_true: ground truth labels (0 for normal, 1 for anomaly)
        y_scores: continuous anomaly scores (higher is more anomalous)
        threshold: threshold to compute discrete metrics (Precision, Recall, F1, FPR)
                   If None, it will only compute AUCs.
    Returns:
        dict containing the metrics.
    """
    metrics = {}
    
    # Check if there are both positive and negative classes
    if len(np.unique(y_true)) > 1:
        metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
        metrics['auc_pr'] = average_precision_score(y_true, y_scores)
    else:
        metrics['auc_roc'] = np.nan
        metrics['auc_pr'] = np.nan
        
    if threshold is not None:
        y_pred = (y_scores >= threshold).astype(int)
        
        # Only compute Precision, Recall, F1 if we have actual anomalies or predictions
        # Or let sklearn handle it (might give warnings if true sum is 0)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # FPR on normal windows
        normal_idx = (y_true == 0)
        if np.any(normal_idx):
            fp = np.sum(y_pred[normal_idx] == 1)
            tn = np.sum(y_pred[normal_idx] == 0)
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            metrics['fpr'] = np.nan
            
    return metrics

def get_best_f1_threshold(y_true, y_scores):
    """
    Helper function to find the threshold that maximizes F1 score.
    """
    thresholds = np.unique(y_scores)
    best_f1 = -1
    best_thresh = None
    
    for th in thresholds:
        y_pred = (y_scores >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
            
    return best_thresh, best_f1
