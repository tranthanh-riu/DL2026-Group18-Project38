import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def threshold_free(y, score):
    # Metric không cần ngưỡng: AUC-ROC và AUC-PR (nhãn 1 = fault = dương tính)
    return {"auc_roc": roc_auc_score(y, score), "auc_pr": average_precision_score(y, score)}


def at_threshold(y, score, thr):
    # Metric tại một ngưỡng: score > thr thì dự đoán là bất thường (1)
    y = np.asarray(y)
    pred = (np.asarray(score) > thr).astype(int)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    # FPR = normal bị báo nhầm / tổng normal
    num_normal = (y == 0).sum()
    false_alarms = ((pred == 1) & (y == 0)).sum()
    fpr = false_alarms / num_normal if num_normal > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def main():
    # Test với dữ liệu bịa: kỳ vọng precision 0.5, recall 0.5, FPR 0.5
    y = [0, 0, 1, 1]
    score = [0.1, 0.9, 0.8, 0.2]
    print("at_threshold:", at_threshold(y, score, 0.5))
    print("threshold_free:", threshold_free(y, score))


if __name__ == "__main__":
    main()
