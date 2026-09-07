import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)

def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes complete set of classification, probabilistic, and calibration metrics.
    Inputs:
        y_true: ground truth binary labels (0 or 1)
        y_prob: predicted fake probabilities in range [0, 1]
        threshold: decision threshold for binary classification
    Returns:
        dict containing metrics, curves, and confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    brier = brier_score_loss(y_true, y_prob)

    # Calculate ROC curve points
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "roc_curve": {
            "fpr": fpr.tolist()[:100], # Subsample points for JSON efficiency
            "tpr": tpr.tolist()[:100]
        }
    }

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Finds threshold that maximizes F1 score on validation set.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_f1 = 0.0
    best_thresh = 0.5

    for th in np.linspace(0.1, 0.9, 81):
        y_pred = (y_prob >= th).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = th

    return float(best_thresh)
