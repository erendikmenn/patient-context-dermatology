"""Metrics explicitly distinguish AP, ROC-AUC and unstandardized high-TPR pAUC."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def high_sensitivity_pauc(y, probabilities, min_tpr: float = 0.8) -> float:
    max_fpr = 1.0 - min_tpr
    if not 0 < max_fpr <= 1:
        raise ValueError("min_tpr must be in [0, 1)")
    standardized = roc_auc_score(1 - np.asarray(y), 1 - np.asarray(probabilities), max_fpr=max_fpr)
    minimum = 0.5 * max_fpr**2
    return float(minimum + (2 * standardized - 1) * (max_fpr - minimum))


def choose_threshold(y, probabilities, target_sensitivity: float = 0.95) -> float:
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    _, tpr, thresholds = roc_curve(y, probabilities, drop_intermediate=False)
    eligible = np.flatnonzero((tpr >= target_sensitivity) & np.isfinite(thresholds))
    return float(thresholds[eligible[0]])


def classification_metrics(y, probabilities, threshold: float | None = None) -> dict:
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Metrics require both classes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predictions must be finite probabilities")
    result = {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pauc_tpr_ge_0_8_unstandardized": high_sensitivity_pauc(y, probabilities),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
    }
    if threshold is not None:
        predicted = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        result["operating_point"] = {
            "threshold": threshold,
            "sensitivity": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp)),
            "precision": float(tp / max(tp + fp, 1)),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        }
    return result
