from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute small-sample-safe binary classification metrics."""

    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    if y_true_array.shape[0] != y_score_array.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    y_pred = (y_score_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = {
            "accuracy": float(accuracy_score(y_true_array, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred)),
            "sensitivity": float(recall_score(y_true_array, y_pred, zero_division=0)),
            "specificity": float(specificity),
            "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        }
    if np.unique(y_true_array).size < 2:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        return metrics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_score_array))
        except ValueError:
            metrics["roc_auc"] = np.nan
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true_array, y_score_array))
        except ValueError:
            metrics["pr_auc"] = np.nan
    return metrics
