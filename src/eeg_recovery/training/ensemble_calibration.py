from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
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

from eeg_recovery.training.metrics import binary_classification_metrics


REQUIRED_PREDICTION_COLUMNS = {"subject_id", "fold_index", "y_true", "y_score"}
THRESHOLD_METRICS = {"balanced_accuracy", "accuracy", "f1"}


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    metric: str
    metric_value: float


def load_prediction_frame(path: str | Path, *, model_name: str | None = None) -> pd.DataFrame:
    """Load a saved prediction CSV and normalize it to one row per subject."""

    prediction_path = Path(path)
    frame = pd.read_csv(prediction_path)
    normalized = _normalize_prediction_frame(frame)
    normalized["source_path"] = str(prediction_path)
    if model_name is not None:
        normalized["source_model"] = model_name
    return normalized


def ensemble_prediction_frames(
    prediction_frames: Mapping[str, pd.DataFrame],
    *,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Average model probabilities after strict subject/y_true alignment."""

    if not prediction_frames:
        raise ValueError("At least one prediction dataframe is required.")

    names = list(prediction_frames.keys())
    normalized_weights = _normalize_weights(names, weights)
    merged: pd.DataFrame | None = None
    score_columns: list[str] = []

    for name in names:
        frame = _normalize_prediction_frame(prediction_frames[name])
        score_column = f"score_{_safe_column_token(name)}"
        score_columns.append(score_column)
        model_frame = frame.rename(columns={"y_score": score_column})
        model_frame = model_frame[["subject_id", "fold_index", "y_true", score_column]]
        if merged is None:
            merged = model_frame
            continue

        merged = merged.merge(
            model_frame,
            on="subject_id",
            how="inner",
            suffixes=("", f"_{_safe_column_token(name)}"),
            validate="one_to_one",
        )
        other_y_true = f"y_true_{_safe_column_token(name)}"
        other_fold_index = f"fold_index_{_safe_column_token(name)}"
        if len(merged) != len(frame):
            raise ValueError("Prediction dataframes must contain identical subject_id sets.")
        if not (merged["y_true"].to_numpy(dtype=int) == merged[other_y_true].to_numpy(dtype=int)).all():
            raise ValueError("Prediction dataframes contain inconsistent y_true values.")
        if not (
            merged["fold_index"].to_numpy(dtype=int)
            == merged[other_fold_index].to_numpy(dtype=int)
        ).all():
            raise ValueError("Prediction dataframes contain inconsistent fold_index values.")
        merged = merged.drop(columns=[other_y_true, other_fold_index])

    assert merged is not None
    if set(merged["subject_id"]) != set(_normalize_prediction_frame(prediction_frames[names[0]])["subject_id"]):
        raise ValueError("Prediction dataframes must contain identical subject_id sets.")

    score_matrix = merged[score_columns].to_numpy(dtype=float)
    merged["y_score"] = np.average(score_matrix, axis=1, weights=normalized_weights)
    merged["ensemble_size"] = len(names)
    for name, weight in zip(names, normalized_weights):
        merged[f"weight_{_safe_column_token(name)}"] = float(weight)
    return merged.sort_values("subject_id").reset_index(drop=True)


def best_threshold_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "balanced_accuracy",
) -> ThresholdSelection:
    """Choose a deterministic threshold by maximizing a small-sample metric."""

    if metric not in THRESHOLD_METRICS:
        raise ValueError(f"Unsupported threshold metric: {metric}")
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    if y_true_array.shape[0] != y_score_array.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")
    if y_true_array.size == 0:
        raise ValueError("At least one sample is required for threshold selection.")
    if not np.isfinite(y_score_array).all():
        raise ValueError("y_score contains non-finite values.")

    best: ThresholdSelection | None = None
    for threshold in _candidate_thresholds(y_score_array):
        y_pred = (y_score_array >= threshold).astype(int)
        metric_value = _score_threshold_metric(y_true_array, y_pred, metric)
        candidate = ThresholdSelection(
            threshold=float(threshold),
            metric=metric,
            metric_value=float(metric_value),
        )
        if best is None or _threshold_sort_key(candidate) > _threshold_sort_key(best):
            best = candidate

    assert best is not None
    return best


def apply_oof_threshold_calibration(
    predictions: pd.DataFrame,
    *,
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Apply leave-one-subject-out threshold calibration to saved OOF scores."""

    frame = _normalize_prediction_frame(predictions)
    if len(frame) < 2:
        raise ValueError("At least two subjects are required for OOF threshold calibration.")

    thresholds: list[float] = []
    threshold_scores: list[float] = []
    for subject_id in frame["subject_id"].tolist():
        calibration = frame.loc[frame["subject_id"] != subject_id]
        selection = best_threshold_from_scores(
            calibration["y_true"].to_numpy(dtype=int),
            calibration["y_score"].to_numpy(dtype=float),
            metric=metric,
        )
        thresholds.append(selection.threshold)
        threshold_scores.append(selection.metric_value)

    calibrated = frame.copy()
    calibrated["calibrated_threshold"] = thresholds
    calibrated["calibration_metric"] = metric
    calibrated["calibration_metric_value"] = threshold_scores
    calibrated["y_pred"] = (
        calibrated["y_score"].to_numpy(dtype=float)
        >= calibrated["calibrated_threshold"].to_numpy(dtype=float)
    ).astype(int)
    return calibrated.sort_values("subject_id").reset_index(drop=True)


def apply_oof_ensemble_weight_threshold_calibration(
    prediction_frames: Mapping[str, pd.DataFrame],
    *,
    weight_grid: list[float] | np.ndarray | None = None,
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Select two-model ensemble weight and threshold using only other subjects."""

    names = list(prediction_frames.keys())
    if len(names) != 2:
        raise ValueError("OOF ensemble weight calibration currently supports exactly two models.")
    grid = np.asarray(weight_grid if weight_grid is not None else np.linspace(0.0, 1.0, 11), dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("weight_grid must be a non-empty one-dimensional sequence.")
    if ((grid < 0.0) | (grid > 1.0)).any():
        raise ValueError("weight_grid values must be between 0 and 1.")

    base = ensemble_prediction_frames(prediction_frames)
    first_name, second_name = names
    first_column = f"score_{_safe_column_token(first_name)}"
    second_column = f"score_{_safe_column_token(second_name)}"
    first_scores = base[first_column].to_numpy(dtype=float)
    second_scores = base[second_column].to_numpy(dtype=float)
    y_true = base["y_true"].to_numpy(dtype=int)

    calibrated_scores: list[float] = []
    thresholds: list[float] = []
    threshold_scores: list[float] = []
    first_weights: list[float] = []
    second_weights: list[float] = []
    y_pred: list[int] = []
    indices = np.arange(len(base))
    for row_index in indices:
        calibration_mask = indices != row_index
        best: tuple[tuple[float, float, float, float], float, ThresholdSelection] | None = None
        for first_weight in grid:
            second_weight = 1.0 - float(first_weight)
            calibration_scores = (
                float(first_weight) * first_scores[calibration_mask]
                + second_weight * second_scores[calibration_mask]
            )
            selection = best_threshold_from_scores(
                y_true[calibration_mask],
                calibration_scores,
                metric=metric,
            )
            key = (
                selection.metric_value,
                -abs(float(first_weight) - 0.5),
                -abs(selection.threshold - 0.5),
                -selection.threshold,
            )
            if best is None or key > best[0]:
                best = (key, float(first_weight), selection)

        assert best is not None
        _, selected_first_weight, selected_threshold = best
        selected_second_weight = 1.0 - selected_first_weight
        score = (
            selected_first_weight * first_scores[row_index]
            + selected_second_weight * second_scores[row_index]
        )
        first_weights.append(float(selected_first_weight))
        second_weights.append(float(selected_second_weight))
        thresholds.append(selected_threshold.threshold)
        threshold_scores.append(selected_threshold.metric_value)
        calibrated_scores.append(float(score))
        y_pred.append(int(score >= selected_threshold.threshold))

    calibrated = base[["subject_id", "fold_index", "y_true", first_column, second_column]].copy()
    calibrated["y_score"] = calibrated_scores
    calibrated[f"calibrated_weight_{_safe_column_token(first_name)}"] = first_weights
    calibrated[f"calibrated_weight_{_safe_column_token(second_name)}"] = second_weights
    calibrated["calibrated_threshold"] = thresholds
    calibrated["calibration_metric"] = metric
    calibrated["calibration_metric_value"] = threshold_scores
    calibrated["y_pred"] = y_pred
    return calibrated.sort_values("subject_id").reset_index(drop=True)


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    model: str,
    threshold: float | None = 0.5,
    y_pred_column: str | None = None,
    decision_rule: str | None = None,
) -> dict[str, float | int | str]:
    """Evaluate fixed-threshold or precomputed per-row-threshold predictions."""

    frame = _normalize_prediction_frame(predictions)
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_score = frame["y_score"].to_numpy(dtype=float)
    if y_pred_column is None:
        if threshold is None:
            raise ValueError("threshold is required when y_pred_column is not provided.")
        metrics = binary_classification_metrics(y_true, y_score, threshold=threshold)
        resolved_threshold: float | str = float(threshold)
    else:
        if y_pred_column not in predictions.columns:
            raise ValueError(f"Missing y_pred column: {y_pred_column}")
        y_pred_frame = predictions[["subject_id", y_pred_column]].copy()
        y_pred_frame["subject_id"] = y_pred_frame["subject_id"].astype(str)
        if y_pred_frame["subject_id"].duplicated().any():
            raise ValueError("Precomputed predictions must contain one y_pred row per subject.")
        frame = frame.merge(y_pred_frame, on="subject_id", how="left", validate="one_to_one")
        if frame[y_pred_column].isna().any():
            raise ValueError(f"Missing {y_pred_column} value for at least one subject.")
        y_pred = frame[y_pred_column].to_numpy(dtype=int)
        metrics = _metrics_from_predictions(y_true, y_score, y_pred)
        resolved_threshold = "per_subject"

    row: dict[str, float | int | str] = {
        "model": model,
        "decision_rule": decision_rule or ("fixed" if y_pred_column is None else "oof_calibrated"),
        "threshold": resolved_threshold,
        "n_subjects": int(len(frame)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }
    row.update(metrics)
    return row


def _normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction dataframe missing required column(s): {', '.join(sorted(missing))}")

    normalized = frame[["subject_id", "fold_index", "y_true", "y_score"]].copy()
    normalized["subject_id"] = normalized["subject_id"].astype(str)
    normalized["fold_index"] = normalized["fold_index"].astype(int)
    normalized["y_true"] = normalized["y_true"].astype(int)
    normalized["y_score"] = normalized["y_score"].astype(float)
    if not np.isfinite(normalized["y_score"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction dataframe contains non-finite y_score values.")

    label_counts = normalized.groupby("subject_id")["y_true"].nunique()
    if int(label_counts.max()) != 1:
        raise ValueError("Prediction dataframe contains inconsistent y_true values by subject.")
    fold_counts = normalized.groupby("subject_id")["fold_index"].nunique()
    if int(fold_counts.max()) != 1:
        raise ValueError("Prediction dataframe contains inconsistent fold_index values by subject.")

    if normalized["subject_id"].duplicated().any():
        normalized = (
            normalized.groupby("subject_id", as_index=False)
            .agg(
                fold_index=("fold_index", "first"),
                y_true=("y_true", "first"),
                y_score=("y_score", "mean"),
            )
        )
    return normalized.sort_values("subject_id").reset_index(drop=True)


def _normalize_weights(names: list[str], weights: Mapping[str, float] | None) -> np.ndarray:
    if weights is None:
        return np.full(len(names), 1.0 / len(names), dtype=float)
    missing = set(names) - set(weights)
    extra = set(weights) - set(names)
    if missing or extra:
        raise ValueError(
            "Weights must match prediction names exactly. "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    values = np.asarray([float(weights[name]) for name in names], dtype=float)
    if (values < 0).any():
        raise ValueError("Ensemble weights must be non-negative.")
    weight_sum = float(values.sum())
    if weight_sum <= 0:
        raise ValueError("At least one ensemble weight must be positive.")
    return values / weight_sum


def _candidate_thresholds(scores: np.ndarray) -> list[float]:
    unique_scores = np.unique(scores.astype(float))
    candidates: set[float] = {0.0, 0.5, 1.0}
    candidates.update(float(value) for value in unique_scores)
    if unique_scores.size > 1:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidates.update(float(value) for value in midpoints)
    candidates.add(float(np.nextafter(float(unique_scores.max()), np.inf)))
    candidates.add(float(np.nextafter(float(unique_scores.min()), -np.inf)))
    return sorted(value for value in candidates if np.isfinite(value))


def _score_threshold_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        if metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric == "f1":
            return float(f1_score(y_true, y_pred, zero_division=0))
    raise ValueError(f"Unsupported threshold metric: {metric}")


def _threshold_sort_key(selection: ThresholdSelection) -> tuple[float, float, float]:
    return (
        selection.metric_value,
        -abs(selection.threshold - 0.5),
        -selection.threshold,
    )


def _metrics_from_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(specificity),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
    if np.unique(y_true).size < 2:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        return metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = np.nan
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            metrics["pr_auc"] = np.nan
    return metrics


def _safe_column_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return token or "model"
