from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from eeg_recovery.training.ensemble_calibration import (
    best_threshold_from_scores,
    ensemble_prediction_frames,
    load_prediction_frame,
)


REQUIRED_PREDICTION_COLUMNS = {"subject_id", "fold_index", "y_true", "y_score"}


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    threshold: float
    metric: str
    metric_value: float
    threshold_method: str
    exploratory: bool
    held_out_seed: int | str | None = None
    calibration_seed_count: int | None = None


def load_required_prediction_csv(path: str | Path, *, model_name: str | None = None) -> pd.DataFrame:
    return load_prediction_frame(path, model_name=model_name)


def equal_weight_model_ensemble(
    prediction_frames: Mapping[str, pd.DataFrame],
    *,
    model_group: str,
    seed: int | str | None = None,
) -> pd.DataFrame:
    ensemble = ensemble_prediction_frames(prediction_frames).copy()
    members = "+".join(prediction_frames.keys())
    ensemble["model_group"] = model_group
    ensemble["ensemble_members"] = members
    ensemble["ensemble_weighting"] = "equal"
    if seed is not None:
        ensemble["seed"] = seed
    return ensemble


def choose_test_label_threshold(
    predictions: pd.DataFrame,
    *,
    metric: str = "balanced_accuracy",
    allow_as_primary: bool = False,
) -> ThresholdCalibrationResult:
    if allow_as_primary:
        raise ValueError("Test-label optimized thresholds are exploratory and cannot be used as primary results.")
    frame = _normalize_prediction_frame(predictions)
    selection = best_threshold_from_scores(
        frame["y_true"].to_numpy(dtype=int),
        frame["y_score"].to_numpy(dtype=float),
        metric=metric,
    )
    return ThresholdCalibrationResult(
        threshold=selection.threshold,
        metric=selection.metric,
        metric_value=selection.metric_value,
        threshold_method="test_label_optimized",
        exploratory=True,
    )


def leave_one_seed_thresholds(
    seed_prediction_frames: Mapping[int | str, pd.DataFrame],
    *,
    metric: str = "balanced_accuracy",
) -> dict[int | str, ThresholdCalibrationResult]:
    if len(seed_prediction_frames) < 2:
        raise ValueError("Leave-one-seed threshold calibration requires at least two seeds.")

    thresholds: dict[int | str, ThresholdCalibrationResult] = {}
    for held_out_seed in seed_prediction_frames:
        calibration_frames = [
            _normalize_prediction_frame(frame)
            for seed, frame in seed_prediction_frames.items()
            if seed != held_out_seed
        ]
        calibration = pd.concat(calibration_frames, ignore_index=True)
        selection = best_threshold_from_scores(
            calibration["y_true"].to_numpy(dtype=int),
            calibration["y_score"].to_numpy(dtype=float),
            metric=metric,
        )
        thresholds[held_out_seed] = ThresholdCalibrationResult(
            threshold=selection.threshold,
            metric=selection.metric,
            metric_value=selection.metric_value,
            threshold_method="leave_one_seed_out_oof",
            exploratory=False,
            held_out_seed=held_out_seed,
            calibration_seed_count=len(calibration_frames),
        )
    return thresholds


def apply_leave_one_seed_thresholds(
    seed_prediction_frames: Mapping[int | str, pd.DataFrame],
    *,
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    thresholds = leave_one_seed_thresholds(seed_prediction_frames, metric=metric)
    calibrated_frames: list[pd.DataFrame] = []
    for seed, frame in seed_prediction_frames.items():
        normalized = _normalize_prediction_frame(frame)
        threshold = thresholds[seed]
        calibrated = normalized.copy()
        calibrated["seed"] = seed
        calibrated["threshold"] = threshold.threshold
        calibrated["threshold_method"] = threshold.threshold_method
        calibrated["calibration_metric"] = threshold.metric
        calibrated["calibration_metric_value"] = threshold.metric_value
        calibrated["exploratory"] = threshold.exploratory
        calibrated["y_pred"] = (calibrated["y_score"].to_numpy(dtype=float) >= threshold.threshold).astype(int)
        calibrated_frames.append(calibrated)
    return pd.concat(calibrated_frames, ignore_index=True)


def aggregate_seed_mean_predictions(
    seed_prediction_frames: Mapping[int | str, pd.DataFrame],
    *,
    model_group: str,
    threshold: float = 0.5,
    threshold_method: str = "fixed_0.5",
    ensemble_members: str = "",
    ensemble_weighting: str = "seed_mean_probability",
) -> pd.DataFrame:
    if not seed_prediction_frames:
        raise ValueError("At least one seed prediction frame is required.")
    normalized_frames = []
    for seed, frame in seed_prediction_frames.items():
        normalized = _normalize_prediction_frame(frame).copy()
        normalized["seed"] = seed
        normalized_frames.append(normalized)
    combined = pd.concat(normalized_frames, ignore_index=True)
    label_counts = combined.groupby("subject_id")["y_true"].nunique()
    fold_counts = combined.groupby("subject_id")["fold_index"].nunique()
    if int(label_counts.max()) != 1 or int(fold_counts.max()) != 1:
        raise ValueError("Seed prediction frames contain inconsistent labels or folds.")
    aggregated = (
        combined.groupby("subject_id", as_index=False)
        .agg(
            fold_index=("fold_index", "first"),
            y_true=("y_true", "first"),
            y_score=("y_score", "mean"),
            n_seed_scores=("y_score", "size"),
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    aggregated["y_pred"] = (aggregated["y_score"].to_numpy(dtype=float) >= threshold).astype(int)
    aggregated["threshold"] = threshold
    aggregated["threshold_method"] = threshold_method
    aggregated["model_group"] = model_group
    aggregated["ensemble_members"] = ensemble_members or model_group
    aggregated["ensemble_weighting"] = ensemble_weighting
    return aggregated


def classification_metrics_from_scores(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    y_pred: Sequence[int] | np.ndarray | None = None,
) -> dict[str, float | int]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    if y_true_array.shape[0] != y_score_array.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")
    if y_pred is None:
        y_pred_array = (y_score_array >= threshold).astype(int)
    else:
        y_pred_array = np.asarray(y_pred, dtype=int)
        if y_pred_array.shape[0] != y_true_array.shape[0]:
            raise ValueError("y_pred and y_true must have the same length.")

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics: dict[str, float | int] = {
            "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
            "sensitivity": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
            "specificity": float(specificity),
            "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
            "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true_array, y_score_array)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
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


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    model_group: str,
    threshold: float = 0.5,
    threshold_method: str = "fixed_0.5",
    ensemble_members: str = "",
    ensemble_weighting: str = "none",
    seed: int | str | None = None,
    n_seeds: int = 1,
    exploratory: bool = False,
) -> dict[str, object]:
    frame = _normalize_prediction_frame(predictions)
    y_pred = predictions["y_pred"].to_numpy(dtype=int) if "y_pred" in predictions.columns else None
    metrics = classification_metrics_from_scores(
        frame["y_true"].to_numpy(dtype=int),
        frame["y_score"].to_numpy(dtype=float),
        threshold=threshold,
        y_pred=y_pred,
    )
    row: dict[str, object] = {
        "model_group": model_group,
        "seed": seed if seed is not None else "",
        "threshold": threshold,
        "threshold_method": threshold_method,
        "ensemble_members": ensemble_members or model_group,
        "ensemble_weighting": ensemble_weighting,
        "n_subjects": int(len(frame)),
        "n_seeds": int(n_seeds),
        "exploratory": bool(exploratory),
    }
    row.update(metrics)
    return row


def compute_per_subject_error_frequency(
    prediction_frames: Sequence[pd.DataFrame],
    *,
    model_group: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    if not prediction_frames:
        raise ValueError("At least one prediction frame is required.")
    frames = []
    for frame in prediction_frames:
        normalized = _normalize_prediction_frame(frame).copy()
        if "y_pred" in frame.columns:
            normalized = normalized.merge(
                frame[["subject_id", "y_pred"]].assign(subject_id=lambda item: item["subject_id"].astype(str)),
                on="subject_id",
                how="left",
                validate="one_to_one",
            )
            normalized["y_pred"] = normalized["y_pred"].astype(int)
        else:
            normalized["y_pred"] = (normalized["y_score"].to_numpy(dtype=float) >= threshold).astype(int)
        frames.append(normalized)
    combined = pd.concat(frames, ignore_index=True)
    combined["is_error"] = combined["y_true"].astype(int) != combined["y_pred"].astype(int)
    summary = (
        combined.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            n_runs=("is_error", "size"),
            n_errors=("is_error", "sum"),
            mean_y_score=("y_score", "mean"),
            std_y_score=("y_score", "std"),
            min_y_score=("y_score", "min"),
            max_y_score=("y_score", "max"),
            mean_y_pred=("y_pred", "mean"),
        )
        .sort_values(["n_errors", "subject_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["model_group"] = model_group
    summary["error_rate"] = summary["n_errors"] / summary["n_runs"]
    summary["std_y_score"] = summary["std_y_score"].fillna(0.0)
    summary["repeatedly_wrong_flag"] = (summary["error_rate"] >= 0.5).map(bool).astype(object)
    summary["borderline_label_flag"] = pd.NA
    summary["residual"] = summary["y_true"].astype(float) - summary["mean_y_score"].astype(float)
    summary["distance_to_threshold"] = (summary["mean_y_score"].astype(float) - float(threshold)).abs()
    columns = [
        "subject_id",
        "y_true",
        "model_group",
        "n_runs",
        "n_errors",
        "error_rate",
        "mean_y_score",
        "std_y_score",
        "min_y_score",
        "max_y_score",
        "mean_y_pred",
        "repeatedly_wrong_flag",
        "borderline_label_flag",
        "residual",
        "distance_to_threshold",
    ]
    return summary[columns]


def validate_locked_seed_summary(
    summary: pd.DataFrame,
    *,
    expected_n_seeds: int = 10,
    required_model_groups: Sequence[str] | None = None,
) -> None:
    required_columns = {"model_group", "aggregation", "threshold_method", "n_seeds"}
    missing = required_columns - set(summary.columns)
    if missing:
        raise ValueError(f"Summary missing required column(s): {', '.join(sorted(missing))}")
    model_groups = tuple(
        required_model_groups
        if required_model_groups is not None
        else (
            "no_ssl_stable_cnn",
            "psd_segbarlow_ssl_cnn",
            "wpli_segbarlow_ssl_cnn",
            "psd_wpli_segbarlow_equal_weight",
        )
    )
    primary = summary.loc[
        (summary["aggregation"] == "seed_mean_probability")
        & (summary["threshold_method"] == "fixed_0.5")
        & (summary["model_group"].isin(model_groups))
    ].copy()
    found = set(primary["model_group"].astype(str))
    missing_groups = sorted(set(model_groups) - found)
    if missing_groups:
        raise ValueError(f"Summary missing locked model group(s): {', '.join(missing_groups)}")
    bad = primary.loc[primary["n_seeds"].astype(int) != int(expected_n_seeds)]
    if not bad.empty:
        details = ", ".join(
            f"{row.model_group}={int(row.n_seeds)}"
            for row in bad.itertuples(index=False)
        )
        raise ValueError(f"Expected {expected_n_seeds} locked seeds for primary rows, got {details}")


def _normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction dataframe missing required column(s): {', '.join(sorted(missing))}")
    normalized = frame[["subject_id", "fold_index", "y_true", "y_score"]].copy()
    normalized["subject_id"] = normalized["subject_id"].astype(str)
    normalized["fold_index"] = normalized["fold_index"].astype(int)
    normalized["y_true"] = normalized["y_true"].astype(int)
    normalized["y_score"] = normalized["y_score"].astype(float)
    if normalized["subject_id"].duplicated().any():
        normalized = (
            normalized.groupby("subject_id", as_index=False)
            .agg(
                fold_index=("fold_index", "first"),
                y_true=("y_true", "first"),
                y_score=("y_score", "mean"),
            )
        )
    label_counts = normalized.groupby("subject_id")["y_true"].nunique()
    fold_counts = normalized.groupby("subject_id")["fold_index"].nunique()
    if int(label_counts.max()) != 1:
        raise ValueError("Prediction dataframe contains inconsistent y_true values by subject.")
    if int(fold_counts.max()) != 1:
        raise ValueError("Prediction dataframe contains inconsistent fold_index values by subject.")
    if not np.isfinite(normalized["y_score"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction dataframe contains non-finite y_score values.")
    return normalized.sort_values("subject_id").reset_index(drop=True)
