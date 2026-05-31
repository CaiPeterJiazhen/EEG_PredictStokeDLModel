from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eeg_recovery.training.metrics import binary_classification_metrics


@dataclass(frozen=True)
class EEGSummaryCalibrationConfig:
    candidate_cs: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1)
    candidate_weights: tuple[float, ...] = tuple(float(value) for value in np.linspace(0.0, 0.9, 19))
    candidate_fusion_modes: tuple[str, ...] = ("linear",)
    candidate_feature_directions: tuple[int, ...] = (1, -1)
    candidate_score_transform_scales: tuple[float, ...] = (1.0,)
    candidate_thresholds: tuple[float, ...] | None = None
    selection_objective: str = "rank"
    threshold: float = 0.5
    preserve_base_classification: bool = False
    min_accuracy_delta: float = 0.0
    min_balanced_accuracy_delta: float = 0.0
    min_sensitivity_delta: float = -1.0
    min_specificity_delta: float = -1.0
    random_state: int = 0


def nested_eeg_summary_residual_calibration(
    base_predictions: pd.DataFrame,
    summary_features: np.ndarray,
    summary_feature_names: Iterable[str],
    config: EEGSummaryCalibrationConfig = EEGSummaryCalibrationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_calibration_inputs(base_predictions, summary_features, tuple(summary_feature_names), config)
    frame = base_predictions.reset_index(drop=True).copy()
    y_true = frame["y_true"].to_numpy(dtype=int)
    base_scores = frame["y_score"].to_numpy(dtype=float)
    features = np.asarray(summary_features, dtype=np.float32)
    feature_names = tuple(summary_feature_names)

    calibrated_scores: list[float] = []
    summary_scores: list[float] = []
    selected_thresholds: list[float] = []
    selected_fusion_modes: list[str] = []
    selected_score_transform_scales: list[float] = []
    choices: list[dict[str, object]] = []
    coefficients: list[np.ndarray] = []
    patient_indices = np.arange(len(frame))
    for test_index in patient_indices:
        outer_train_mask = patient_indices != test_index
        selected = _select_calibration_hyperparameters(
            features,
            y_true,
            base_scores,
            outer_train_mask,
            config,
        )
        summary_score, coefficient = _fit_summary_model_and_score(
            features,
            y_true,
            train_mask=outer_train_mask,
            test_index=test_index,
            c_value=selected["selected_c"],
            random_state=config.random_state,
        )
        blended_score = _fuse_base_summary_scores(
            base_scores[test_index],
            summary_score,
            weight=selected["selected_weight"],
            fusion_mode=str(selected["selected_fusion_mode"]),
        )
        transformed_score = _transform_probability_scores(
            blended_score,
            logit_scale=float(selected["selected_score_transform_scale"]),
        )
        calibrated_scores.append(float(transformed_score))
        summary_scores.append(float(summary_score))
        selected_thresholds.append(float(selected["selected_threshold"]))
        selected_fusion_modes.append(str(selected["selected_fusion_mode"]))
        selected_score_transform_scales.append(float(selected["selected_score_transform_scale"]))
        coefficients.append(coefficient)
        choices.append(
            {
                "subject_id": frame.loc[test_index, "subject_id"],
                "selected_c": selected["selected_c"],
                "selected_weight": selected["selected_weight"],
                "selected_fusion_mode": selected["selected_fusion_mode"],
                "selected_score_transform_scale": selected["selected_score_transform_scale"],
                "selected_threshold": selected["selected_threshold"],
                "fallback_to_base": selected["fallback_to_base"],
                "selection_objective": config.selection_objective,
                "inner_selection_score": selected["inner_selection_score"],
            }
        )

    predictions = frame.copy()
    predictions["base_y_score"] = base_scores
    predictions["summary_y_score"] = summary_scores
    predictions["calibrated_y_score"] = calibrated_scores
    predictions["selected_fusion_mode"] = selected_fusion_modes
    predictions["selected_score_transform_scale"] = selected_score_transform_scales
    predictions["selected_threshold"] = selected_thresholds
    predictions["y_score"] = predictions["calibrated_y_score"]
    predictions["y_pred"] = (predictions["calibrated_y_score"] >= predictions["selected_threshold"]).astype(int)
    threshold_strategy = "inner_oof" if config.candidate_thresholds is not None else "fixed"
    metrics = pd.DataFrame(
        [
            {
                "model": "eeg_summary_nested_calibrator",
                "n_patients": len(predictions),
                "selection_objective": config.selection_objective,
                "threshold": float(np.mean(selected_thresholds)),
                "threshold_strategy": threshold_strategy,
                **_binary_classification_metrics_from_predictions(
                    predictions["y_true"].to_numpy(dtype=int),
                    predictions["calibrated_y_score"].to_numpy(dtype=float),
                    predictions["y_pred"].to_numpy(dtype=int),
                ),
            }
        ]
    )
    choices_frame = pd.DataFrame(choices)
    importance = _summarize_coefficients(feature_names, np.vstack(coefficients))
    return predictions, metrics, choices_frame, importance


def nested_univariate_eeg_summary_residual_calibration(
    base_predictions: pd.DataFrame,
    summary_features: np.ndarray,
    summary_feature_names: Iterable[str],
    config: EEGSummaryCalibrationConfig = EEGSummaryCalibrationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_names = tuple(summary_feature_names)
    _validate_calibration_inputs(base_predictions, summary_features, feature_names, config)
    frame = base_predictions.reset_index(drop=True).copy()
    y_true = frame["y_true"].to_numpy(dtype=int)
    base_scores = frame["y_score"].to_numpy(dtype=float)
    features = np.asarray(summary_features, dtype=np.float32)

    calibrated_scores: list[float] = []
    summary_scores: list[float] = []
    selected_thresholds: list[float] = []
    selected_fusion_modes: list[str] = []
    selected_score_transform_scales: list[float] = []
    selected_feature_names: list[str] = []
    selected_feature_directions: list[int] = []
    choices: list[dict[str, object]] = []
    patient_indices = np.arange(len(frame))
    for test_index in patient_indices:
        outer_train_mask = patient_indices != test_index
        selected = _select_univariate_calibration_hyperparameters(
            features,
            feature_names,
            y_true,
            base_scores,
            outer_train_mask,
            config,
        )
        feature_index = int(selected["selected_feature_index"])
        direction = int(selected["selected_feature_direction"])
        summary_score = _univariate_feature_score_for_test(
            features[:, feature_index],
            train_mask=outer_train_mask,
            test_index=int(test_index),
            direction=direction,
        )
        blended_score = _fuse_base_summary_scores(
            base_scores[test_index],
            summary_score,
            weight=float(selected["selected_weight"]),
            fusion_mode=str(selected["selected_fusion_mode"]),
        )
        transformed_score = _transform_probability_scores(
            blended_score,
            logit_scale=float(selected["selected_score_transform_scale"]),
        )
        calibrated_scores.append(float(transformed_score))
        summary_scores.append(float(summary_score))
        selected_thresholds.append(float(selected["selected_threshold"]))
        selected_fusion_modes.append(str(selected["selected_fusion_mode"]))
        selected_score_transform_scales.append(float(selected["selected_score_transform_scale"]))
        selected_feature_names.append(str(selected["selected_feature_name"]))
        selected_feature_directions.append(direction)
        choices.append(
            {
                "subject_id": frame.loc[test_index, "subject_id"],
                "selected_feature_index": feature_index,
                "selected_feature_name": selected["selected_feature_name"],
                "selected_feature_direction": direction,
                "selected_weight": selected["selected_weight"],
                "selected_fusion_mode": selected["selected_fusion_mode"],
                "selected_score_transform_scale": selected["selected_score_transform_scale"],
                "selected_threshold": selected["selected_threshold"],
                "fallback_to_base": selected["fallback_to_base"],
                "selection_objective": config.selection_objective,
                "inner_selection_score": selected["inner_selection_score"],
            }
        )

    predictions = frame.copy()
    predictions["base_y_score"] = base_scores
    predictions["summary_y_score"] = summary_scores
    predictions["calibrated_y_score"] = calibrated_scores
    predictions["selected_feature_name"] = selected_feature_names
    predictions["selected_feature_direction"] = selected_feature_directions
    predictions["selected_weight"] = [choice["selected_weight"] for choice in choices]
    predictions["selected_fusion_mode"] = selected_fusion_modes
    predictions["selected_score_transform_scale"] = selected_score_transform_scales
    predictions["selected_threshold"] = selected_thresholds
    predictions["y_score"] = predictions["calibrated_y_score"]
    predictions["y_pred"] = (predictions["calibrated_y_score"] >= predictions["selected_threshold"]).astype(int)
    threshold_strategy = "inner_oof" if config.candidate_thresholds is not None else "fixed"
    metrics = pd.DataFrame(
        [
            {
                "model": "eeg_summary_univariate_nested_calibrator",
                "n_patients": len(predictions),
                "selection_objective": config.selection_objective,
                "threshold": float(np.mean(selected_thresholds)),
                "threshold_strategy": threshold_strategy,
                **_binary_classification_metrics_from_predictions(
                    predictions["y_true"].to_numpy(dtype=int),
                    predictions["calibrated_y_score"].to_numpy(dtype=float),
                    predictions["y_pred"].to_numpy(dtype=int),
                ),
            }
        ]
    )
    choices_frame = pd.DataFrame(choices)
    importance = _summarize_univariate_choices(choices_frame)
    return predictions, metrics, choices_frame, importance


def select_summary_feature_columns(
    features: np.ndarray,
    feature_names: Iterable[str],
    *,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(feature_names)
    feature_matrix = np.asarray(features, dtype=np.float32)
    if feature_matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array.")
    if feature_matrix.shape[1] != len(names):
        raise ValueError("feature_names length must match features columns.")
    include_pattern = re.compile(include_regex) if include_regex else None
    exclude_pattern = re.compile(exclude_regex) if exclude_regex else None
    indices = []
    for index, name in enumerate(names):
        if include_pattern is not None and include_pattern.search(name) is None:
            continue
        if exclude_pattern is not None and exclude_pattern.search(name) is not None:
            continue
        indices.append(index)
    if not indices:
        raise ValueError("No EEG summary features remain after regex filtering.")
    selected = feature_matrix[:, indices]
    selected_names = tuple(names[index] for index in indices)
    return selected, selected_names


def _validate_calibration_inputs(
    base_predictions: pd.DataFrame,
    summary_features: np.ndarray,
    summary_feature_names: tuple[str, ...],
    config: EEGSummaryCalibrationConfig,
) -> None:
    missing = {"subject_id", "y_true", "y_score"} - set(base_predictions.columns)
    if missing:
        raise ValueError(f"base_predictions missing required column(s): {', '.join(sorted(missing))}")
    if len(base_predictions) != int(np.asarray(summary_features).shape[0]):
        raise ValueError("base_predictions and summary_features must have the same number of rows.")
    if np.asarray(summary_features).ndim != 2:
        raise ValueError("summary_features must be a two-dimensional array.")
    if np.asarray(summary_features).shape[1] != len(summary_feature_names):
        raise ValueError("summary_feature_names length must match summary_features columns.")
    if len(base_predictions) < 4:
        raise ValueError("Nested calibration requires at least four patients.")
    if not config.candidate_cs or any(value <= 0 for value in config.candidate_cs):
        raise ValueError("candidate_cs must contain positive values.")
    if not config.candidate_weights or any(value < 0 or value > 1 for value in config.candidate_weights):
        raise ValueError("candidate_weights must be within [0, 1].")
    valid_fusion_modes = {"linear", "geometric", "veto"}
    if not config.candidate_fusion_modes or any(value not in valid_fusion_modes for value in config.candidate_fusion_modes):
        raise ValueError("candidate_fusion_modes must contain only linear, geometric, or veto.")
    if not config.candidate_feature_directions or any(value not in {-1, 1} for value in config.candidate_feature_directions):
        raise ValueError("candidate_feature_directions must contain only -1 or 1.")
    if not config.candidate_score_transform_scales or any(value <= 0 for value in config.candidate_score_transform_scales):
        raise ValueError("candidate_score_transform_scales must contain positive values.")
    if config.selection_objective not in {"brier", "rank", "aucpr", "balanced_accuracy"}:
        raise ValueError("selection_objective must be one of brier, rank, aucpr, balanced_accuracy.")
    if not 0 <= config.threshold <= 1:
        raise ValueError("threshold must be within [0, 1].")
    if config.candidate_thresholds is not None and (
        not config.candidate_thresholds or any(value < 0 or value > 1 for value in config.candidate_thresholds)
    ):
        raise ValueError("candidate_thresholds must be within [0, 1].")


def _select_calibration_hyperparameters(
    features: np.ndarray,
    y_true: np.ndarray,
    base_scores: np.ndarray,
    outer_train_mask: np.ndarray,
    config: EEGSummaryCalibrationConfig,
) -> dict[str, float | str | bool]:
    train_indices = np.where(outer_train_mask)[0]
    best: tuple[float, float, float, float, str, float, float] | None = None
    base_metrics = binary_classification_metrics(
        y_true[train_indices],
        base_scores[train_indices],
        threshold=config.threshold,
    )
    for c_value in config.candidate_cs:
        inner_scores = _inner_oof_summary_scores(
            features,
            y_true,
            outer_train_mask,
            c_value=c_value,
            random_state=config.random_state,
        )
        for weight in config.candidate_weights:
            for fusion_mode in config.candidate_fusion_modes:
                blended = _fuse_base_summary_scores(
                    base_scores[train_indices],
                    inner_scores,
                    weight=weight,
                    fusion_mode=fusion_mode,
                )
                for scale in config.candidate_score_transform_scales:
                    transformed = _transform_probability_scores(blended, logit_scale=scale)
                    for threshold in _candidate_thresholds(config):
                        metrics = binary_classification_metrics(y_true[train_indices], transformed, threshold=threshold)
                        if config.preserve_base_classification and not _classification_constraints_satisfied(
                            metrics,
                            base_metrics,
                            config,
                        ):
                            continue
                        objective = _selection_score(metrics, config.selection_objective)
                        tie_breaker = metrics["balanced_accuracy"] - metrics["brier_score"]
                        if best is None or objective > best[0] or (objective == best[0] and tie_breaker > best[1]):
                            best = (
                                objective,
                                tie_breaker,
                                float(c_value),
                                float(weight),
                                fusion_mode,
                                float(scale),
                                float(threshold),
                            )
    if best is None:
        return {
            "inner_selection_score": _selection_score(base_metrics, config.selection_objective),
            "selected_c": float(config.candidate_cs[0]),
            "selected_weight": 0.0,
            "selected_fusion_mode": "linear",
            "selected_score_transform_scale": 1.0,
            "selected_threshold": float(config.threshold),
            "fallback_to_base": True,
        }
    return {
        "inner_selection_score": best[0],
        "selected_c": best[2],
        "selected_weight": best[3],
        "selected_fusion_mode": best[4],
        "selected_score_transform_scale": best[5],
        "selected_threshold": best[6],
        "fallback_to_base": False,
    }


def _select_univariate_calibration_hyperparameters(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    y_true: np.ndarray,
    base_scores: np.ndarray,
    outer_train_mask: np.ndarray,
    config: EEGSummaryCalibrationConfig,
) -> dict[str, float | int | str | bool]:
    train_indices = np.where(outer_train_mask)[0]
    base_metrics = binary_classification_metrics(
        y_true[train_indices],
        base_scores[train_indices],
        threshold=config.threshold,
    )
    best: tuple[float, float, int, int, float, str, float, float] | None = None
    for feature_index, feature_name in enumerate(feature_names):
        raw_feature = np.asarray(features[:, feature_index], dtype=float)
        if not np.isfinite(raw_feature[train_indices]).all() or np.nanstd(raw_feature[train_indices]) < 1e-9:
            continue
        for direction in config.candidate_feature_directions:
            inner_scores = _inner_oof_univariate_feature_scores(
                raw_feature,
                outer_train_mask,
                direction=direction,
            )
            for weight in config.candidate_weights:
                for fusion_mode in config.candidate_fusion_modes:
                    blended = _fuse_base_summary_scores(
                        base_scores[train_indices],
                        inner_scores,
                        weight=weight,
                        fusion_mode=fusion_mode,
                    )
                    for scale in config.candidate_score_transform_scales:
                        transformed = _transform_probability_scores(blended, logit_scale=scale)
                        for threshold in _candidate_thresholds(config):
                            metrics = binary_classification_metrics(y_true[train_indices], transformed, threshold=threshold)
                            if config.preserve_base_classification and not _classification_constraints_satisfied(
                                metrics,
                                base_metrics,
                                config,
                            ):
                                continue
                            objective = _selection_score(metrics, config.selection_objective)
                            tie_breaker = metrics["balanced_accuracy"] - metrics["brier_score"]
                            if best is None or objective > best[0] or (objective == best[0] and tie_breaker > best[1]):
                                best = (
                                    objective,
                                    tie_breaker,
                                    int(feature_index),
                                    int(direction),
                                    float(weight),
                                    str(fusion_mode),
                                    float(scale),
                                    float(threshold),
                                )
    if best is None:
        return {
            "inner_selection_score": _selection_score(base_metrics, config.selection_objective),
            "selected_feature_index": 0,
            "selected_feature_name": feature_names[0],
            "selected_feature_direction": 1,
            "selected_weight": 0.0,
            "selected_fusion_mode": "linear",
            "selected_score_transform_scale": 1.0,
            "selected_threshold": float(config.threshold),
            "fallback_to_base": True,
        }
    return {
        "inner_selection_score": best[0],
        "selected_feature_index": best[2],
        "selected_feature_name": feature_names[best[2]],
        "selected_feature_direction": best[3],
        "selected_weight": best[4],
        "selected_fusion_mode": best[5],
        "selected_score_transform_scale": best[6],
        "selected_threshold": best[7],
        "fallback_to_base": False,
    }


def _inner_oof_univariate_feature_scores(
    raw_feature: np.ndarray,
    outer_train_mask: np.ndarray,
    *,
    direction: int,
) -> np.ndarray:
    train_indices = np.where(outer_train_mask)[0]
    scores = np.zeros(len(train_indices), dtype=float)
    for output_index, validation_index in enumerate(train_indices):
        inner_train_mask = outer_train_mask.copy()
        inner_train_mask[validation_index] = False
        scores[output_index] = _univariate_feature_score_for_test(
            raw_feature,
            train_mask=inner_train_mask,
            test_index=int(validation_index),
            direction=direction,
        )
    return scores


def _univariate_feature_score_for_test(
    raw_feature: np.ndarray,
    *,
    train_mask: np.ndarray,
    test_index: int,
    direction: int,
) -> float:
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1.")
    train_values = np.asarray(raw_feature[train_mask], dtype=float)
    test_value = float(np.asarray(raw_feature, dtype=float)[test_index])
    if train_values.size == 0:
        return 0.5
    transformed_train = direction * train_values
    transformed_test = direction * test_value
    less = float(np.sum(transformed_train < transformed_test))
    equal = float(np.sum(transformed_train == transformed_test))
    score = (less + 0.5 * equal) / float(train_values.size)
    return float(np.clip(score, 0.0, 1.0))


def _fuse_base_summary_scores(
    base_scores: np.ndarray | float,
    summary_scores: np.ndarray | float,
    *,
    weight: float,
    fusion_mode: str,
) -> np.ndarray:
    base = np.asarray(base_scores, dtype=float)
    summary = np.asarray(summary_scores, dtype=float)
    if base.shape != summary.shape:
        try:
            base, summary = np.broadcast_arrays(base, summary)
        except ValueError as exc:
            raise ValueError("base_scores and summary_scores must be broadcastable.") from exc
    if not 0 <= weight <= 1:
        raise ValueError("weight must be within [0, 1].")
    if fusion_mode == "linear":
        fused = (1.0 - weight) * base + weight * summary
    elif fusion_mode == "geometric":
        eps = 1e-7
        fused = np.exp((1.0 - weight) * np.log(np.clip(base, eps, 1.0)) + weight * np.log(np.clip(summary, eps, 1.0)))
    elif fusion_mode == "veto":
        lower_support = np.minimum(base, summary)
        fused = (1.0 - weight) * base + weight * lower_support
    else:
        raise ValueError("fusion_mode must be one of linear, geometric, or veto.")
    return np.clip(fused, 0.0, 1.0)


def _transform_probability_scores(
    scores: np.ndarray | float,
    *,
    logit_scale: float,
) -> np.ndarray:
    if logit_scale <= 0:
        raise ValueError("logit_scale must be positive.")
    score_array = np.asarray(scores, dtype=float)
    if np.isclose(float(logit_scale), 1.0):
        return np.clip(score_array, 0.0, 1.0)
    eps = 1e-7
    clipped = np.clip(score_array, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped))
    transformed = 1.0 / (1.0 + np.exp(-float(logit_scale) * logits))
    return np.clip(transformed, 0.0, 1.0)


def _candidate_thresholds(config: EEGSummaryCalibrationConfig) -> tuple[float, ...]:
    if config.candidate_thresholds is None:
        return (float(config.threshold),)
    return tuple(float(value) for value in config.candidate_thresholds)


def _classification_constraints_satisfied(
    candidate_metrics: dict[str, float],
    base_metrics: dict[str, float],
    config: EEGSummaryCalibrationConfig,
) -> bool:
    if candidate_metrics["accuracy"] + 1e-12 < base_metrics["accuracy"] + config.min_accuracy_delta:
        return False
    if (
        candidate_metrics["balanced_accuracy"] + 1e-12
        < base_metrics["balanced_accuracy"] + config.min_balanced_accuracy_delta
    ):
        return False
    if candidate_metrics["sensitivity"] + 1e-12 < base_metrics["sensitivity"] + config.min_sensitivity_delta:
        return False
    if candidate_metrics["specificity"] + 1e-12 < base_metrics["specificity"] + config.min_specificity_delta:
        return False
    return True


def _inner_oof_summary_scores(
    features: np.ndarray,
    y_true: np.ndarray,
    outer_train_mask: np.ndarray,
    *,
    c_value: float,
    random_state: int,
) -> np.ndarray:
    train_indices = np.where(outer_train_mask)[0]
    scores = np.zeros(len(train_indices), dtype=float)
    for output_index, validation_index in enumerate(train_indices):
        inner_train_mask = outer_train_mask.copy()
        inner_train_mask[validation_index] = False
        score, _ = _fit_summary_model_and_score(
            features,
            y_true,
            train_mask=inner_train_mask,
            test_index=validation_index,
            c_value=c_value,
            random_state=random_state,
        )
        scores[output_index] = score
    return scores


def _fit_summary_model_and_score(
    features: np.ndarray,
    y_true: np.ndarray,
    *,
    train_mask: np.ndarray,
    test_index: int,
    c_value: float,
    random_state: int,
) -> tuple[float, np.ndarray]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(c_value),
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(features[train_mask], y_true[train_mask])
    score = float(model.predict_proba(features[[test_index]])[0, 1])
    coefficient = model.named_steps["logistic"].coef_[0].astype(float)
    return score, coefficient


def _selection_score(metrics: dict[str, float], objective: str) -> float:
    if objective == "brier":
        return -metrics["brier_score"]
    if objective == "rank":
        return metrics["roc_auc"] + metrics["pr_auc"] - metrics["brier_score"]
    if objective == "aucpr":
        return metrics["roc_auc"] + metrics["pr_auc"]
    if objective == "balanced_accuracy":
        return metrics["balanced_accuracy"]
    raise ValueError(f"Unsupported selection objective: {objective}")


def _binary_classification_metrics_from_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=int)
    if y_true_array.shape[0] != y_score_array.shape[0] or y_true_array.shape[0] != y_pred_array.shape[0]:
        raise ValueError("y_true, y_score, and y_pred must have the same length.")
    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = {
            "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
            "sensitivity": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
            "specificity": float(specificity),
            "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
            "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true_array, y_score_array)),
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


def _summarize_coefficients(feature_names: tuple[str, ...], coefficients: np.ndarray) -> pd.DataFrame:
    rows = []
    mean_coefficients = coefficients.mean(axis=0)
    mean_abs_coefficients = np.abs(coefficients).mean(axis=0)
    for name, coefficient, abs_coefficient in zip(feature_names, mean_coefficients, mean_abs_coefficients, strict=True):
        rows.append(
            {
                "feature_name": name,
                "mean_coefficient": float(coefficient),
                "mean_abs_coefficient": float(abs_coefficient),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_coefficient", ascending=False).reset_index(drop=True)


def _summarize_univariate_choices(choices: pd.DataFrame) -> pd.DataFrame:
    if choices.empty:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "feature_direction",
                "selection_count",
                "mean_selected_weight",
            ]
        )
    group_columns = ["selected_feature_name", "selected_feature_direction"]
    if "selected_fusion_mode" in choices.columns:
        group_columns.append("selected_fusion_mode")
    if "selected_score_transform_scale" in choices.columns:
        group_columns.append("selected_score_transform_scale")
    grouped = (
        choices.groupby(group_columns, dropna=False)
        .agg(
            selection_count=("selected_feature_name", "size"),
            mean_selected_weight=("selected_weight", "mean"),
            mean_inner_selection_score=("inner_selection_score", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "selected_feature_name": "feature_name",
                "selected_feature_direction": "feature_direction",
                "selected_fusion_mode": "fusion_mode",
                "selected_score_transform_scale": "score_transform_scale",
            }
        )
    )
    return grouped.sort_values(["selection_count", "mean_inner_selection_score"], ascending=[False, False]).reset_index(drop=True)
