from __future__ import annotations

from itertools import combinations
from typing import Callable, Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression

from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.metrics import binary_classification_metrics


METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "roc_auc",
    "pr_auc",
    "brier_score",
)


def subject_level_predictions(
    frame: pd.DataFrame,
    *,
    subject_column: str = "subject_id",
    score_column: str = "y_score",
    label_column: str = "y_true",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return exactly one prediction row per subject.

    Seed-level or repeated rows are averaged by subject. Labels must be
    internally consistent for each subject; otherwise the input is invalid for
    patient-level LOSO evaluation.
    """

    required = {subject_column, score_column, label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing required column(s): {sorted(missing)}")
    data = frame.copy()
    data[subject_column] = data[subject_column].map(normalize_subject_id)
    rows: list[dict[str, object]] = []
    for subject_id, group in data.groupby(subject_column, sort=True):
        labels = group[label_column].dropna().astype(int).unique()
        if len(labels) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels: {labels.tolist()}")
        score = float(group[score_column].astype(float).mean())
        rows.append(
            {
                "subject_id": subject_id,
                "y_true": int(labels[0]),
                "y_score": score,
                "y_pred": int(score >= threshold),
                "n_source_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=["subject_id", "y_true", "y_score", "y_pred", "n_source_rows"])


def compute_metric(
    y_true: Iterable[int],
    y_score: Iterable[float],
    metric: str,
    *,
    threshold: float = 0.5,
) -> float:
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    values = _fast_metric_dict(y_true_array, y_score_array, threshold=threshold)
    if metric not in values:
        raise ValueError(f"Unsupported metric: {metric}")
    return float(values[metric])


def bootstrap_metric_ci(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    metrics: Iterable[str] = METRIC_NAMES,
    n_bootstrap: int = 5000,
    random_state: int = 17,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Bootstrap percentile confidence intervals over subjects."""

    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    _validate_prediction_arrays(y_true_array, y_score_array)
    rng = np.random.default_rng(random_state)
    names = tuple(metrics)
    samples = {name: [] for name in names}
    for _ in range(int(n_bootstrap)):
        index = rng.integers(0, len(y_true_array), size=len(y_true_array))
        values = _fast_metric_dict(y_true_array[index], y_score_array[index], threshold=threshold)
        for name in names:
            value = float(values.get(name, np.nan))
            if np.isfinite(value):
                samples[name].append(value)
    output: dict[str, float] = {"n_subjects": int(len(y_true_array))}
    for name in names:
        values = samples[name]
        if values:
            output[f"{name}_low"] = float(np.quantile(values, 0.025))
            output[f"{name}_high"] = float(np.quantile(values, 0.975))
        else:
            output[f"{name}_low"] = np.nan
            output[f"{name}_high"] = np.nan
    return output


def exact_binomial_accuracy_pvalue(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    chance: float = 0.5,
    threshold: float = 0.5,
    alternative: str = "greater",
) -> float:
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    _validate_prediction_arrays(y_true_array, y_pred)
    correct = int(np.sum(y_true_array == y_pred))
    return float(binomtest(correct, len(y_true_array), p=chance, alternative=alternative).pvalue)


def label_permutation_test(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    metric: str = "roc_auc",
    n_permutations: int = 5000,
    random_state: int = 23,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Shuffle labels at subject level and compare the requested metric."""

    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    _validate_prediction_arrays(y_true_array, y_score_array)
    observed = compute_metric(y_true_array, y_score_array, metric, threshold=threshold)
    rng = np.random.default_rng(random_state)
    count = 1
    valid = 0
    for _ in range(int(n_permutations)):
        shuffled = rng.permutation(y_true_array)
        value = compute_metric(shuffled, y_score_array, metric, threshold=threshold)
        if not np.isfinite(value):
            continue
        valid += 1
        if value >= observed:
            count += 1
    denominator = valid + 1
    return {
        "metric": metric,
        "observed": float(observed),
        "p_value": float(count / denominator) if denominator else np.nan,
        "n_permutations": int(n_permutations),
        "n_valid_permutations": int(valid),
    }


def calibration_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    n_bins: int = 5,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    _validate_prediction_arrays(y_true_array, y_score_array)
    base = binary_classification_metrics(y_true_array, y_score_array)
    ece = expected_calibration_error(y_true_array, y_score_array, n_bins=n_bins)
    intercept, slope = calibration_intercept_slope(y_true_array, y_score_array)
    return {
        "brier_score": float(base["brier_score"]),
        "ece": float(ece),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
    }


def expected_calibration_error(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    n_bins: int = 5,
) -> float:
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    _validate_prediction_arrays(y_true_array, y_score_array)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        if high == 1.0:
            mask = (y_score_array >= low) & (y_score_array <= high)
        else:
            mask = (y_score_array >= low) & (y_score_array < high)
        if not mask.any():
            continue
        confidence = float(np.mean(y_score_array[mask]))
        observed = float(np.mean(y_true_array[mask]))
        ece += float(mask.mean()) * abs(confidence - observed)
    return float(ece)


def calibration_intercept_slope(
    y_true: Iterable[int],
    y_score: Iterable[float],
) -> tuple[float, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    _validate_prediction_arrays(y_true_array, y_score_array)
    if np.unique(y_true_array).size < 2:
        return np.nan, np.nan
    clipped = np.clip(y_score_array, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
                model.fit(logits, y_true_array)
            except TypeError:
                model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
                model.fit(logits, y_true_array)
            except ValueError:
                model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                model.fit(logits, y_true_array)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def paired_bootstrap_difference(
    y_true: Iterable[int],
    y_score_a: Iterable[float],
    y_score_b: Iterable[float],
    *,
    metric: str,
    n_bootstrap: int = 5000,
    random_state: int = 31,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    score_a = np.asarray(y_score_a, dtype=float)
    score_b = np.asarray(y_score_b, dtype=float)
    _validate_prediction_arrays(y_true_array, score_a)
    _validate_prediction_arrays(y_true_array, score_b)
    observed_a = compute_metric(y_true_array, score_a, metric, threshold=threshold)
    observed_b = compute_metric(y_true_array, score_b, metric, threshold=threshold)
    observed_diff = observed_b - observed_a
    rng = np.random.default_rng(random_state)
    diffs: list[float] = []
    for _ in range(int(n_bootstrap)):
        index = rng.integers(0, len(y_true_array), size=len(y_true_array))
        value_a = compute_metric(y_true_array[index], score_a[index], metric, threshold=threshold)
        value_b = compute_metric(y_true_array[index], score_b[index], metric, threshold=threshold)
        if np.isfinite(value_a) and np.isfinite(value_b):
            diffs.append(float(value_b - value_a))
    return {
        "metric": metric,
        "metric_a": float(observed_a),
        "metric_b": float(observed_b),
        "difference": float(observed_diff),
        "ci_low": float(np.quantile(diffs, 0.025)) if diffs else np.nan,
        "ci_high": float(np.quantile(diffs, 0.975)) if diffs else np.nan,
        "p_value_two_sided": _two_sided_bootstrap_pvalue(diffs) if diffs else np.nan,
        "n_bootstrap": int(n_bootstrap),
    }


def mcnemar_test(
    y_true: Iterable[int],
    y_score_a: Iterable[float],
    y_score_b: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    pred_a = (np.asarray(y_score_a, dtype=float) >= threshold).astype(int)
    pred_b = (np.asarray(y_score_b, dtype=float) >= threshold).astype(int)
    _validate_prediction_arrays(y_true_array, pred_a)
    _validate_prediction_arrays(y_true_array, pred_b)
    correct_a = pred_a == y_true_array
    correct_b = pred_b == y_true_array
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    discordant = b + c
    p_value = float(binomtest(min(b, c), discordant, p=0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "n_discordant": discordant,
        "p_value": p_value,
    }


def align_prediction_frames(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Align two subject-level prediction frames by subject_id."""

    left = subject_level_predictions(first)
    right = subject_level_predictions(second)
    merged = left.merge(right, on="subject_id", suffixes=("_a", "_b"), validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping subjects between prediction frames.")
    if not np.array_equal(merged["y_true_a"].to_numpy(int), merged["y_true_b"].to_numpy(int)):
        raise ValueError("Aligned prediction frames have inconsistent labels.")
    return (
        merged["y_true_a"].to_numpy(int),
        merged["y_score_a"].to_numpy(float),
        merged["y_score_b"].to_numpy(float),
        merged["subject_id"].tolist(),
    )


def pairwise_model_names(names: Iterable[str]) -> list[tuple[str, str]]:
    return list(combinations(tuple(names), 2))


def _validate_prediction_arrays(y_true: np.ndarray, y_score: np.ndarray) -> None:
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be one-dimensional.")
    if y_true.shape[0] == 0:
        raise ValueError("At least one subject-level prediction is required.")


def _fast_metric_dict(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "roc_auc": float(_fast_roc_auc(y_true, y_score)),
        "pr_auc": float(_fast_average_precision(y_true, y_score)),
        "brier_score": float(np.mean((y_score - y_true) ** 2)),
    }


def _fast_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[start:end] = average_rank
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = float(np.sum(original_ranks[y_true == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _fast_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return np.nan
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tp_cumulative = np.cumsum(sorted_true == 1)
    ranks = np.arange(1, len(sorted_true) + 1)
    precision_at_k = tp_cumulative / ranks
    return float(np.sum(precision_at_k[sorted_true == 1]) / n_pos)


def _two_sided_bootstrap_pvalue(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return np.nan
    p_lower = float(np.mean(array <= 0.0))
    p_upper = float(np.mean(array >= 0.0))
    return float(min(1.0, 2.0 * min(p_lower, p_upper)))
