from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.evaluation.statistical_validation import paired_bootstrap_difference, subject_level_predictions
from eeg_recovery.features.feature_tables import (
    load_fc_feature_table,
    load_psd_band_power_table,
    merge_feature_tables,
)
from eeg_recovery.metadata.labels import MODEL_INPUT_COLUMNS, load_supervised_label_table, model_input_metadata
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.metrics import binary_classification_metrics


CLINICAL_INPUT_COLUMNS = tuple(MODEL_INPUT_COLUMNS)
FORBIDDEN_CLINICAL_INPUT_COLUMNS = {
    "FMA_post",
    "MBI_post",
    "Delta_FMA_obs",
    "Delta_FMA_pred",
    "Residual",
    "label",
    "y_true",
    "y_score",
    "y_pred",
}
CATEGORICAL_COLUMNS = {"sex", "affected_hand"}
TABULAR_MODEL_NAMES = ("logistic_l1", "logistic_l2", "svm_rbf", "random_forest", "gaussian_nb", "knn")


@dataclass(frozen=True)
class TabularModelSpec:
    name: str
    estimator: BaseEstimator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train clinical-only and EEG+clinical patient-level LOSO baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=list(TABULAR_MODEL_NAMES), help="Clinical-only models.")
    parser.add_argument(
        "--incremental-models",
        nargs="+",
        default=["logistic_l1", "logistic_l2", "svm_rbf"],
        help="EEG+clinical models. Keep this smaller for interactive runs; all transforms remain fold-local.",
    )
    parser.add_argument(
        "--selector-options",
        nargs="+",
        choices=("none", "selectk100"),
        default=["selectk100"],
        help=(
            "Feature-selection settings for EEG+clinical models. Clinical-only always uses none. "
            "EEG-only no-selector and SelectK references are read from locked existing baseline files."
        ),
    )
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    labels = load_supervised_label_table(path_config)
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    clinical = model_input_metadata(labels)

    clinical_predictions, clinical_metrics = run_loso_tabular_models(
        clinical,
        label_table=labels[["subject_id", "label"]],
        input_columns=CLINICAL_INPUT_COLUMNS,
        model_names=args.models,
        feature_selection="none",
        random_state=args.random_state,
        model_family="clinical_only",
    )

    feature_table = build_psd_wpli_feature_table(output_root, labels["subject_id"].tolist())
    eeg_clinical = feature_table.merge(clinical, on="subject_id", how="inner", validate="one_to_one")
    eeg_clinical_columns = [column for column in eeg_clinical.columns if column != "subject_id"]

    incremental_prediction_frames = [load_existing_eeg_only_predictions(output_root)]
    incremental_metric_frames = [load_existing_eeg_only_metrics(output_root)]
    for option in args.selector_options:
        predictions, metrics = run_loso_tabular_models(
            eeg_clinical,
            label_table=labels[["subject_id", "label"]],
            input_columns=eeg_clinical_columns,
            model_names=args.incremental_models,
            feature_selection=option,
            random_state=args.random_state,
            model_family="eeg_clinical",
        )
        incremental_prediction_frames.append(predictions)
        incremental_metric_frames.append(metrics)

    incremental_predictions = pd.concat(
        [frame for frame in incremental_prediction_frames if frame is not None and not frame.empty],
        ignore_index=True,
    )
    incremental_metrics = pd.concat(
        [frame for frame in incremental_metric_frames if frame is not None and not frame.empty],
        ignore_index=True,
    )

    metrics_dir = output_root / "results" / "metrics"
    predictions_dir = output_root / "results" / "predictions"
    statistics_dir = output_root / "results" / "statistics"
    docs_dir = output_root / "docs"
    for directory in (metrics_dir, predictions_dir, statistics_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    clinical_metrics.to_csv(metrics_dir / "clinical_only_model_comparison.csv", index=False)
    clinical_predictions.to_csv(predictions_dir / "clinical_only_loso_predictions.csv", index=False)
    incremental_metrics.to_csv(metrics_dir / "eeg_clinical_incremental_model_comparison.csv", index=False)
    incremental_predictions.to_csv(predictions_dir / "eeg_clinical_incremental_loso_predictions.csv", index=False)

    comparisons = clinical_incremental_comparisons(clinical_predictions, incremental_predictions, n_bootstrap=500)
    comparisons.to_csv(statistics_dir / "clinical_incremental_paired_bootstrap_comparison.csv", index=False)
    write_clinical_incremental_doc(
        docs_dir / "clinical_incremental_value_results.md",
        clinical_metrics,
        incremental_metrics,
        comparisons,
    )
    print(f"Wrote {metrics_dir / 'clinical_only_model_comparison.csv'}")
    print(f"Wrote {metrics_dir / 'eeg_clinical_incremental_model_comparison.csv'}")
    print(f"Wrote {statistics_dir / 'clinical_incremental_paired_bootstrap_comparison.csv'}")


def validate_clinical_predictors(columns: Sequence[str]) -> None:
    forbidden = FORBIDDEN_CLINICAL_INPUT_COLUMNS.intersection(columns)
    if forbidden:
        raise ValueError(f"Post-treatment, outcome, or label-derived variables are not allowed as predictors: {sorted(forbidden)}")
    unknown = set(columns).intersection({"subject_id"})
    if unknown:
        raise ValueError("subject_id is an identifier and must not be used as a model predictor.")


def fit_fold_mixed_preprocessor(
    train_frame: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    transformers = []
    numeric = list(numeric_columns)
    categorical = list(categorical_columns)
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("At least one numeric or categorical predictor is required.")
    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    preprocessor.fit(train_frame.loc[:, [*numeric, *categorical]])
    return preprocessor


def run_loso_tabular_models(
    feature_frame: pd.DataFrame,
    *,
    label_table: pd.DataFrame,
    input_columns: Sequence[str],
    model_names: Sequence[str],
    feature_selection: str,
    random_state: int,
    model_family: str = "clinical_only",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_clinical_predictors([column for column in input_columns if column in CLINICAL_INPUT_COLUMNS or column in FORBIDDEN_CLINICAL_INPUT_COLUMNS])
    if feature_selection not in {"none", "selectk100"}:
        raise ValueError("feature_selection must be 'none' or 'selectk100'.")
    if "subject_id" not in feature_frame.columns:
        raise ValueError("feature_frame must contain subject_id.")
    if not {"subject_id", "label"}.issubset(label_table.columns):
        raise ValueError("label_table must contain subject_id and label.")

    features = feature_frame.copy()
    if "label" in features.columns:
        features = features.drop(columns=["label"])
    labels = label_table.loc[:, ["subject_id", "label"]].copy()
    features["subject_id"] = features["subject_id"].map(normalize_subject_id)
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    merged = features.merge(labels, on="subject_id", how="inner", validate="one_to_one")
    merged = merged.sort_values("subject_id").reset_index(drop=True)
    if merged.empty:
        raise ValueError("No matched subject rows for LOSO training.")

    selected_columns = list(input_columns)
    missing = [column for column in selected_columns if column not in merged.columns]
    if missing:
        raise ValueError(f"Missing input column(s): {missing}")
    categorical = [column for column in selected_columns if column in CATEGORICAL_COLUMNS]
    numeric = [column for column in selected_columns if column not in categorical]
    registry = tabular_model_registry(random_state)
    unknown = [name for name in model_names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown model name(s): {unknown}")

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for raw_model_name in model_names:
        model = registry[raw_model_name]
        model_name = f"{model_family}_{raw_model_name}"
        if model_family == "eeg_clinical":
            model_name = f"{model_name}_{feature_selection}"
        rows_for_model: list[dict[str, object]] = []
        for fold_index, subject_id in enumerate(merged["subject_id"].tolist()):
            train = merged[merged["subject_id"] != subject_id].copy()
            test = merged[merged["subject_id"] == subject_id].copy()
            preprocessor = fit_fold_mixed_preprocessor(
                train,
                numeric_columns=numeric,
                categorical_columns=categorical,
            )
            x_train = preprocessor.transform(train.loc[:, selected_columns])
            x_test = preprocessor.transform(test.loc[:, selected_columns])
            selector = _fit_selector(x_train, train["label"].to_numpy(int), feature_selection)
            if selector is not None:
                x_train = selector.transform(x_train)
                x_test = selector.transform(x_test)
            estimator = clone(model)
            estimator.fit(x_train, train["label"].to_numpy(int))
            y_score = float(_positive_class_score(estimator, x_test)[0])
            y_true = int(test["label"].iloc[0])
            rows_for_model.append(
                {
                    "model": model_name,
                    "model_family": model_family,
                    "feature_selection": feature_selection,
                    "fold_index": int(fold_index),
                    "subject_id": subject_id,
                    "y_true": y_true,
                    "y_score": y_score,
                    "y_pred": int(y_score >= 0.5),
                    "status": "trained",
                    "skip_reason": "",
                }
            )
        prediction_rows.extend(rows_for_model)
        prediction_frame = pd.DataFrame(rows_for_model)
        metrics = binary_classification_metrics(
            prediction_frame["y_true"].to_numpy(int),
            prediction_frame["y_score"].to_numpy(float),
        )
        metric_rows.append(
            {
                "model": model_name,
                "model_family": model_family,
                "input_features": "baseline clinical only" if model_family == "clinical_only" else "PSD+WPLI EO+EC + baseline clinical",
                "feature_selection": feature_selection,
                "n_subjects": int(prediction_frame["subject_id"].nunique()),
                "n_positive": int(prediction_frame["y_true"].sum()),
                "n_negative": int(len(prediction_frame) - prediction_frame["y_true"].sum()),
                "status": "trained",
                "skip_reason": "",
                **metrics,
            }
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def build_psd_wpli_feature_table(output_root: Path, subject_ids: Sequence[str]) -> pd.DataFrame:
    psd = load_psd_band_power_table(output_root / "data" / "features" / "psd", subject_ids=subject_ids)
    wpli = load_fc_feature_table(output_root / "data" / "features" / "fc", metric="wpli", subject_ids=subject_ids)
    return merge_feature_tables(psd, wpli)


def tabular_model_registry(random_state: int) -> dict[str, BaseEstimator]:
    return {
        "logistic_l1": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
        "logistic_l2": LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state),
        "gaussian_nb": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=3),
    }


def load_existing_eeg_only_predictions(output_root: Path) -> pd.DataFrame:
    frames = []
    sources = [
        ("none", output_root / "results" / "predictions" / "ml_baseline_loso_predictions_updated_sub05_sub28_psdfcwpli_no_selector.csv"),
        ("selectk100", output_root / "results" / "predictions" / "ml_baseline_loso_predictions_updated_sub05_sub28_psdfcwpli_selectk100.csv"),
    ]
    for feature_selection, path in sources:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame.get("status", "trained") == "trained"].copy()
        frame["subject_id"] = frame["subject_id"].map(normalize_subject_id)
        frame["model"] = "eeg_only_" + frame["model"].astype(str) + f"_{feature_selection}"
        frame["model_family"] = "eeg_only"
        frame["feature_selection"] = feature_selection
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_existing_eeg_only_metrics(output_root: Path) -> pd.DataFrame:
    predictions = load_existing_eeg_only_predictions(output_root)
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for model_name, group in predictions.groupby("model", sort=True):
        metrics = binary_classification_metrics(group["y_true"].to_numpy(int), group["y_score"].to_numpy(float))
        rows.append(
            {
                "model": model_name,
                "model_family": "eeg_only",
                "input_features": "PSD+WPLI EO+EC",
                "feature_selection": str(group["feature_selection"].iloc[0]),
                "n_subjects": int(group["subject_id"].nunique()),
                "n_positive": int(group["y_true"].sum()),
                "n_negative": int(len(group) - group["y_true"].sum()),
                "status": "reference",
                "skip_reason": "",
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def clinical_incremental_comparisons(
    clinical_predictions: pd.DataFrame,
    incremental_predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    if clinical_predictions.empty or incremental_predictions.empty:
        return pd.DataFrame()
    clinical_best = _best_model_by_auc_then_brier(clinical_predictions)
    rows = []
    clinical_subject = subject_level_predictions(clinical_predictions[clinical_predictions["model"] == clinical_best])
    metrics = ("accuracy", "roc_auc", "brier_score")
    for model_name, group in _key_incremental_candidates(incremental_predictions).groupby("model", sort=True):
        candidate = subject_level_predictions(group)
        merged = clinical_subject.merge(candidate, on="subject_id", suffixes=("_clinical", "_candidate"), validate="one_to_one")
        if merged.empty:
            continue
        for metric in metrics:
            diff = paired_bootstrap_difference(
                merged["y_true_clinical"].to_numpy(int),
                merged["y_score_clinical"].to_numpy(float),
                merged["y_score_candidate"].to_numpy(float),
                metric=metric,
                n_bootstrap=n_bootstrap,
                random_state=97,
            )
            rows.append(
                {
                    "reference_model": clinical_best,
                    "candidate_model": model_name,
                    "n_subjects": int(merged["subject_id"].nunique()),
                    "n_positive": int(merged["y_true_clinical"].sum()),
                    "n_negative": int(len(merged) - merged["y_true_clinical"].sum()),
                    "metric": metric,
                    **diff,
                }
            )
    return pd.DataFrame(rows)


def _key_incremental_candidates(predictions: pd.DataFrame) -> pd.DataFrame:
    key_models = {
        "eeg_only_logistic_l1_none",
        "eeg_only_logistic_l2_none",
        "eeg_only_svm_rbf_selectk100",
        "eeg_clinical_logistic_l1_selectk100",
        "eeg_clinical_logistic_l2_selectk100",
        "eeg_clinical_svm_rbf_selectk100",
    }
    subset = predictions[predictions["model"].isin(key_models)].copy()
    if subset.empty:
        return predictions
    return subset


def write_clinical_incremental_doc(
    path: Path,
    clinical_metrics: pd.DataFrame,
    incremental_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clinical_display = _display_metrics(clinical_metrics)
    incremental_display = _display_metrics(incremental_metrics)
    comparison_display = comparisons.head(30) if not comparisons.empty else pd.DataFrame()
    lines = [
        "# Clinical Baseline And EEG Incremental Value",
        "",
        "All clinical predictors are baseline-only: age, sex, duration, affected_hand, FMA_pre, and MBI_pre.",
        "Post-treatment variables, observed/predicted deltas, residuals, and labels are blocked from model inputs.",
        "",
        "## Clinical-only LOSO Models",
        "",
        _to_markdown(clinical_display),
        "",
        "## EEG-only And EEG+clinical Models",
        "",
        _to_markdown(incremental_display),
        "",
        "## Paired Bootstrap Versus Best Clinical-only Model",
        "",
        _to_markdown(comparison_display),
        "",
        "The paired bootstrap uses subject-level LOSO predictions and resamples subjects, not seeds or segments.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fit_selector(x_train: np.ndarray, y_train: np.ndarray, feature_selection: str) -> SelectKBest | None:
    if feature_selection == "none":
        return None
    k = min(100, x_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        selector.fit(x_train, y_train)
    return selector


def _positive_class_score(model: BaseEstimator, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return probabilities[:, classes.index(1)]
        return probabilities[:, -1]
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x_test), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    return np.asarray(model.predict(x_test), dtype=float)


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _best_model_by_auc_then_brier(predictions: pd.DataFrame) -> str:
    rows = []
    for model_name, group in predictions.groupby("model", sort=True):
        metrics = binary_classification_metrics(group["y_true"].to_numpy(int), group["y_score"].to_numpy(float))
        rows.append({"model": model_name, **metrics})
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["roc_auc", "brier_score", "accuracy"], ascending=[False, True, False])
    return str(frame.iloc[0]["model"])


def _display_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame({"status": ["not available"]})
    columns = [
        "model",
        "model_family",
        "feature_selection",
        "n_subjects",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]
    available = [column for column in columns if column in metrics.columns]
    return metrics.loc[:, available].sort_values([column for column in ["model_family", "roc_auc", "brier_score"] if column in available], ascending=[True, False, True][: len([column for column in ["model_family", "roc_auc", "brier_score"] if column in available])])


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
