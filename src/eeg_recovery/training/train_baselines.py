from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.loso import LOSOFold, fit_transformer_on_train, make_loso_folds
from eeg_recovery.training.metrics import binary_classification_metrics


FORBIDDEN_LABEL_COLUMNS = {
    "FMA_post",
    "MBI_post",
    "Delta_FMA_obs",
    "Delta_FMA_pred",
    "Residual",
}


@dataclass(frozen=True)
class BaselinePreprocessorConfig:
    selector_k: int | None = 100
    pca_components: int | None = None
    enable_selector: bool = True
    enable_pca: bool = False


@dataclass
class FoldPreprocessor:
    scaler: StandardScaler
    selector: SelectKBest | None = None
    pca: PCA | None = None

    def transform(self, X: Any) -> np.ndarray:
        transformed = self.scaler.transform(np.asarray(X, dtype=float))
        if self.selector is not None:
            transformed = self.selector.transform(transformed)
        if self.pca is not None:
            transformed = self.pca.transform(transformed)
        return transformed


@dataclass(frozen=True)
class BaselineModelRegistry:
    models: dict[str, BaseEstimator]
    skipped: dict[str, str]


def fit_fold_preprocessor(
    X: Any,
    y: Any,
    subject_ids: Iterable[object],
    fold: LOSOFold,
    config: BaselinePreprocessorConfig | None = None,
) -> FoldPreprocessor:
    """Fit scaler, optional selector, and optional PCA using only fold training subjects."""

    resolved = config or BaselinePreprocessorConfig()
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=int)
    normalized_subjects = [normalize_subject_id(subject_id) for subject_id in subject_ids]
    train_subjects = {normalize_subject_id(subject_id) for subject_id in fold.train_subject_ids}
    train_mask = np.array([subject_id in train_subjects for subject_id in normalized_subjects])
    if not train_mask.any():
        raise ValueError("No training samples found for this LOSO fold.")

    scaler = fit_transformer_on_train(StandardScaler(), X_array, normalized_subjects, fold)
    scaled = scaler.transform(X_array)
    scaled_train = scaled[train_mask]
    y_train = y_array[train_mask]

    selector = None
    selected = scaled
    if resolved.enable_selector and resolved.selector_k is not None:
        k = min(int(resolved.selector_k), scaled_train.shape[1])
        if k > 0 and k < scaled_train.shape[1]:
            selector = SelectKBest(score_func=f_classif, k=k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                selector.fit(scaled_train, y_train)
            selected = selector.transform(scaled)
        elif k == scaled_train.shape[1]:
            selector = SelectKBest(score_func=f_classif, k="all")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                selector.fit(scaled_train, y_train)
            selected = selector.transform(scaled)

    pca = None
    if resolved.enable_pca and resolved.pca_components is not None:
        selected_train = selected[train_mask]
        n_components = min(int(resolved.pca_components), selected_train.shape[0], selected_train.shape[1])
        if n_components > 0:
            pca = PCA(n_components=n_components, random_state=0)
            pca.fit(selected_train)

    return FoldPreprocessor(scaler=scaler, selector=selector, pca=pca)


def available_baseline_models(random_state: int | None = None) -> BaselineModelRegistry:
    """Return required baseline estimators plus optional boosted tree models when installed."""

    models: dict[str, BaseEstimator] = {
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
        "svm_linear": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=random_state),
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
        ),
        "gaussian_nb": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=3),
    }
    skipped: dict[str, str] = {}

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=random_state,
        )
    except ImportError:
        skipped["xgboost"] = "xgboost is not installed"

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            random_state=random_state,
            verbose=-1,
        )
    except ImportError:
        skipped["lightgbm"] = "lightgbm is not installed"

    return BaselineModelRegistry(models=models, skipped=skipped)


def run_loso_baselines(
    feature_table: pd.DataFrame,
    label_table: pd.DataFrame,
    model_names: Iterable[str] | None = None,
    preprocessor_config: BaselinePreprocessorConfig | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run patient-level LOSO baselines from one row per subject feature and label tables."""

    merged = _merge_features_and_labels(feature_table, label_table)
    subject_ids = merged["subject_id"].tolist()
    y = merged["label"].to_numpy(dtype=int)
    feature_columns = _feature_columns(merged)
    X = merged.loc[:, feature_columns].to_numpy(dtype=float)

    registry = available_baseline_models(random_state=random_state)
    names = (
        list(model_names)
        if model_names is not None
        else [*registry.models.keys(), *registry.skipped.keys()]
    )
    known_names = set(registry.models) | set(registry.skipped)
    unknown = [name for name in names if name not in known_names]
    if unknown:
        raise ValueError(f"Unknown baseline model(s): {', '.join(unknown)}")

    rows: list[dict[str, Any]] = []
    folds = make_loso_folds(subject_ids)
    subject_to_index = {subject_id: index for index, subject_id in enumerate(subject_ids)}
    for model_name in names:
        if model_name in registry.skipped:
            continue
        for fold in folds:
            test_index = subject_to_index[fold.test_subject_id]
            train_indices = [subject_to_index[subject_id] for subject_id in fold.train_subject_ids]
            preprocessor = fit_fold_preprocessor(
                X,
                y,
                subject_ids,
                fold,
                preprocessor_config,
            )
            X_transformed = preprocessor.transform(X)
            model = clone(registry.models[model_name])
            model.fit(X_transformed[train_indices], y[train_indices])
            score = _positive_class_score(model, X_transformed[[test_index]])
            y_score = float(score[0])
            rows.append(
                {
                    "model": model_name,
                    "fold_index": fold.fold_index,
                    "subject_id": fold.test_subject_id,
                    "y_true": int(y[test_index]),
                    "y_score": y_score,
                    "y_pred": int(y_score >= 0.5),
                    "status": "trained",
                    "skip_reason": "",
                }
            )

    prediction_columns = [
        "model",
        "fold_index",
        "subject_id",
        "y_true",
        "y_score",
        "y_pred",
        "status",
        "skip_reason",
    ]
    predictions = pd.DataFrame(rows, columns=prediction_columns)
    metric_rows = []
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "pr_auc",
    )
    for model_name in names:
        if model_name in registry.skipped:
            metric_rows.append(
                {
                    "model": model_name,
                    "status": "skipped",
                    "skip_reason": registry.skipped[model_name],
                    **{metric_name: np.nan for metric_name in metric_names},
                }
            )
            continue
        group = predictions.loc[predictions["model"] == model_name]
        values = binary_classification_metrics(
            group["y_true"].to_numpy(dtype=int),
            group["y_score"].to_numpy(dtype=float),
        )
        metric_rows.append({"model": model_name, "status": "trained", "skip_reason": "", **values})
    metrics = pd.DataFrame(metric_rows)
    return predictions, metrics


def write_baseline_outputs(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_root: str | Path,
) -> tuple[Path, Path]:
    """Write Task 9 formal prediction and metric CSV outputs below output_root."""

    root = Path(output_root)
    prediction_path = root / "results" / "predictions" / "ml_baseline_loso_predictions.csv"
    metric_path = root / "results" / "metrics" / "ml_baseline_model_comparison.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(prediction_path, index=False)
    metrics_df.to_csv(metric_path, index=False)
    return prediction_path, metric_path


def _merge_features_and_labels(feature_table: pd.DataFrame, label_table: pd.DataFrame) -> pd.DataFrame:
    if "subject_id" not in feature_table.columns:
        raise ValueError("feature_table must contain subject_id.")
    if not {"subject_id", "label"}.issubset(label_table.columns):
        raise ValueError("label_table must contain subject_id and label.")

    features = feature_table.copy()
    labels = label_table.loc[:, ["subject_id", "label"]].copy()
    features["subject_id"] = features["subject_id"].map(normalize_subject_id)
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    return features.merge(labels, on="subject_id", how="inner", validate="one_to_one")


def _feature_columns(table: pd.DataFrame) -> list[str]:
    excluded = {"subject_id", "label", *FORBIDDEN_LABEL_COLUMNS}
    columns = [column for column in table.columns if column not in excluded]
    if not columns:
        raise ValueError("No model input feature columns are available.")
    return columns


def _positive_class_score(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    return np.asarray(model.predict(X), dtype=float)
