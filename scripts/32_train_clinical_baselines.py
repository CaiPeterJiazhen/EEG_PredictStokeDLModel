from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.metrics import binary_classification_metrics


FORBIDDEN_POST_TREATMENT_COLUMNS = {
    "FMA_post",
    "MBI_post",
    "Delta_FMA_obs",
    "Residual",
    "label",
}

NUMERIC_COLUMNS = {"age", "duration", "FMA_pre", "MBI_pre"}
CATEGORICAL_COLUMNS = {"sex", "affected_hand"}


@dataclass(frozen=True)
class ClinicalModelSpec:
    name: str
    features: tuple[str, ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect locked EEG reference baselines. Clinical and qEEG-only "
            "baseline rows are intentionally excluded from current reports."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    predictions, metrics = collect_reference_model_rows(output_root)

    metrics_dir = output_root / "results" / "metrics"
    prediction_dir = output_root / "results" / "predictions"
    docs_dir = output_root / "docs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_dir / "eeg_reference_model_comparison.csv", index=False)
    predictions.to_csv(prediction_dir / "eeg_reference_model_predictions.csv", index=False)
    write_eeg_reference_baseline_doc(metrics, docs_dir / "eeg_reference_baseline_results.md")
    print(f"Wrote {metrics_dir / 'eeg_reference_model_comparison.csv'}")
    print(f"Wrote {prediction_dir / 'eeg_reference_model_predictions.csv'}")


def clinical_model_specs() -> tuple[ClinicalModelSpec, ...]:
    return (
        ClinicalModelSpec("FMA_pre_only_logistic", ("FMA_pre",)),
        ClinicalModelSpec("MBI_pre_only_logistic", ("MBI_pre",)),
        ClinicalModelSpec("age_sex_duration_logistic", ("age", "sex", "duration")),
        ClinicalModelSpec("baseline_clinical_only_logistic", ("age", "sex", "duration", "affected_hand", "FMA_pre", "MBI_pre")),
    )


def fit_clinical_preprocessor(train_frame: pd.DataFrame, features: Sequence[str]) -> ColumnTransformer:
    _validate_features(features)
    numeric = [column for column in features if column in NUMERIC_COLUMNS]
    categorical = [column for column in features if column in CATEGORICAL_COLUMNS]
    transformers = []
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    preprocessor.fit(train_frame.loc[:, list(features)])
    return preprocessor


def run_clinical_baselines_loso(
    label_table: pd.DataFrame,
    *,
    specs: Sequence[ClinicalModelSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = label_table.copy()
    table["subject_id"] = table["subject_id"].map(normalize_subject_id)
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for spec in specs:
        _validate_features(spec.features)
        spec_rows = []
        for fold_index, test_subject in enumerate(table["subject_id"].tolist()):
            train = table[table["subject_id"] != test_subject].copy()
            test = table[table["subject_id"] == test_subject].copy()
            preprocessor = fit_clinical_preprocessor(train, spec.features)
            x_train = preprocessor.transform(train.loc[:, list(spec.features)])
            x_test = preprocessor.transform(test.loc[:, list(spec.features)])
            y_train = train["label"].to_numpy(int)
            classifier = LogisticRegression(
                solver="liblinear",
                C=1.0,
                class_weight="balanced",
                random_state=0,
                max_iter=1000,
            )
            classifier.fit(x_train, y_train)
            y_score = float(classifier.predict_proba(x_test)[0, 1])
            y_true = int(test["label"].iloc[0])
            spec_rows.append(
                {
                    "model": spec.name,
                    "fold_index": int(fold_index),
                    "subject_id": test_subject,
                    "y_true": y_true,
                    "y_score": y_score,
                    "y_pred": int(y_score >= 0.5),
                    "features": ";".join(spec.features),
                    "status": "trained",
                    "skip_reason": "",
                }
            )
        rows.extend(spec_rows)
        spec_frame = pd.DataFrame(spec_rows)
        metric_rows.append(
            {
                "model": spec.name,
                "model_family": "clinical_loso",
                "features": ";".join(spec.features),
                "status": "trained",
                "skip_reason": "",
                **_metrics_with_ci(spec_frame["y_true"].to_numpy(int), spec_frame["y_score"].to_numpy(float)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(metric_rows)


def collect_reference_model_rows(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    metric_rows = []
    references = [
        (
            "no_SSL_CNN_updated_sub05_sub28_seedensemble10",
            output_root / "results" / "predictions" / "dl_loso_predictions_updated_sub05_sub28_10seed_no_ssl_psdfcwpli_gated_schemeA_seedensemble10.csv",
            "EEG_CNN_reference",
        ),
        ("residual_aware_SSL_CNN_seedmean10", output_root / "results" / "predictions" / "seedmean_patient_barlow_residualaware_highrank_swa_clsalpha1_10seed.csv", "EEG_CNN_reference"),
    ]
    for model_name, path, family in references:
        if not path.exists():
            continue
        frame = _normalize_prediction_frame(pd.read_csv(path), model_name)
        prediction_frames.append(frame.assign(model=model_name, model_family=family, status="reference", skip_reason=""))
        metric_rows.append(
            {
                "model": model_name,
                "model_family": family,
                "features": "see_source_predictions",
                "status": "reference",
                "skip_reason": "",
                **_metrics_with_ci(frame["y_true"].to_numpy(int), frame["y_score"].to_numpy(float)),
            }
        )
    ml_reference_files = [
        (
            "updated_no_selector",
            output_root / "results" / "predictions" / "ml_baseline_loso_predictions_updated_sub05_sub28_psdfcwpli_no_selector.csv",
            "PSD/WPLI updated_sub05_sub28 ML baseline; no feature selector",
        ),
        (
            "updated_selectk100",
            output_root / "results" / "predictions" / "ml_baseline_loso_predictions_updated_sub05_sub28_psdfcwpli_selectk100.csv",
            "PSD/WPLI updated_sub05_sub28 ML baseline; SelectK=100 inside LOSO folds",
        ),
    ]
    for tag, path, feature_description in ml_reference_files:
        if not path.exists():
            continue
        ml = pd.read_csv(path)
        trained = ml[ml.get("status", "trained") == "trained"].copy()
        for model_name in sorted(trained["model"].unique()):
            subset = trained[trained["model"] == model_name].copy()
            if subset.empty:
                continue
            output_model_name = f"ML_EEG_{tag}_{model_name}"
            frame = _normalize_prediction_frame(subset, output_model_name)
            prediction_frames.append(
                frame.assign(
                    model=output_model_name,
                    model_family="ML_EEG_updated_sub05_sub28_reference",
                    status="reference",
                    skip_reason="",
                )
            )
            metric_rows.append(
                {
                    "model": output_model_name,
                    "model_family": "ML_EEG_updated_sub05_sub28_reference",
                    "features": feature_description,
                    "status": "reference",
                    "skip_reason": "",
                    **_metrics_with_ci(frame["y_true"].to_numpy(int), frame["y_score"].to_numpy(float)),
                }
            )
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return predictions, pd.DataFrame(metric_rows)


def write_eeg_reference_baseline_doc(metrics: pd.DataFrame, path: Path) -> None:
    display = metrics[
        [
            "model",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "roc_auc",
            "pr_auc",
            "brier_score",
        ]
    ].copy()
    lines = [
        "# EEG Reference Baseline Results",
        "",
        "This table keeps only the current EEG-reference rows used for manuscript comparison. Deprecated clinical logistic and qEEG-only logistic baselines are intentionally excluded from the committed report.",
        "",
        display.to_markdown(index=False),
        "",
        "Reference rows include updated_sub05_sub28 PSD/WPLI ML baselines, updated_sub05_sub28 no-SSL CNN, and the final residual-aware SSL-CNN where corresponding locked predictions already existed. These rows do not change model selection.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_features(features: Sequence[str]) -> None:
    forbidden = FORBIDDEN_POST_TREATMENT_COLUMNS.intersection(features)
    if forbidden:
        raise ValueError(f"Post-treatment/outcome columns are not allowed as clinical predictors: {sorted(forbidden)}")
    allowed = NUMERIC_COLUMNS | CATEGORICAL_COLUMNS
    unknown = set(features) - allowed
    if unknown:
        raise ValueError(f"Unknown clinical feature(s): {sorted(unknown)}")


def _normalize_prediction_frame(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    columns = {"subject_id", "y_true", "y_score"}
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{model_name} prediction frame is missing {sorted(missing)}")
    normalized = frame.copy()
    normalized["subject_id"] = normalized["subject_id"].map(normalize_subject_id)
    if "y_pred" not in normalized.columns:
        normalized["y_pred"] = (normalized["y_score"].astype(float) >= 0.5).astype(int)
    if "fold_index" not in normalized.columns:
        normalized["fold_index"] = np.arange(len(normalized))
    return normalized[["model", "fold_index", "subject_id", "y_true", "y_score", "y_pred"]] if "model" in normalized.columns else normalized.assign(model=model_name)[["model", "fold_index", "subject_id", "y_true", "y_score", "y_pred"]]


def _metrics_with_ci(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = binary_classification_metrics(y_true, y_score)
    cis = _bootstrap_metric_ci(y_true, y_score)
    return {**metrics, **cis}


def _bootstrap_metric_ci(y_true: np.ndarray, y_score: np.ndarray, *, n_bootstrap: int = 1000, seed: int = 17) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    metric_names = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc", "brier_score"]
    samples = {name: [] for name in metric_names}
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y_true), size=len(y_true))
        sample_metrics = binary_classification_metrics(y_true[index], y_score[index])
        for name in metric_names:
            value = sample_metrics.get(name, np.nan)
            if np.isfinite(value):
                samples[name].append(float(value))
    output: dict[str, float] = {}
    for name, values in samples.items():
        if values:
            output[f"{name}_ci_low"] = float(np.quantile(values, 0.025))
            output[f"{name}_ci_high"] = float(np.quantile(values, 0.975))
        else:
            output[f"{name}_ci_low"] = np.nan
            output[f"{name}_ci_high"] = np.nan
    return output


if __name__ == "__main__":
    main()
