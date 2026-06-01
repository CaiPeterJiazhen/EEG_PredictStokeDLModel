from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.evaluation.statistical_validation import subject_level_predictions
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.metrics import binary_classification_metrics


@dataclass(frozen=True)
class PredictionSource:
    model_name: str
    path: Path
    model_family: str
    input_features: str
    feature_selection: str
    inference_type: str
    result_role: str
    source_model: str | None = None
    legacy_warning: str = ""


MAIN_MODEL_ORDER = (
    "ML_EEG_updated_no_selector_logistic_l1",
    "ML_EEG_updated_no_selector_logistic_l2",
    "ML_EEG_updated_selectk100_svm_rbf",
    "no_SSL_CNN_updated_sub05_sub28_seedensemble10",
    "residual_aware_SSL_CNN_seedmean10",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build locked manuscript model-performance tables from subject-level predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    predictions = collect_locked_predictions(output_root)
    performance = build_performance_table(predictions)
    ensure_normalized_ml_reference_copy(output_root)

    tables_dir = output_root / "results" / "tables"
    predictions_dir = output_root / "results" / "predictions"
    docs_dir = output_root / "docs"
    for directory in (tables_dir, predictions_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_dir / "paper_locked_model_predictions.csv", index=False)
    performance.to_csv(tables_dir / "paper_locked_model_performance.csv", index=False)
    (tables_dir / "paper_locked_model_performance.md").write_text(_to_markdown(performance), encoding="utf-8")
    write_summary_doc(docs_dir / "paper_locked_results_summary.md", performance)
    print(f"Wrote {tables_dir / 'paper_locked_model_performance.csv'}")
    print(f"Wrote {predictions_dir / 'paper_locked_model_predictions.csv'}")


def collect_locked_predictions(output_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in locked_prediction_sources(output_root):
        if not source.path.exists():
            continue
        frame = pd.read_csv(source.path)
        if source.source_model is not None and "model" in frame.columns:
            frame = frame[frame["model"].astype(str) == source.source_model].copy()
        if frame.empty:
            continue
        if source.source_model is None and source.model_name.endswith("_best_available") and "model" in frame.columns:
            for raw_model, group in frame.groupby("model", sort=True):
                derived = PredictionSource(
                    f"{source.model_name}_{raw_model}",
                    source.path,
                    source.model_family,
                    source.input_features,
                    source.feature_selection,
                    source.inference_type,
                    source.result_role,
                    legacy_warning=source.legacy_warning,
                )
                frames.append(_normalize_prediction_frame(group, derived))
        else:
            normalized = _normalize_prediction_frame(frame, source)
            frames.append(normalized)
    if not frames:
        raise FileNotFoundError("No locked prediction files were found under results/predictions.")
    return pd.concat(frames, ignore_index=True, sort=False)


def build_performance_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model_name", sort=False):
        subject = subject_level_predictions(group)
        metrics = binary_classification_metrics(subject["y_true"].to_numpy(int), subject["y_score"].to_numpy(float))
        first = group.iloc[0]
        rows.append(
            {
                "model_family": first["model_family"],
                "model_name": model_name,
                "input_features": first["input_features"],
                "feature_selection": first["feature_selection"],
                "inference_type": first["inference_type"],
                "result_role": first["result_role"],
                "legacy_warning": first["legacy_warning"],
                "n_subjects": int(subject["subject_id"].nunique()),
                "n_positive": int(subject["y_true"].sum()),
                "n_negative": int(len(subject) - subject["y_true"].sum()),
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    frame["sort_key"] = frame["model_name"].map({model: index for index, model in enumerate(MAIN_MODEL_ORDER)}).fillna(1000)
    frame = frame.sort_values(["sort_key", "result_role", "model_name"]).drop(columns=["sort_key"]).reset_index(drop=True)
    return frame


def locked_prediction_sources(output_root: Path) -> list[PredictionSource]:
    prediction_dir = output_root / "results" / "predictions"
    sources: list[PredictionSource] = [
        PredictionSource(
            "no_SSL_CNN_updated_sub05_sub28_seedensemble10",
            prediction_dir / "dl_loso_predictions_updated_sub05_sub28_10seed_no_ssl_psdfcwpli_gated_schemeA_seedensemble10.csv",
            "CNN",
            "PSD+WPLI EO+EC",
            "none",
            "10-seed ensemble",
            "primary_cnn_reference",
        ),
        PredictionSource(
            "residual_aware_SSL_CNN_seedmean10",
            prediction_dir / "seedmean_patient_barlow_residualaware_highrank_swa_clsalpha1_10seed.csv",
            "Residual-aware SSL-CNN",
            "PSD+WPLI EO+EC",
            "none",
            "10-seed mean; classification-head inference",
            "primary_final_model",
        ),
    ]
    for tag, filename, feature_selection in (
        ("updated_no_selector", "ml_baseline_loso_predictions_updated_sub05_sub28_psdfcwpli_no_selector.csv", "none"),
        ("updated_selectk100", "ml_baseline_loso_predictions_updated_sub05_sub28_psdfcwpli_selectk100.csv", "SelectK=100 inside LOSO train folds"),
    ):
        path = prediction_dir / filename
        for model in ("logistic_l1", "logistic_l2", "svm_linear", "svm_rbf", "random_forest", "gaussian_nb", "knn"):
            model_name = f"ML_EEG_{tag}_{model}"
            role = "exploratory_ml_reference"
            if model_name in {
                "ML_EEG_updated_no_selector_logistic_l1",
                "ML_EEG_updated_no_selector_logistic_l2",
                "ML_EEG_updated_selectk100_svm_rbf",
            }:
                role = "primary_ml_baseline"
            sources.append(
                PredictionSource(
                    model_name,
                    path,
                    "Traditional ML",
                    "PSD+WPLI EO+EC",
                    feature_selection,
                    "single LOSO prediction per subject",
                    role,
                    source_model=model,
                )
            )
    optional_sources = [
        PredictionSource(
            "clinical_only_best_available",
            prediction_dir / "clinical_only_loso_predictions.csv",
            "Clinical baseline",
            "baseline clinical variables",
            "none",
            "LOSO",
            "exploratory_clinical",
            legacy_warning="Generated by script 39 if present; not an EEG main result.",
        ),
        PredictionSource(
            "eeg_clinical_best_available",
            prediction_dir / "eeg_clinical_incremental_loso_predictions.csv",
            "EEG+clinical baseline",
            "PSD+WPLI EO+EC + baseline clinical variables",
            "mixed",
            "LOSO",
            "exploratory_incremental",
            legacy_warning="Generated by script 39 if present; used for incremental value only.",
        ),
    ]
    sources.extend(optional_sources)
    return sources


def ensure_normalized_ml_reference_copy(output_root: Path) -> None:
    target = output_root / "results" / "metrics" / "clinical_baseline_model_comparison.csv"
    if target.exists():
        return
    source = output_root / "results" / "metrics" / "eeg_reference_model_comparison.csv"
    if not source.exists():
        return
    frame = pd.read_csv(source)
    if "model_family" in frame.columns:
        frame = frame[frame["model_family"].astype(str) == "ML_EEG_updated_sub05_sub28_reference"].copy()
    if frame.empty:
        return
    frame["normalized_copy_note"] = (
        "Generated because clinical_baseline_model_comparison.csv was missing; "
        "contains current updated PSD+WPLI ML reference rows from eeg_reference_model_comparison.csv."
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)


def write_summary_doc(path: Path, performance: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    main = performance[performance["result_role"].astype(str).str.startswith("primary")]
    exploratory = performance[~performance["result_role"].astype(str).str.startswith("primary")]
    lines = [
        "# Paper Locked Results Summary",
        "",
        "This file freezes the manuscript-facing model table by recomputing metrics from subject-level LOSO prediction CSVs. Segment rows and seed rows are not treated as independent patients.",
        "",
        "## Primary Results",
        "",
        _to_markdown(
            main[
                [
                    "model_name",
                    "input_features",
                    "feature_selection",
                    "n_subjects",
                    "accuracy",
                    "balanced_accuracy",
                    "roc_auc",
                    "pr_auc",
                    "brier_score",
                ]
            ]
        ),
        "",
        "## Exploratory Or Support Rows",
        "",
        _to_markdown(
            exploratory[
                [
                    "model_name",
                    "result_role",
                    "input_features",
                    "feature_selection",
                    "accuracy",
                    "roc_auc",
                    "pr_auc",
                    "brier_score",
                    "legacy_warning",
                ]
            ]
        )
        if not exploratory.empty
        else "_No exploratory rows were available._",
        "",
        "## Locked Interpretation",
        "",
        "The current manuscript baseline is the updated PSD+WPLI ML comparison, not the older 0.8421 traditional-ML number from deprecated notes. The main claim should compare the CNN rows against the current Logistic/SVM PSD+WPLI baselines, then emphasize the residual-aware SSL-CNN gains in ROC-AUC, PR-AUC, Brier/calibration, and seed stability.",
        "",
        "Rows marked exploratory or legacy-support must not be promoted to primary manuscript results without a new locked analysis plan.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _normalize_prediction_frame(frame: pd.DataFrame, source: PredictionSource) -> pd.DataFrame:
    required = {"subject_id", "y_true", "y_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source.model_name} prediction file is missing {sorted(missing)}")
    normalized = frame.copy()
    normalized["subject_id"] = normalized["subject_id"].map(normalize_subject_id)
    if "fold_index" not in normalized.columns:
        normalized["fold_index"] = np.arange(len(normalized))
    if "y_pred" not in normalized.columns:
        normalized["y_pred"] = (normalized["y_score"].astype(float) >= 0.5).astype(int)
    normalized["model_name"] = source.model_name
    normalized["model_family"] = source.model_family
    normalized["input_features"] = source.input_features
    normalized["feature_selection"] = source.feature_selection
    normalized["inference_type"] = source.inference_type
    normalized["result_role"] = source.result_role
    normalized["legacy_warning"] = source.legacy_warning
    normalized["source_file"] = source.path.name
    return normalized[
        [
            "model_name",
            "model_family",
            "input_features",
            "feature_selection",
            "inference_type",
            "result_role",
            "legacy_warning",
            "source_file",
            "fold_index",
            "subject_id",
            "y_true",
            "y_score",
            "y_pred",
        ]
    ]


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
