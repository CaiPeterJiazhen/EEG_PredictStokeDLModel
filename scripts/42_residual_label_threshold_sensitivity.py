from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.evaluation.statistical_validation import subject_level_predictions
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.metrics import binary_classification_metrics


CURRENT_THRESHOLD = 1.5
MODEL_PREDICTION_SOURCES = (
    (
        "ML_EEG_updated_no_selector_logistic_l1",
        "ml_baseline_loso_predictions_updated_sub05_sub28_psdfcwpli_no_selector.csv",
        "logistic_l1",
    ),
    (
        "no_SSL_CNN_updated_sub05_sub28_seedensemble10",
        "dl_loso_predictions_updated_sub05_sub28_10seed_no_ssl_psdfcwpli_gated_schemeA_seedensemble10.csv",
        None,
    ),
    (
        "residual_aware_SSL_CNN_seedmean10",
        "seedmean_patient_barlow_residualaware_highrank_swa_clsalpha1_10seed.csv",
        None,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate sensitivity to residual-label threshold definitions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--fixed-thresholds", nargs="+", type=float, default=[0.0, 1.5, 3.0])
    parser.add_argument("--margins", nargs="+", type=float, default=[0.5, 1.0])
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    labels = load_supervised_label_table(path_config)
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    predictions = load_model_predictions(output_root)
    rows = evaluate_threshold_sensitivity_for_predictions(
        predictions,
        labels,
        fixed_thresholds=args.fixed_thresholds,
        margins=args.margins,
    )

    metrics_dir = output_root / "results" / "metrics"
    tables_dir = output_root / "results" / "tables"
    docs_dir = output_root / "docs"
    for directory in (metrics_dir, tables_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rows.to_csv(metrics_dir / "residual_threshold_sensitivity.csv", index=False)
    (tables_dir / "residual_threshold_sensitivity.md").write_text(_to_markdown(rows), encoding="utf-8")
    write_threshold_doc(docs_dir / "residual_threshold_sensitivity.md", rows, labels)
    print(f"Wrote {metrics_dir / 'residual_threshold_sensitivity.csv'}")
    print(f"Wrote {tables_dir / 'residual_threshold_sensitivity.md'}")


def load_model_predictions(output_root: Path) -> pd.DataFrame:
    frames = []
    for model_name, filename, source_model in MODEL_PREDICTION_SOURCES:
        path = output_root / "results" / "predictions" / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if source_model is not None and "model" in frame.columns:
            frame = frame[frame["model"] == source_model].copy()
        frame["model"] = model_name
        frame["subject_id"] = frame["subject_id"].map(normalize_subject_id)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No locked model prediction files were found for threshold sensitivity.")
    return pd.concat(frames, ignore_index=True)


def fold_local_median_thresholds(label_table: pd.DataFrame) -> pd.DataFrame:
    labels = label_table.loc[:, ["subject_id", "Residual"]].copy()
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    rows = []
    for subject_id in labels["subject_id"].tolist():
        train = labels[labels["subject_id"] != subject_id]
        threshold = float(train["Residual"].median())
        rows.append({"subject_id": subject_id, "residual_threshold": threshold})
    return pd.DataFrame(rows)


def evaluate_threshold_sensitivity_for_predictions(
    predictions: pd.DataFrame,
    label_table: pd.DataFrame,
    *,
    fixed_thresholds: Sequence[float],
    margins: Sequence[float],
) -> pd.DataFrame:
    labels = label_table.loc[:, ["subject_id", "Residual"]].copy()
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    label_by_subject = labels.set_index("subject_id", drop=False)
    fold_thresholds = fold_local_median_thresholds(labels).set_index("subject_id")
    cohort_median = float(labels["Residual"].median())
    definitions: list[dict[str, object]] = [
        {"name": "current_fixed_1.5", "kind": "fixed", "threshold": CURRENT_THRESHOLD, "margin": np.nan},
        {"name": "cohort_median", "kind": "fixed", "threshold": cohort_median, "margin": np.nan},
        {"name": "fold_local_train_median", "kind": "fold_local", "threshold": np.nan, "margin": np.nan},
    ]
    for threshold in fixed_thresholds:
        definitions.append({"name": f"fixed_{_safe_float(threshold)}", "kind": "fixed", "threshold": float(threshold), "margin": np.nan})
    for margin in margins:
        definitions.append(
            {
                "name": f"exclude_margin_{_safe_float(margin)}_at_1.5",
                "kind": "exclude_margin",
                "threshold": CURRENT_THRESHOLD,
                "margin": float(margin),
            }
        )

    rows: list[dict[str, object]] = []
    for model_name, model_frame in predictions.groupby("model", sort=True):
        subject_predictions = _subject_scores_for_thresholding(model_frame)
        merged = subject_predictions.merge(labels, on="subject_id", how="inner", validate="one_to_one")
        for definition in definitions:
            evaluated = merged.copy()
            if definition["kind"] == "fold_local":
                evaluated = evaluated.merge(
                    fold_thresholds,
                    left_on="subject_id",
                    right_index=True,
                    how="inner",
                    validate="one_to_one",
                )
            else:
                evaluated["residual_threshold"] = float(definition["threshold"])
            if definition["kind"] == "exclude_margin":
                margin = float(definition["margin"])
                keep = (evaluated["Residual"] - evaluated["residual_threshold"]).abs() > margin
                n_excluded = int((~keep).sum())
                evaluated = evaluated.loc[keep].copy()
            else:
                n_excluded = 0
            if evaluated.empty or evaluated["Residual"].nunique() == 0:
                metrics = {name: np.nan for name in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score", "roc_auc", "pr_auc"]}
                y_true = np.asarray([], dtype=int)
            else:
                y_true = (evaluated["Residual"].to_numpy(float) <= evaluated["residual_threshold"].to_numpy(float)).astype(int)
                metrics = binary_classification_metrics(y_true, evaluated["y_score"].to_numpy(float))
            rows.append(
                {
                    "model": model_name,
                    "label_definition": definition["name"],
                    "threshold_type": definition["kind"],
                    "residual_threshold": definition["threshold"],
                    "margin": definition["margin"],
                    "n_subjects": int(len(evaluated)),
                    "n_positive": int(y_true.sum()) if len(y_true) else 0,
                    "n_negative": int(len(y_true) - y_true.sum()) if len(y_true) else 0,
                    "n_excluded_near_threshold": n_excluded,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _subject_scores_for_thresholding(frame: pd.DataFrame) -> pd.DataFrame:
    if "y_true" in frame.columns:
        return subject_level_predictions(frame).drop(columns=["y_true", "y_pred", "n_source_rows"])
    data = frame.copy()
    data["subject_id"] = data["subject_id"].map(normalize_subject_id)
    return (
        data.groupby("subject_id", as_index=False)["y_score"]
        .mean()
        .sort_values("subject_id")
        .reset_index(drop=True)
    )


def write_threshold_doc(path: Path, rows: pd.DataFrame, labels: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = rows[rows["label_definition"] == "current_fixed_1.5"]
    current_display = current[
        ["model", "n_subjects", "n_positive", "n_negative", "accuracy", "roc_auc", "pr_auc", "brier_score"]
    ]
    fold_local = rows[rows["label_definition"] == "fold_local_train_median"]
    text = [
        "# Residual Label Threshold Sensitivity",
        "",
        "Primary labels in the current project are derived from the residual threshold of 1.5: `Residual <= 1.5` is the proportional-recovery group.",
        "",
        f"The supervised cohort median residual is {float(labels['Residual'].median()):.3f}. If this median was selected after viewing the full cohort, it is a task-definition choice that can introduce optimistic bias at the label-definition level. The model training and LOSO prediction files are unchanged here; this script only re-scores existing subject-level predictions under alternative label definitions.",
        "",
        "## Locked Current Definition",
        "",
        _to_markdown(current_display),
        "",
        "## Fold-local Train-only Median",
        "",
        _to_markdown(fold_local[["model", "n_subjects", "n_positive", "n_negative", "accuracy", "roc_auc", "pr_auc", "brier_score"]]),
        "",
        "Near-threshold exclusion rows report how many subjects were excluded before metrics were recomputed. These sensitivity checks are descriptive and must not replace the primary locked label without a new pre-specified analysis plan.",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_csv(index=False)


def _safe_float(value: float) -> str:
    return str(float(value))


if __name__ == "__main__":
    main()
