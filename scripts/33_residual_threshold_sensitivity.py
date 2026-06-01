from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.metrics import binary_classification_metrics


MAIN_THRESHOLD = 1.5
PREDICTION_FILE = (
    "seedmean_patient_barlow_residualaware_highrank_swa_clsalpha1_10seed.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Residual-label sensitivity analysis for locked final model."
    )
    parser.add_argument("--config", default="configs/paths.example.yaml")
    args = parser.parse_args()

    config = load_path_config(args.config)
    output_root = config.output_root
    label_table = load_supervised_label_table(config)
    label_table["subject_id"] = label_table["subject_id"].map(normalize_subject_id)

    predictions_path = output_root / "results" / "predictions" / PREDICTION_FILE
    predictions = pd.read_csv(predictions_path)
    predictions["subject_id"] = predictions["subject_id"].map(normalize_subject_id)

    merged = label_table.merge(
        predictions[["subject_id", "y_score"]],
        on="subject_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 19:
        raise ValueError(f"Expected 19 matched LOSO predictions, found {len(merged)}")

    thresholds = _threshold_candidates(merged["Residual"].to_numpy(dtype=float))
    rows = []
    for threshold_name, threshold in thresholds:
        y_true = (merged["Residual"].to_numpy(dtype=float) <= threshold).astype(int)
        y_score = merged["y_score"].to_numpy(dtype=float)
        metrics = binary_classification_metrics(y_true, y_score, threshold=0.5)
        rows.append(
            {
                "threshold_name": threshold_name,
                "residual_threshold": threshold,
                "n_positive": int(y_true.sum()),
                "n_negative": int(len(y_true) - y_true.sum()),
                **metrics,
            }
        )

    current_signed_distance = MAIN_THRESHOLD - merged["Residual"].to_numpy(dtype=float)
    corr_residual = spearmanr(merged["y_score"], merged["Residual"])
    corr_signed = spearmanr(merged["y_score"], current_signed_distance)
    rows.extend(
        [
            {
                "threshold_name": "continuous_y_score_vs_residual",
                "residual_threshold": np.nan,
                "n_positive": np.nan,
                "n_negative": np.nan,
                "spearman_r": float(corr_residual.statistic),
                "spearman_p": float(corr_residual.pvalue),
                "note": "Higher residual indicates poorer proportional-recovery fit.",
            },
            {
                "threshold_name": "continuous_y_score_vs_signed_distance",
                "residual_threshold": MAIN_THRESHOLD,
                "n_positive": np.nan,
                "n_negative": np.nan,
                "spearman_r": float(corr_signed.statistic),
                "spearman_p": float(corr_signed.pvalue),
                "note": "signed_distance = 1.5 - residual; higher values align with positive label.",
            },
        ]
    )

    metrics_path = output_root / "results" / "metrics" / "residual_threshold_sensitivity.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    doc_path = output_root / "docs" / "residual_threshold_sensitivity.md"
    _write_doc(doc_path, pd.DataFrame(rows), merged)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {doc_path}")


def _threshold_candidates(residual: np.ndarray) -> list[tuple[str, float]]:
    median = float(np.median(residual))
    candidates = [
        ("locked_current_threshold_1p5", MAIN_THRESHOLD),
        ("cohort_median_threshold", median),
        ("lower_tertile_like_threshold", float(np.quantile(residual, 1 / 3))),
        ("upper_tertile_like_threshold", float(np.quantile(residual, 2 / 3))),
        ("balanced_nearest_integer_threshold", float(round(median))),
    ]
    seen: set[tuple[str, float]] = set()
    unique: list[tuple[str, float]] = []
    for name, threshold in candidates:
        key = (name, round(threshold, 10))
        if key not in seen:
            seen.add(key)
            unique.append((name, threshold))
    return unique


def _write_doc(path: Path, rows: pd.DataFrame, merged: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    locked = rows.loc[rows["threshold_name"] == "locked_current_threshold_1p5"].iloc[0]
    corr_res = rows.loc[rows["threshold_name"] == "continuous_y_score_vs_residual"].iloc[0]
    corr_dist = rows.loc[
        rows["threshold_name"] == "continuous_y_score_vs_signed_distance"
    ].iloc[0]
    text = f"""# Residual Threshold Sensitivity

This analysis keeps the locked final model unchanged:
`residualaware_highrank_swa_clsalpha1`, fixed classification-head inference, and fixed threshold 0.5.
It does not reselect the outcome threshold or the model.

## Locked Outcome Definition

The primary binary label remains the current cohort residual threshold:
`Residual <= 1.5` is proportional recovery and `Residual > 1.5` is poor recovery.
At this locked threshold the cohort has {int(locked['n_positive'])} positive and {int(locked['n_negative'])} negative subjects.

The final seed-mean model at this threshold has accuracy {locked['accuracy']:.3f},
balanced accuracy {locked['balanced_accuracy']:.3f}, ROC AUC {locked['roc_auc']:.3f},
PR AUC {locked['pr_auc']:.3f}, and Brier score {locked['brier_score']:.3f}.

## Sensitivity Analysis

Alternative residual thresholds are reported only as sensitivity checks in
`results/metrics/residual_threshold_sensitivity.csv`. They must not be used to
replace the locked primary threshold without a new pre-specified analysis plan.

## Continuous Residual Association

The final model score has Spearman correlation with residual:
rho = {corr_res['spearman_r']:.3f}, p = {corr_res['spearman_p']:.4f}.
Because higher residual indicates poorer proportional-recovery fit, a negative
association is directionally expected.

Using signed distance (`1.5 - residual`), Spearman rho = {corr_dist['spearman_r']:.3f},
p = {corr_dist['spearman_p']:.4f}. This is the continuous target behind the
residual-aware auxiliary training heads.

## Rationale For Residual-Aware Auxiliary Objectives

The binary outcome is a thresholded version of a continuous residual. The
regression, pairwise-ranking, and soft-label heads use continuous residual
structure during training, while final inference remains the binary
classification head with threshold 0.5. This preserves the locked clinical
decision rule while reducing information loss from binarization.

## Cohort Note

This is a 19-patient LOSO pilot cohort. Threshold sensitivity is descriptive
and should be treated as manuscript support, not as evidence for a newly
optimized label definition.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
