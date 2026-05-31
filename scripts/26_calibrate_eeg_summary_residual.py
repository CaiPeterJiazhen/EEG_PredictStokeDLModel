from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.features.clinical_summary import baseline_clinical_feature_matrix
from eeg_recovery.features.complexity import compute_subject_complexity_summary_features
from eeg_recovery.features.eeg_reactivity import compute_subject_eeg_reactivity_features
from eeg_recovery.features.eeg_summary import compute_subject_eeg_summary_features
from eeg_recovery.features.imaginary_coherence_summary import compute_subject_imaginary_coherence_summary_features
from eeg_recovery.features.qeeg_slowing import compute_subject_qeeg_slowing_features
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.training.eeg_summary_calibration import (
    EEGSummaryCalibrationConfig,
    nested_eeg_summary_residual_calibration,
    nested_univariate_eeg_summary_residual_calibration,
    select_summary_feature_columns,
)
from eeg_recovery.training.train_feature_ssl import _safe_run_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested patient-LOO summary calibration for CNN/SSL-CNN seedmean predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--base-predictions", required=True, help="Seedmean prediction CSV with subject_id,y_true,y_score.")
    parser.add_argument("--base-name", default="base_cnn")
    parser.add_argument(
        "--summary-source",
        choices=("eeg", "clinical", "complexity", "reactivity", "imagcoh", "qeeg_slowing"),
        default="eeg",
        help="Feature source for the low-capacity residual calibrator.",
    )
    parser.add_argument("--selection-objective", choices=("brier", "rank", "aucpr", "balanced_accuracy"), default="rank")
    parser.add_argument("--candidate-cs", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1])
    parser.add_argument("--candidate-weights", type=float, nargs="+", default=[round(value * 0.05, 2) for value in range(19)])
    parser.add_argument(
        "--fusion-modes",
        nargs="+",
        choices=("linear", "geometric", "veto"),
        default=["linear"],
        help="Nested-OOF probability fusion modes to search.",
    )
    parser.add_argument(
        "--feature-directions",
        type=int,
        nargs="+",
        choices=(-1, 1),
        default=[1, -1],
        help="Allowed univariate feature directions; use -1 to lock inverse association.",
    )
    parser.add_argument(
        "--score-transform-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Positive logit scales applied after score fusion; values above 1 sharpen probabilities.",
    )
    parser.add_argument(
        "--candidate-thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Optional thresholds selected inside each outer training fold by inner OOF predictions.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--preserve-base-classification",
        action="store_true",
        help="Skip inner-OOF calibration candidates that reduce base accuracy or balanced accuracy.",
    )
    parser.add_argument("--min-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--min-balanced-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--min-sensitivity-delta", type=float, default=-1.0)
    parser.add_argument("--min-specificity-delta", type=float, default=-1.0)
    parser.add_argument("--include-feature-regex", default=None)
    parser.add_argument("--exclude-feature-regex", default=None)
    parser.add_argument(
        "--univariate-feature-search",
        action="store_true",
        help="Use nested OOF selection of one interpretable EEG summary feature, direction, and residual weight.",
    )
    parser.add_argument("--output-tag", default="eegsummary_nested_calibration")
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    labels = load_supervised_label_table(path_config).sort_values("subject_id").reset_index(drop=True)
    if args.summary_source == "clinical":
        summary_features, feature_names = baseline_clinical_feature_matrix(labels)
    elif args.summary_source == "complexity":
        vectors = [compute_subject_complexity_summary_features(path_config.output_root, subject_id) for subject_id in labels["subject_id"]]
        feature_names = vectors[0].feature_names
        summary_features = np.vstack([vector.values for vector in vectors]).astype(np.float32)
    elif args.summary_source == "reactivity":
        vectors = [compute_subject_eeg_reactivity_features(path_config.output_root, subject_id) for subject_id in labels["subject_id"]]
        feature_names = vectors[0].feature_names
        summary_features = np.vstack([vector.values for vector in vectors]).astype(np.float32)
    elif args.summary_source == "imagcoh":
        vectors = [compute_subject_imaginary_coherence_summary_features(path_config.output_root, subject_id) for subject_id in labels["subject_id"]]
        feature_names = vectors[0].feature_names
        summary_features = np.vstack([vector.values for vector in vectors]).astype(np.float32)
    elif args.summary_source == "qeeg_slowing":
        vectors = [compute_subject_qeeg_slowing_features(path_config.output_root, subject_id) for subject_id in labels["subject_id"]]
        feature_names = vectors[0].feature_names
        summary_features = np.vstack([vector.values for vector in vectors]).astype(np.float32)
    else:
        vectors = [compute_subject_eeg_summary_features(path_config.output_root, subject_id) for subject_id in labels["subject_id"]]
        feature_names = vectors[0].feature_names
        summary_features = np.vstack([vector.values for vector in vectors]).astype(np.float32)
    summary_features, feature_names = select_summary_feature_columns(
        summary_features,
        feature_names,
        include_regex=args.include_feature_regex,
        exclude_regex=args.exclude_feature_regex,
    )
    base_predictions = pd.read_csv(args.base_predictions).sort_values("subject_id").reset_index(drop=True)
    expected_subjects = labels["subject_id"].tolist()
    if base_predictions["subject_id"].tolist() != expected_subjects:
        raise SystemExit("Base prediction subjects do not match the supervised label table after subject_id sorting.")
    if not np.array_equal(base_predictions["y_true"].to_numpy(dtype=int), labels["label"].to_numpy(dtype=int)):
        raise SystemExit("Base prediction labels do not match the supervised label table.")

    calibration_config = EEGSummaryCalibrationConfig(
        candidate_cs=tuple(args.candidate_cs),
        candidate_weights=tuple(args.candidate_weights),
        candidate_fusion_modes=tuple(args.fusion_modes),
        candidate_feature_directions=tuple(args.feature_directions),
        candidate_score_transform_scales=tuple(args.score_transform_scales),
        candidate_thresholds=tuple(args.candidate_thresholds) if args.candidate_thresholds is not None else None,
        selection_objective=args.selection_objective,
        threshold=args.threshold,
        preserve_base_classification=args.preserve_base_classification,
        min_accuracy_delta=args.min_accuracy_delta,
        min_balanced_accuracy_delta=args.min_balanced_accuracy_delta,
        min_sensitivity_delta=args.min_sensitivity_delta,
        min_specificity_delta=args.min_specificity_delta,
    )
    calibrator = (
        nested_univariate_eeg_summary_residual_calibration
        if args.univariate_feature_search
        else nested_eeg_summary_residual_calibration
    )
    predictions, metrics, choices, importance = calibrator(
        base_predictions,
        summary_features,
        feature_names,
        calibration_config,
    )
    run_name = _safe_run_name(f"{args.base_name}_{args.output_tag}_{args.selection_objective}")
    root = Path(path_config.output_root)
    paths = {
        "predictions": root / "results" / "predictions" / f"dl_loso_predictions_{run_name}.csv",
        "metrics": root / "results" / "metrics" / f"dl_model_comparison_{run_name}.csv",
        "choices": root / "results" / "metrics" / f"{run_name}_choices.csv",
        "importance": root / "results" / "metrics" / f"{run_name}_feature_importance.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for frame in (predictions, metrics, choices, importance):
        frame["base_name"] = args.base_name
        frame["base_predictions"] = str(Path(args.base_predictions))
        frame["summary_source"] = args.summary_source
        frame["selection_objective"] = args.selection_objective
        frame["output_tag"] = args.output_tag
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    choices.to_csv(paths["choices"], index=False)
    importance.to_csv(paths["importance"], index=False)
    for key, path in paths.items():
        print(f"Wrote {key}: {path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
