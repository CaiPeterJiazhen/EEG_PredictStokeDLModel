from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.evaluation.statistical_validation import (
    METRIC_NAMES,
    align_prediction_frames,
    bootstrap_metric_ci,
    calibration_metrics,
    exact_binomial_accuracy_pvalue,
    label_permutation_test,
    mcnemar_test,
    paired_bootstrap_difference,
    subject_level_predictions,
)
from eeg_recovery.training.metrics import binary_classification_metrics


PRIMARY_MODELS = (
    "ML_EEG_updated_no_selector_logistic_l1",
    "no_SSL_CNN_updated_sub05_sub28_seedensemble10",
    "residual_aware_SSL_CNN_seedmean10",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run statistical validation for locked manuscript models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=123)
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    predictions = load_or_build_locked_predictions(output_root)
    main_predictions = predictions[predictions["model_name"].isin(PRIMARY_MODELS)].copy()
    if main_predictions.empty:
        raise ValueError("No primary locked models were available for statistical validation.")

    ci_rows, permutation_rows = model_uncertainty_rows(
        main_predictions,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )
    comparison_rows = pairwise_comparison_rows(
        main_predictions,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
    )

    statistics_dir = output_root / "results" / "statistics"
    figures_dir = output_root / "results" / "figures" / "paper"
    docs_dir = output_root / "docs"
    for directory in (statistics_dir, figures_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    ci_rows.to_csv(statistics_dir / "model_metric_confidence_intervals.csv", index=False)
    comparison_rows.to_csv(statistics_dir / "model_pairwise_comparisons.csv", index=False)
    permutation_rows.to_csv(statistics_dir / "model_permutation_tests.csv", index=False)
    plot_calibration_curves(main_predictions, figures_dir / "calibration_curves.png")
    write_statistical_summary(
        docs_dir / "statistical_validation_summary.md",
        ci_rows,
        comparison_rows,
        permutation_rows,
    )
    print(f"Wrote {statistics_dir / 'model_metric_confidence_intervals.csv'}")
    print(f"Wrote {statistics_dir / 'model_pairwise_comparisons.csv'}")
    print(f"Wrote {figures_dir / 'calibration_curves.png'}")


def load_or_build_locked_predictions(output_root: Path) -> pd.DataFrame:
    path = output_root / "results" / "predictions" / "paper_locked_model_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    module = _load_script37_module()
    predictions = module.collect_locked_predictions(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return predictions


def model_uncertainty_rows(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ci_rows = []
    permutation_rows = []
    for model_name, group in predictions.groupby("model_name", sort=False):
        subject = subject_level_predictions(group)
        y_true = subject["y_true"].to_numpy(int)
        y_score = subject["y_score"].to_numpy(float)
        point = binary_classification_metrics(y_true, y_score)
        ci = bootstrap_metric_ci(
            y_true,
            y_score,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        cal = calibration_metrics(y_true, y_score)
        ci_rows.append(
            {
                "model_name": model_name,
                "n_subjects": int(len(subject)),
                **point,
                **{f"{key}_ci": value for key, value in ci.items() if key != "n_subjects"},
                "binomial_accuracy_p": exact_binomial_accuracy_pvalue(y_true, y_score),
                **cal,
            }
        )
        for metric in ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"):
            permutation = label_permutation_test(
                y_true,
                y_score,
                metric=metric,
                n_permutations=n_permutations,
                random_state=random_state + len(permutation_rows),
            )
            permutation_rows.append({"model_name": model_name, **permutation})
    return pd.DataFrame(ci_rows), pd.DataFrame(permutation_rows)


def pairwise_comparison_rows(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    by_model = {model: group for model, group in predictions.groupby("model_name", sort=False)}
    ordered = [model for model in PRIMARY_MODELS if model in by_model]
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            y_true, score_a, score_b, subjects = align_prediction_frames(by_model[first], by_model[second])
            for metric in METRIC_NAMES:
                diff = paired_bootstrap_difference(
                    y_true,
                    score_a,
                    score_b,
                    metric=metric,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state + len(rows),
                )
                rows.append(
                    {
                        "model_a": first,
                        "model_b": second,
                        "n_subjects": int(len(subjects)),
                        **diff,
                    }
                )
            rows.append(
                {
                    "model_a": first,
                    "model_b": second,
                    "n_subjects": int(len(subjects)),
                    "metric": "mcnemar_hard_predictions",
                    **mcnemar_test(y_true, score_a, score_b),
                }
            )
    return pd.DataFrame(rows)


def plot_calibration_curves(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Ideal")
    for model_name, group in predictions.groupby("model_name", sort=False):
        subject = subject_level_predictions(group)
        y_true = subject["y_true"].to_numpy(int)
        y_score = subject["y_score"].to_numpy(float)
        bins = np.linspace(0, 1, 6)
        xs = []
        ys = []
        for low, high in zip(bins[:-1], bins[1:], strict=True):
            if high == 1:
                mask = (y_score >= low) & (y_score <= high)
            else:
                mask = (y_score >= low) & (y_score < high)
            if not mask.any():
                continue
            xs.append(float(np.mean(y_score[mask])))
            ys.append(float(np.mean(y_true[mask])))
        ax.plot(xs, ys, marker="o", label=model_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Subject-level calibration curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_statistical_summary(
    path: Path,
    ci_rows: pd.DataFrame,
    comparisons: pd.DataFrame,
    permutations: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Statistical Validation Summary",
        "",
        "All uncertainty estimates use subject-level LOSO predictions. Bootstrap resampling is over subjects; no segment-level rows or seed rows are treated as independent observations.",
        "",
        "## Model Confidence Intervals And Calibration",
        "",
        _to_markdown(
            ci_rows[
                [
                    "model_name",
                    "n_subjects",
                    "accuracy",
                    "balanced_accuracy",
                    "roc_auc",
                    "pr_auc",
                    "brier_score",
                    "ece",
                    "calibration_intercept",
                    "calibration_slope",
                    "binomial_accuracy_p",
                ]
            ]
        ),
        "",
        "## Paired Model Comparisons",
        "",
        _to_markdown(comparisons.head(40)),
        "",
        "## Permutation Tests",
        "",
        _to_markdown(permutations.head(40)),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_script37_module():
    script_path = PROJECT_ROOT / "scripts" / "37_build_paper_locked_results.py"
    spec = spec_from_file_location("paper_locked_results_for_stats", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
