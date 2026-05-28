from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.training.segment_barlow_model_selection import (
    classification_metrics_from_scores,
    load_required_prediction_csv,
)


BOOTSTRAP_METRICS = ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"]
SSL_ONLY_MODEL_GROUPS = {
    "psd_segbarlow_ssl_cnn",
    "wpli_segbarlow_ssl_cnn",
    "psd_wpli_segbarlow_equal_weight",
}
PAIRED_COMPARISONS = [
    ("no_ssl_stable_cnn", "psd_segbarlow_ssl_cnn"),
    ("no_ssl_stable_cnn", "wpli_segbarlow_ssl_cnn"),
    ("no_ssl_stable_cnn", "psd_wpli_segbarlow_equal_weight"),
    ("wpli_segbarlow_ssl_cnn", "psd_wpli_segbarlow_equal_weight"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap and paired statistical comparisons for Segment Barlow model-selection predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--model-selection-summary", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutation", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260527)
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    metric_dir = output_root / "results" / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.model_selection_summary)
    primary = _primary_prediction_rows(summary)
    predictions = {
        row["model_group"]: _load_prediction_row(row)
        for _, row in primary.iterrows()
        if isinstance(row.get("prediction_path"), str) and row["prediction_path"]
    }
    rng = np.random.default_rng(args.random_seed)
    rows: list[dict[str, object]] = []

    for _, row in primary.iterrows():
        model_group = str(row["model_group"])
        if model_group not in predictions:
            continue
        rows.extend(
            _bootstrap_ci_rows(
                predictions[model_group],
                model_group=model_group,
                threshold=float(row["threshold"]),
                n_bootstrap=args.bootstrap,
                rng=rng,
            )
        )

    for model_a, model_b in PAIRED_COMPARISONS:
        if model_a not in predictions or model_b not in predictions:
            continue
        rows.extend(_paired_comparison_rows(predictions[model_a], predictions[model_b], model_a=model_a, model_b=model_b, rng=rng))

    seen_permutation_models = set()
    for selection_scope, selected_row in _selected_permutation_rows(primary):
        if selected_row is None or selected_row["model_group"] not in predictions:
            continue
        model_group = str(selected_row["model_group"])
        if model_group in seen_permutation_models and selection_scope != "overall_selected":
            continue
        seen_permutation_models.add(model_group)
        permutation_row = _permutation_random_label_row(
            predictions[model_group],
            model_group=model_group,
            threshold=float(selected_row["threshold"]),
            n_permutations=args.permutation,
            rng=rng,
        )
        permutation_row["selection_scope"] = selection_scope
        rows.append(
            permutation_row
        )

    output_path = metric_dir / _tagged_output_name("segment_barlow_statistical_comparison.csv", output_tag=args.output_tag)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote statistical comparison: {output_path}")


def _primary_prediction_rows(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"model_group", "aggregation", "threshold_method", "prediction_path", "threshold"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Model-selection summary missing required column(s): {', '.join(sorted(missing))}")
    primary = summary.loc[
        (summary["aggregation"] == "seed_mean_probability")
        & (summary["threshold_method"] == "fixed_0.5")
    ].copy()
    if primary.empty:
        raise ValueError("No fixed-threshold seed_mean_probability rows found in model-selection summary.")
    return primary


def _load_prediction_row(row: pd.Series) -> pd.DataFrame:
    frame = load_required_prediction_csv(row["prediction_path"], model_name=str(row["model_group"]))
    if "y_pred" not in frame.columns:
        frame["y_pred"] = (frame["y_score"].to_numpy(dtype=float) >= float(row["threshold"])).astype(int)
    return frame


def _bootstrap_ci_rows(
    predictions: pd.DataFrame,
    *,
    model_group: str,
    threshold: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_score = predictions["y_score"].to_numpy(dtype=float)
    n_subjects = len(predictions)
    observed = classification_metrics_from_scores(y_true, y_score, threshold=threshold)
    values = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_subjects, size=n_subjects)
        metrics = classification_metrics_from_scores(y_true[indices], y_score[indices], threshold=threshold)
        for metric in BOOTSTRAP_METRICS:
            values[metric].append(metrics[metric])
    rows = []
    for metric in BOOTSTRAP_METRICS:
        samples = np.asarray(values[metric], dtype=float)
        rows.append(
            {
                "analysis_type": "bootstrap_ci",
                "model_group": model_group,
                "metric": metric,
                "observed": observed[metric],
                "ci_low": _nanpercentile(samples, 2.5),
                "ci_high": _nanpercentile(samples, 97.5),
                "n_resamples": n_bootstrap,
                "n_subjects": n_subjects,
            }
        )
    return rows


def _paired_comparison_rows(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    aligned = first[["subject_id", "y_true", "y_score", "y_pred"]].merge(
        second[["subject_id", "y_true", "y_score", "y_pred"]],
        on="subject_id",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if not (aligned["y_true_a"].to_numpy(dtype=int) == aligned["y_true_b"].to_numpy(dtype=int)).all():
        raise ValueError(f"Cannot compare {model_a} and {model_b}: y_true mismatch.")
    y_true = aligned["y_true_a"].to_numpy(dtype=int)
    correct_a = aligned["y_pred_a"].to_numpy(dtype=int) == y_true
    correct_b = aligned["y_pred_b"].to_numpy(dtype=int) == y_true
    b_only = int((~correct_a & correct_b).sum())
    a_only = int((correct_a & ~correct_b).sum())
    discordant = b_only + a_only
    sign_p = float(binomtest(b_only, discordant, p=0.5).pvalue) if discordant else 1.0
    rows = [
        {
            "analysis_type": "paired_correctness_sign_test",
            "model_a": model_a,
            "model_b": model_b,
            "metric": "correctness_b_minus_a",
            "observed": float(correct_b.mean() - correct_a.mean()),
            "b_correct_a_wrong": b_only,
            "a_correct_b_wrong": a_only,
            "discordant_subjects": discordant,
            "p_value": sign_p,
            "n_subjects": len(aligned),
        },
    ]
    rows.extend(_paired_metric_bootstrap_rows(aligned, model_a=model_a, model_b=model_b, rng=rng))
    return rows


def _paired_metric_bootstrap_rows(
    aligned: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    rng: np.random.Generator,
    n_resamples: int = 5000,
) -> list[dict[str, object]]:
    y_true = aligned["y_true_a"].to_numpy(dtype=int)
    score_a = aligned["y_score_a"].to_numpy(dtype=float)
    score_b = aligned["y_score_b"].to_numpy(dtype=float)
    metrics = ("roc_auc", "pr_auc", "brier_score")
    observed_a = classification_metrics_from_scores(y_true, score_a, threshold=0.5)
    observed_b = classification_metrics_from_scores(y_true, score_b, threshold=0.5)
    values = {metric: [] for metric in metrics}
    for _ in range(n_resamples):
        indices = rng.integers(0, len(aligned), size=len(aligned))
        boot_a = classification_metrics_from_scores(y_true[indices], score_a[indices], threshold=0.5)
        boot_b = classification_metrics_from_scores(y_true[indices], score_b[indices], threshold=0.5)
        for metric in metrics:
            values[metric].append(float(boot_b[metric]) - float(boot_a[metric]))
    return [
        {
            "analysis_type": "paired_score_bootstrap",
            "model_a": model_a,
            "model_b": model_b,
            "metric": f"{metric}_b_minus_a",
            "observed": float(observed_b[metric]) - float(observed_a[metric]),
            "ci_low": _nanpercentile(np.asarray(values[metric], dtype=float), 2.5),
            "ci_high": _nanpercentile(np.asarray(values[metric], dtype=float), 97.5),
            "n_resamples": n_resamples,
            "n_subjects": len(aligned),
        }
        for metric in metrics
    ]


def _select_final_candidate(primary: pd.DataFrame) -> pd.Series | None:
    candidate = primary.sort_values(
        ["balanced_accuracy", "accuracy", "roc_auc", "pr_auc"],
        ascending=[False, False, False, False],
    )
    if candidate.empty:
        return None
    return candidate.iloc[0]


def _selected_permutation_rows(primary: pd.DataFrame) -> list[tuple[str, pd.Series | None]]:
    ssl_primary = primary.loc[primary["model_group"].isin(SSL_ONLY_MODEL_GROUPS)].copy()
    return [
        ("overall_selected", _select_final_candidate(primary)),
        ("ssl_only_selected", _select_final_candidate(ssl_primary) if not ssl_primary.empty else None),
    ]


def _permutation_random_label_row(
    predictions: pd.DataFrame,
    *,
    model_group: str,
    threshold: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_score = predictions["y_score"].to_numpy(dtype=float)
    observed = classification_metrics_from_scores(y_true, y_score, threshold=threshold)["balanced_accuracy"]
    permuted = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(y_true)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            permuted.append(classification_metrics_from_scores(shuffled, y_score, threshold=threshold)["balanced_accuracy"])
    permuted_array = np.asarray(permuted, dtype=float)
    p_value = float((np.sum(permuted_array >= float(observed)) + 1) / (n_permutations + 1))
    return {
        "analysis_type": "random_label_permutation",
        "model_group": model_group,
        "metric": "balanced_accuracy",
        "observed": observed,
        "null_mean": float(np.nanmean(permuted_array)),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "n_subjects": len(predictions),
    }


def _nanpercentile(values: np.ndarray, percentile: float) -> float:
    if np.isfinite(values).any():
        return float(np.nanpercentile(values, percentile))
    return float("nan")


def _tagged_output_name(filename: str, *, output_tag: str | None) -> str:
    token = _output_tag_token(output_tag)
    if token is None:
        return filename
    path = Path(filename)
    stem = path.stem
    if stem.startswith("segment_barlow_"):
        stem = stem.replace("segment_barlow_", f"segment_barlow_{token}_", 1)
    else:
        stem = f"{stem}_{token}"
    return f"{stem}{path.suffix}"


def _output_tag_token(output_tag: str | None) -> str | None:
    if not output_tag:
        return None
    if output_tag in {"10seed", "10seed_locked"}:
        return "10seed"
    return "".join(character if character.isalnum() else "_" for character in output_tag).strip("_")


if __name__ == "__main__":
    main()
