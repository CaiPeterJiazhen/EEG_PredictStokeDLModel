from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.training.segment_barlow_model_selection import (
    aggregate_seed_mean_predictions,
    apply_leave_one_seed_thresholds,
    compute_per_subject_error_frequency,
    equal_weight_model_ensemble,
    evaluate_prediction_frame,
    leave_one_seed_thresholds,
    load_required_prediction_csv,
    validate_locked_seed_summary,
)


DEFAULT_SEEDS = [0, 1, 2, 3, 7, 13]
SINGLE_MODEL_GROUPS = {
    "no_ssl": "no_ssl_stable_cnn",
    "psd": "psd_segbarlow_ssl_cnn",
    "wpli": "wpli_segbarlow_ssl_cnn",
}
ENSEMBLE_DEFINITIONS = {
    "psd_wpli": {
        "model_group": "psd_wpli_segbarlow_equal_weight",
        "members": ("psd", "wpli"),
        "label": "psd+wpli",
        "output": "ensemble_predictions_psd_wpli_segbarlow_equal_weight.csv",
        "metrics": "ensemble_metrics_psd_wpli_segbarlow_equal_weight.csv",
    },
    "no_ssl_wpli": {
        "model_group": "no_ssl_wpli_equal_weight",
        "members": ("no_ssl", "wpli"),
        "label": "no_ssl+wpli",
        "output": "ensemble_predictions_no_ssl_wpli_equal_weight.csv",
        "metrics": "ensemble_metrics_no_ssl_wpli_equal_weight.csv",
    },
    "no_ssl_psd_wpli": {
        "model_group": "no_ssl_psd_wpli_equal_weight",
        "members": ("no_ssl", "psd", "wpli"),
        "label": "no_ssl+psd+wpli",
        "output": "ensemble_predictions_no_ssl_psd_wpli_equal_weight.csv",
        "metrics": "ensemble_metrics_no_ssl_psd_wpli_equal_weight.csv",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble PSD and FC/wPLI Segment Barlow SSL-CNN LOSO predictions without test-label tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--psd-results-dir", nargs="+", default=None)
    parser.add_argument("--wpli-results-dir", nargs="+", default=None)
    parser.add_argument("--no-ssl-results-dir", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--ensemble",
        nargs="+",
        choices=sorted(ENSEMBLE_DEFINITIONS),
        default=["psd_wpli", "no_ssl_wpli", "no_ssl_psd_wpli"],
    )
    parser.add_argument(
        "--threshold-method",
        nargs="+",
        choices=("fixed", "oof", "leave_one_seed_out_oof"),
        default=["fixed", "leave_one_seed_out_oof"],
    )
    parser.add_argument("--output-tag", default="segment_barlow_model_selection")
    parser.add_argument("--calibration-metric", choices=("balanced_accuracy", "accuracy", "f1"), default="balanced_accuracy")
    args = parser.parse_args()
    threshold_methods = _normalize_threshold_methods(args.threshold_method)

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    prediction_dir = output_root / "results" / "predictions"
    metric_dir = output_root / "results" / "metrics"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    seed_frames_by_model: dict[str, dict[int, pd.DataFrame]] = {
        "psd": _load_seed_frames(_result_roots(args.psd_results_dir, output_root, default_export="20260527_local_psd_segssl"), args.seeds, model_kind="psd"),
        "wpli": _load_seed_frames(_result_roots(args.wpli_results_dir, output_root, default_export="20260527_remote_fc_wpli_segssl"), args.seeds, model_kind="wpli"),
    }
    try:
        seed_frames_by_model["no_ssl"] = _load_seed_frames(
            _result_roots(args.no_ssl_results_dir, output_root),
            args.seeds,
            model_kind="no_ssl",
        )
    except FileNotFoundError as exc:
        print(f"[segment-barlow-selection] no-SSL predictions unavailable: {exc}")

    model_groups: dict[str, dict[int, pd.DataFrame]] = {}
    group_labels: dict[str, tuple[str, str]] = {}
    for source_name, model_group in SINGLE_MODEL_GROUPS.items():
        if source_name not in seed_frames_by_model:
            continue
        model_groups[model_group] = {
            seed: _annotate_model_frame(frame, model_group=model_group, seed=seed)
            for seed, frame in seed_frames_by_model[source_name].items()
        }
        group_labels[model_group] = (model_group, "single_model")

    ensemble_output_paths: dict[str, Path] = {}
    ensemble_metric_rows: dict[str, list[dict[str, object]]] = {}
    for ensemble_name in args.ensemble:
        definition = ENSEMBLE_DEFINITIONS[ensemble_name]
        members = definition["members"]
        if any(member not in seed_frames_by_model for member in members):
            print(f"[segment-barlow-selection] skipped {ensemble_name}: missing one or more members {members}")
            continue
        model_group = definition["model_group"]
        per_seed = {}
        for seed in args.seeds:
            member_frames = {member: seed_frames_by_model[member][seed] for member in members}
            per_seed[seed] = equal_weight_model_ensemble(
                member_frames,
                model_group=model_group,
                seed=seed,
            )
            per_seed[seed]["ensemble_members"] = definition["label"]
        model_groups[model_group] = per_seed
        group_labels[model_group] = (definition["label"], "equal")
        ensemble_predictions = pd.concat(per_seed.values(), ignore_index=True)
        ensemble_predictions["y_pred_fixed_0.5"] = (ensemble_predictions["y_score"].to_numpy(dtype=float) >= 0.5).astype(int)
        ensemble_path = prediction_dir / _tagged_output_name(definition["output"], output_tag=args.output_tag)
        ensemble_predictions.to_csv(ensemble_path, index=False)
        ensemble_output_paths[model_group] = ensemble_path

    metrics_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    error_frequency_frames: list[pd.DataFrame] = []
    sub_focus_frames: list[pd.DataFrame] = []

    for model_group, per_seed in model_groups.items():
        ensemble_members, ensemble_weighting = group_labels[model_group]
        threshold_map = leave_one_seed_thresholds(per_seed, metric=args.calibration_metric) if "leave_one_seed_out_oof" in threshold_methods else {}
        if threshold_map:
            for seed, threshold in threshold_map.items():
                threshold_rows.append(
                    {
                        "model_group": model_group,
                        "seed": seed,
                        "threshold": threshold.threshold,
                        "threshold_method": threshold.threshold_method,
                        "calibration_metric": threshold.metric,
                        "calibration_metric_value": threshold.metric_value,
                        "calibration_seed_count": threshold.calibration_seed_count,
                        "exploratory": threshold.exploratory,
                    }
                )

        for method in threshold_methods:
            seed_metric_rows = []
            prediction_frames_for_errors = []
            if method == "fixed":
                for seed, frame in per_seed.items():
                    fixed = _with_fixed_prediction(frame, threshold=0.5)
                    prediction_frames_for_errors.append(fixed)
                    row = evaluate_prediction_frame(
                        fixed,
                        model_group=model_group,
                        threshold=0.5,
                        threshold_method="fixed_0.5",
                        ensemble_members=ensemble_members,
                        ensemble_weighting=ensemble_weighting,
                        seed=seed,
                        n_seeds=len(per_seed),
                    )
                    seed_metric_rows.append(row)
                    metrics_rows.append(row)
            else:
                calibrated_all = apply_leave_one_seed_thresholds(per_seed, metric=args.calibration_metric)
                for seed, frame in calibrated_all.groupby("seed", sort=False):
                    seed_frame = frame.copy()
                    prediction_frames_for_errors.append(seed_frame)
                    threshold = float(seed_frame["threshold"].iloc[0])
                    row = evaluate_prediction_frame(
                        seed_frame,
                        model_group=model_group,
                        threshold=threshold,
                        threshold_method="leave_one_seed_out_oof",
                        ensemble_members=ensemble_members,
                        ensemble_weighting=ensemble_weighting,
                        seed=seed,
                        n_seeds=len(per_seed),
                    )
                    seed_metric_rows.append(row)
                    metrics_rows.append(row)

            if method == "fixed":
                seed_mean_threshold = 0.5
                seed_mean_method = "fixed_0.5"
            else:
                seed_mean_threshold = float(np.median([item.threshold for item in threshold_map.values()]))
                seed_mean_method = "leave_one_seed_out_oof_median_threshold"
            seed_mean_predictions = aggregate_seed_mean_predictions(
                per_seed,
                model_group=model_group,
                threshold=seed_mean_threshold,
                threshold_method=seed_mean_method,
                ensemble_members=ensemble_members,
                ensemble_weighting=f"{ensemble_weighting}+seed_mean_probability",
            )
            seed_mean_predictions["exploratory"] = False
            seed_mean_name = f"{model_group}_{seed_mean_method}_seed_mean_predictions.csv"
            seed_mean_path = prediction_dir / seed_mean_name
            seed_mean_predictions.to_csv(seed_mean_path, index=False)
            seed_mean_row = evaluate_prediction_frame(
                seed_mean_predictions,
                model_group=model_group,
                threshold=seed_mean_threshold,
                threshold_method=seed_mean_method,
                ensemble_members=ensemble_members,
                ensemble_weighting=f"{ensemble_weighting}+seed_mean_probability",
                seed="seed_mean",
                n_seeds=len(per_seed),
            )
            seed_mean_row["aggregation"] = "seed_mean_probability"
            seed_mean_row["prediction_path"] = str(seed_mean_path)
            summary_rows.append(seed_mean_row)

            seed_summary = _summarize_seed_metric_rows(seed_metric_rows)
            seed_summary.update(
                {
                    "model_group": model_group,
                    "threshold_method": "fixed_0.5" if method == "fixed" else "leave_one_seed_out_oof",
                    "ensemble_members": ensemble_members,
                    "ensemble_weighting": ensemble_weighting,
                    "n_subjects": int(next(iter(per_seed.values())).shape[0]),
                    "n_seeds": len(per_seed),
                    "aggregation": "seed_runs",
                    "prediction_path": "",
                    "exploratory": False,
                }
            )
            summary_rows.append(seed_summary)

            if method == "fixed":
                errors = compute_per_subject_error_frequency(prediction_frames_for_errors, model_group=model_group)
                error_frequency_frames.append(errors)
                sub_focus_frames.append(errors.loc[errors["subject_id"].isin(["sub09", "sub14"])].copy())

        if model_group in ensemble_output_paths:
            ensemble_metric_rows[model_group] = [
                row for row in metrics_rows if row["model_group"] == model_group
            ]

    for model_group, rows in ensemble_metric_rows.items():
        path = metric_dir / _tagged_output_name(_ensemble_metric_filename(model_group), output_tag=args.output_tag)
        pd.DataFrame(rows).to_csv(path, index=False)

    summary_frame = pd.DataFrame(summary_rows)
    if _output_tag_token(args.output_tag) == "10seed":
        validate_locked_seed_summary(summary_frame, expected_n_seeds=len(args.seeds))
    pd.DataFrame(metrics_rows).to_csv(metric_dir / _tagged_output_name("segment_barlow_ensemble_metrics_all.csv", output_tag=args.output_tag), index=False)
    pd.DataFrame(threshold_rows).to_csv(metric_dir / _tagged_output_name("segment_barlow_ensemble_threshold_summary.csv", output_tag=args.output_tag), index=False)
    summary_frame.to_csv(metric_dir / _tagged_output_name("segment_barlow_model_selection_summary.csv", output_tag=args.output_tag), index=False)
    if error_frequency_frames:
        errors_all = pd.concat(error_frequency_frames, ignore_index=True)
        errors_all.to_csv(metric_dir / _tagged_output_name("segment_barlow_subject_error_frequency.csv", output_tag=args.output_tag), index=False)
    if sub_focus_frames:
        sub_focus = pd.concat(sub_focus_frames, ignore_index=True)
        sub_focus.to_csv(metric_dir / _tagged_output_name("sub09_sub14_model_scores_summary.csv", output_tag=args.output_tag), index=False)

    exploratory = pd.DataFrame(
        [
            {
                "analysis": "oof_weighted_psd_wpli",
                "status": "disabled",
                "reason": "No strict inner/OOF weight-selection predictions are available; equal-weight ensembles are reported as primary.",
            }
        ]
    )
    exploratory.to_csv(metric_dir / _tagged_output_name("segment_barlow_exploratory_disabled.csv", output_tag=args.output_tag), index=False)
    print(f"Wrote model selection summary: {metric_dir / _tagged_output_name('segment_barlow_model_selection_summary.csv', output_tag=args.output_tag)}")


def _load_seed_frames(roots: list[Path], seeds: list[int], *, model_kind: str) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        path = _find_seed_prediction(roots, seed=seed, model_kind=model_kind)
        frame = load_required_prediction_csv(path, model_name=model_kind)
        frame["seed"] = seed
        frames[seed] = frame
    return frames


def _find_seed_prediction(roots: list[Path], *, seed: int, model_kind: str) -> Path:
    all_matches: list[Path] = []
    for root in roots:
        matched = _find_seed_prediction_in_root(root, seed=seed, model_kind=model_kind)
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            joined = "\n".join(str(path) for path in matched[:10])
            raise FileNotFoundError(
                f"Expected exactly one {model_kind} prediction CSV for seed {seed} under {root}, "
                f"found {len(matched)}.\n{joined}"
            )
        all_matches.extend(matched)
    searched = "\n".join(str(root) for root in roots)
    raise FileNotFoundError(
        f"Expected exactly one {model_kind} prediction CSV for seed {seed}; searched:\n{searched}"
    )


def _find_seed_prediction_in_root(root: Path, *, seed: int, model_kind: str) -> list[Path]:
    search_dirs = [path for path in (root, root / "predictions", root / "results" / "predictions") if path.exists()]
    paths: list[Path] = []
    for search_dir in search_dirs:
        paths.extend(search_dir.glob("*.csv"))
    candidates = [
        path
        for path in paths
        if path.name.startswith("dl_loso_predictions_")
        and f"seed{seed}_" in path.name
        and "seedensemble" not in path.name
    ]
    matched: list[Path] = []
    for path in candidates:
        name = path.name
        if model_kind == "no_ssl":
            if "no_ssl_patient_psd_fc_wpli" in name:
                matched.append(path)
            continue
        if "segssl_barlow" not in name or "vicreg" in name:
            continue
        try:
            header = pd.read_csv(path, nrows=1)
        except pd.errors.EmptyDataError:
            continue
        feature_kind = str(header.get("segment_ssl_feature_kind", pd.Series([""])).iloc[0])
        objective = str(header.get("segment_ssl_objective", pd.Series([""])).iloc[0])
        if objective and objective != "barlow":
            continue
        if model_kind == "psd" and feature_kind == "psd":
            matched.append(path)
        elif model_kind == "wpli" and feature_kind == "fc-wpli":
            matched.append(path)
    return matched


def _annotate_model_frame(frame: pd.DataFrame, *, model_group: str, seed: int) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["model_group"] = model_group
    annotated["ensemble_members"] = model_group
    annotated["ensemble_weighting"] = "single_model"
    annotated["seed"] = seed
    return annotated


def _with_fixed_prediction(frame: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    fixed = frame.copy()
    fixed["threshold"] = threshold
    fixed["threshold_method"] = "fixed_0.5"
    fixed["y_pred"] = (fixed["y_score"].to_numpy(dtype=float) >= threshold).astype(int)
    return fixed


def _summarize_seed_metric_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]
    frame = pd.DataFrame(rows)
    summary: dict[str, object] = {}
    for metric in metrics:
        values = frame[metric].astype(float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
        summary[f"{metric}_min"] = float(values.min())
        summary[f"{metric}_max"] = float(values.max())
    return summary


def _ensemble_metric_filename(model_group: str) -> str:
    if model_group == "psd_wpli_segbarlow_equal_weight":
        return "ensemble_metrics_psd_wpli_segbarlow_equal_weight.csv"
    if model_group == "no_ssl_wpli_equal_weight":
        return "ensemble_metrics_no_ssl_wpli_equal_weight.csv"
    if model_group == "no_ssl_psd_wpli_equal_weight":
        return "ensemble_metrics_no_ssl_psd_wpli_equal_weight.csv"
    return f"ensemble_metrics_{model_group}.csv"


def _normalize_threshold_methods(methods: list[str]) -> list[str]:
    normalized = []
    for method in methods:
        resolved = "leave_one_seed_out_oof" if method == "oof" else method
        if resolved not in normalized:
            normalized.append(resolved)
    return normalized


def _result_roots(values: list[str] | None, output_root: Path, *, default_export: str | None = None) -> list[Path]:
    if values:
        return [Path(value) for value in values]
    roots = [output_root]
    if default_export is not None:
        export_root = PROJECT_ROOT / "docs" / "experiment_exports" / default_export
        if export_root.exists():
            roots.append(export_root)
    return roots


def _tagged_output_name(filename: str, *, output_tag: str | None) -> str:
    token = _output_tag_token(output_tag)
    if token is None:
        return filename
    if filename == "sub09_sub14_model_scores_summary.csv" and token == "10seed":
        return "sub09_sub14_10seed_model_scores_summary.csv"
    path = Path(filename)
    stem = path.stem
    if stem.startswith("segment_barlow_"):
        stem = stem.replace("segment_barlow_", f"segment_barlow_{token}_", 1)
    else:
        stem = f"{stem}_{token}"
    return f"{stem}{path.suffix}"


def _output_tag_token(output_tag: str | None) -> str | None:
    if not output_tag or output_tag == "segment_barlow_model_selection":
        return None
    if output_tag in {"10seed", "10seed_locked"}:
        return "10seed"
    return "".join(character if character.isalnum() else "_" for character in output_tag).strip("_")


if __name__ == "__main__":
    main()
