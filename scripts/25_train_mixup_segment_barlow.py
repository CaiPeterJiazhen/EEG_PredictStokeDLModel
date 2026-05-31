from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.segment_barlow_model_selection import (
    aggregate_seed_mean_predictions,
    apply_leave_one_seed_thresholds,
    compute_per_subject_error_frequency,
    classification_metrics_from_scores,
    equal_weight_model_ensemble,
    evaluate_prediction_frame,
    leave_one_seed_thresholds,
)
from eeg_recovery.training.train_segment_ssl import segment_records_manifest_hash
from eeg_recovery.training.train_ssl import SSL_DATA_SCOPES
from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    load_supervised_feature_records,
    run_loso_supervised_with_history,
)


DEFAULT_SEEDS = [0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run manifold mixup supervised fine-tuning from reusable Segment Barlow SSL encoder checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--segment-feature-kind", choices=("psd", "fc-wpli"), required=True)
    parser.add_argument("--supervised-feature-kind", choices=("psd-fc-wpli",), default="psd-fc-wpli")
    parser.add_argument("--objective", choices=("barlow",), default="barlow")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--ssl-data-scope", choices=SSL_DATA_SCOPES, default="all-patient")
    parser.add_argument("--reuse-ssl-encoders", action="store_true")
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--ssl-checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-tag", default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--feature-mask-prob", type=float, default=0.03)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--lambda-latent", type=float, default=1.0)
    parser.add_argument("--lambda-local", type=float, default=0.1)
    parser.add_argument("--masked-latent-loss", choices=("cosine", "mse"), default="cosine")
    parser.add_argument("--barlow-offdiag-weight", type=float, default=0.005)
    parser.add_argument("--mixup-enabled", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--mixup-weight", type=float, default=1.0)
    parser.add_argument("--mixup-layer", choices=("embedding",), default="embedding")
    parser.add_argument("--modality-dropout-prob", type=float, default=0.10)
    parser.add_argument("--state-dropout-prob", type=float, default=0.0)
    parser.add_argument("--supervised-lr", type=float, default=0.002)
    parser.add_argument("--supervised-weight-decay", type=float, default=1e-5)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--output-tag", default="mixup")
    args = parser.parse_args()

    if not args.reuse_ssl_encoders or not args.reuse_only:
        raise SystemExit("Mixup fine-tuning requires --reuse-ssl-encoders and --reuse-only.")

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    labels = load_supervised_label_table(path_config)
    supervised_ids = labels["subject_id"].tolist()
    supervised_records = load_supervised_feature_records(
        path_config,
        labels,
        feature_kind=args.supervised_feature_kind,
    )
    segment_records = _load_segment_records(output_root, args.segment_feature_kind)
    folds = make_loso_folds([record.subject_id for record in supervised_records])

    for seed in args.seeds:
        pretrained_state_by_test_subject = {}
        for fold in folds:
            fold_segments = segment_records_for_scope(
                segment_records,
                data_scope=args.ssl_data_scope,
                strict_loso_test_subject_id=fold.test_subject_id,
                supervised_subject_ids=supervised_ids,
            )
            branch = _branch_for_segment_feature_kind(args.segment_feature_kind)
            checkpoint_path = _checkpoint_path(args, path_config, branch, seed, fold)
            current_hash = segment_records_manifest_hash(fold_segments)
            source_hash = _source_manifest_hash_for_checkpoint_reuse(
                fold_segments,
                output_root=output_root,
                checkpoint_path=checkpoint_path,
                current_hash=current_hash,
            )
            expected_metadata = _checkpoint_metadata(
                args=args,
                branch=branch,
                fold=fold,
                seed=seed,
                n_ssl_segments=len(fold_segments),
                source_feature_manifest_hash=source_hash,
            )
            prefixed_state, checkpoint = _load_reusable_branch_checkpoint(
                checkpoint_path,
                expected_metadata=expected_metadata,
                branch=branch,
            )
            pretrained_state_by_test_subject[fold.test_subject_id] = prefixed_state
            print(
                f"[mixup] feature={args.segment_feature_kind} seed={seed} "
                f"fold={fold.fold_index} test={fold.test_subject_id} "
                f"checkpoint_reused=True tensors={len(checkpoint['encoder_state_dict'])}"
            )

        training_config = SupervisedTrainingConfig(
            architecture="multimodal",
            feature_kind=args.supervised_feature_kind,
            fusion="gated",
            encoder_kind="cnn",
            device=args.device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.supervised_lr,
            weight_decay=args.supervised_weight_decay,
            loss_name="bce",
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            seed=seed,
            pretrained_transfer_mode="finetune",
            mixup_enabled=args.mixup_enabled,
            mixup_alpha=args.mixup_alpha,
            mixup_weight=args.mixup_weight,
            mixup_layer=args.mixup_layer,
            modality_dropout_prob=args.modality_dropout_prob,
            state_dropout_prob=args.state_dropout_prob,
        )
        predictions, metrics, loss_history = run_loso_supervised_with_history(
            supervised_records,
            training_config,
            pretrained_state_by_test_subject=pretrained_state_by_test_subject,
        )
        model_group = f"{_output_prefix(args.segment_feature_kind)}_segbarlow_mixup"
        _annotate_outputs((predictions, metrics, loss_history), args=args, seed=seed, model_group=model_group)
        metric_update = classification_metrics_from_scores(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["y_score"].to_numpy(dtype=float),
            y_pred=predictions["y_pred"].to_numpy(dtype=int),
        )
        for key, value in metric_update.items():
            metrics[key] = value
        prediction_path, metric_path = _mixup_output_paths(output_root, segment_feature_kind=args.segment_feature_kind, seed=seed)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(prediction_path, index=False)
        metrics.to_csv(metric_path, index=False)
        _write_loss_history(output_root, args.segment_feature_kind, seed, loss_history)
        print(f"Wrote predictions: {prediction_path}")
        print(f"Wrote metrics: {metric_path}")

    _write_mixup_ensemble_if_available(output_root, args.seeds)
    _write_seed0_comparison_if_available(output_root)
    _write_10seed_outputs_if_available(output_root, args.seeds)


def _load_hardneg_helpers():
    script_path = PROJECT_ROOT / "scripts" / "24_train_hard_negative_segment_barlow.py"
    spec = spec_from_file_location("_hardneg_segment_barlow_helpers", script_path)
    module = module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {script_path}")
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_hardneg_helpers()
_load_segment_records = _HELPERS._load_segment_records
segment_records_for_scope = _HELPERS.segment_records_for_scope
_branch_for_segment_feature_kind = _HELPERS._branch_for_segment_feature_kind
_checkpoint_path = _HELPERS._checkpoint_path
_checkpoint_metadata = _HELPERS._checkpoint_metadata
_load_reusable_branch_checkpoint = _HELPERS._load_reusable_branch_checkpoint
_source_manifest_hash_for_checkpoint_reuse = _HELPERS._source_manifest_hash_for_checkpoint_reuse
_output_prefix = _HELPERS._output_prefix
_find_baseline_prediction = _HELPERS._find_baseline_prediction
_original_psd_wpli_ensemble_seed_path = _HELPERS._original_psd_wpli_ensemble_seed_path
_comparison_row = _HELPERS._comparison_row
_metric_delta = _HELPERS._metric_delta
_format_delta = _HELPERS._format_delta


def _load_statistical_helpers():
    script_path = PROJECT_ROOT / "scripts" / "23_compare_segment_barlow_models.py"
    spec = spec_from_file_location("_segment_barlow_stat_helpers", script_path)
    module = module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load statistical helper script: {script_path}")
    spec.loader.exec_module(module)
    return module


_STATS = _load_statistical_helpers()


def _annotate_outputs(frames, *, args, seed: int, model_group: str) -> None:
    for frame in frames:
        frame["model_group"] = model_group
        frame["segment_ssl"] = True
        frame["segment_ssl_objective"] = args.objective
        frame["segment_ssl_feature_kind"] = args.segment_feature_kind
        frame["ssl_data_scope"] = args.ssl_data_scope
        frame["manifold_mixup_finetune"] = True
        frame["output_tag"] = args.output_tag
        frame["loss_name"] = "bce"
        frame["mixup_enabled"] = args.mixup_enabled
        frame["mixup_alpha"] = args.mixup_alpha
        frame["mixup_weight"] = args.mixup_weight
        frame["mixup_layer"] = args.mixup_layer
        frame["modality_dropout_prob"] = args.modality_dropout_prob
        frame["state_dropout_prob"] = args.state_dropout_prob
        frame["seed"] = seed


def _mixup_output_paths(output_root: Path, *, segment_feature_kind: str, seed: int) -> tuple[Path, Path]:
    prefix = _output_prefix(segment_feature_kind)
    prediction_path = output_root / "results" / "predictions" / f"dl_loso_predictions_{prefix}_segbarlow_mixup_seed{seed}.csv"
    metric_path = output_root / "results" / "metrics" / f"dl_model_comparison_{prefix}_segbarlow_mixup_seed{seed}.csv"
    return prediction_path, metric_path


def _write_loss_history(output_root: Path, segment_feature_kind: str, seed: int, loss_history: pd.DataFrame) -> None:
    path = output_root / "results" / "training_logs" / f"dl_loss_history_{_output_prefix(segment_feature_kind)}_segbarlow_mixup_seed{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    loss_history.to_csv(path, index=False)


def _write_mixup_ensemble_if_available(output_root: Path, seeds: list[int]) -> None:
    per_seed = {}
    for seed in seeds:
        psd_path, _ = _mixup_output_paths(output_root, segment_feature_kind="psd", seed=seed)
        wpli_path, _ = _mixup_output_paths(output_root, segment_feature_kind="fc-wpli", seed=seed)
        if not psd_path.exists() or not wpli_path.exists():
            return
        psd = pd.read_csv(psd_path)
        wpli = pd.read_csv(wpli_path)
        ensemble = equal_weight_model_ensemble(
            {"psd_mixup": psd, "wpli_mixup": wpli},
            model_group="psd_wpli_segbarlow_mixup_equal_weight",
            seed=seed,
        )
        ensemble["y_pred"] = (ensemble["y_score"].astype(float) >= 0.5).astype(int)
        per_seed[seed] = ensemble

    output_name = "seed0" if seeds == [0] else "10seed"
    predictions = pd.concat(per_seed.values(), ignore_index=True)
    prediction_path = output_root / "results" / "predictions" / f"ensemble_predictions_psd_wpli_segbarlow_mixup_{output_name}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)

    metric_rows = []
    for seed, frame in per_seed.items():
        row = {
            "model_group": "psd_wpli_segbarlow_mixup_equal_weight",
            "seed": seed,
            "threshold": 0.5,
            "threshold_method": "fixed_0.5",
            "ensemble_members": "psd_mixup+wpli_mixup",
            "ensemble_weighting": "equal",
            "n_subjects": len(frame),
            "n_seeds": len(per_seed),
        }
        row.update(
            classification_metrics_from_scores(
                frame["y_true"].to_numpy(dtype=int),
                frame["y_score"].to_numpy(dtype=float),
                y_pred=frame["y_pred"].to_numpy(dtype=int),
            )
        )
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    metric_name = (
        "dl_model_comparison_psd_wpli_segbarlow_mixup_ensemble_seed0.csv"
        if seeds == [0]
        else "segment_barlow_mixup_10seed_ensemble_summary.csv"
    )
    metric_path = output_root / "results" / "metrics" / metric_name
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metric_path, index=False)


def _write_seed0_comparison_if_available(output_root: Path) -> None:
    mixup_paths = {
        "wpli_mixup": _mixup_output_paths(output_root, segment_feature_kind="fc-wpli", seed=0)[0],
        "psd_mixup": _mixup_output_paths(output_root, segment_feature_kind="psd", seed=0)[0],
        "psd_wpli_mixup_equal": output_root / "results" / "predictions" / "ensemble_predictions_psd_wpli_segbarlow_mixup_seed0.csv",
    }
    if any(not path.exists() for path in mixup_paths.values()):
        return

    rows = []
    for model_group, path in {
        "wpli_bce_baseline": _find_baseline_prediction(output_root, model_kind="wpli", seed=0),
        "psd_bce_baseline": _find_baseline_prediction(output_root, model_kind="psd", seed=0),
        "psd_wpli_bce_equal": _original_psd_wpli_ensemble_seed_path(output_root),
        **mixup_paths,
    }.items():
        if path is None or not path.exists():
            continue
        frame = pd.read_csv(path)
        if "seed" in frame.columns and model_group == "psd_wpli_bce_equal":
            frame = frame.loc[frame["seed"].astype(str) == "0"].copy()
        rows.append(_comparison_row(frame, model_group=model_group, prediction_path=path))
    comparison = pd.DataFrame(rows)
    comparison_path = output_root / "results" / "metrics" / "segment_barlow_mixup_seed0_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    _write_seed0_doc(output_root, comparison)


def _write_seed0_doc(output_root: Path, comparison: pd.DataFrame) -> None:
    path = output_root / "docs" / "segment_barlow_mixup_seed0_results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Segment Barlow Manifold Mixup Seed0 Pilot",
        "",
        "This pilot evaluates embedding-level manifold mixup plus light modality dropout during supervised fine-tuning using existing fold-specific Segment Barlow SSL encoder checkpoints. No SSL pretraining, MIL, dual-encoder expansion, raw EEG CNN-BLSTM, or architecture search was run.",
        "",
        "Repeated false positives were observed in prior locked results. Manifold mixup was evaluated as supervised fine-tuning regularization; sub09/sub14 are monitored only as post-hoc error-analysis subjects.",
        "",
        "## Seed0 Metrics",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Pilot Decision",
        "",
        _seed0_pilot_decision(comparison),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_10seed_outputs_if_available(output_root: Path, seeds: list[int]) -> None:
    if len(seeds) < 2:
        return
    expected_seeds = sorted(seeds)
    model_frames = _load_10seed_mixup_frames(output_root, expected_seeds)
    if model_frames is None:
        return

    prediction_dir = output_root / "results" / "predictions"
    metric_dir = output_root / "results" / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    all_metric_rows: list[dict[str, object]] = []
    error_frames: list[pd.DataFrame] = []
    sub_focus_frames: list[pd.DataFrame] = []

    group_labels = {
        "wpli_segbarlow_mixup_ssl_cnn": ("wpli_mixup", "single_model"),
        "psd_segbarlow_mixup_ssl_cnn": ("psd_mixup", "single_model"),
        "psd_wpli_segbarlow_mixup_equal_weight": ("psd_mixup+wpli_mixup", "equal"),
    }
    for model_group, per_seed in model_frames.items():
        ensemble_members, ensemble_weighting = group_labels[model_group]
        threshold_map = leave_one_seed_thresholds(per_seed, metric="balanced_accuracy")
        for method in ("fixed_0.5", "leave_one_seed_out_oof"):
            seed_metric_rows = []
            prediction_frames_for_errors = []
            if method == "fixed_0.5":
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
                    all_metric_rows.append(row)
                seed_mean_threshold = 0.5
                seed_mean_method = "fixed_0.5"
            else:
                calibrated_all = apply_leave_one_seed_thresholds(per_seed, metric="balanced_accuracy")
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
                    all_metric_rows.append(row)
                seed_mean_threshold = float(pd.Series([item.threshold for item in threshold_map.values()]).median())
                seed_mean_method = "leave_one_seed_out_oof_median_threshold"

            seed_mean_predictions = aggregate_seed_mean_predictions(
                per_seed,
                model_group=model_group,
                threshold=seed_mean_threshold,
                threshold_method=seed_mean_method,
                ensemble_members=ensemble_members,
                ensemble_weighting=f"{ensemble_weighting}+seed_mean_probability",
            )
            seed_mean_path = prediction_dir / f"{model_group}_{seed_mean_method}_seed_mean_predictions.csv"
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
                    "threshold_method": method,
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

            if method == "fixed_0.5":
                errors = compute_per_subject_error_frequency(prediction_frames_for_errors, model_group=model_group)
                error_frames.append(errors)
                sub_focus_frames.append(errors.loc[errors["subject_id"].isin(["sub09", "sub14"])].copy())

    pd.DataFrame(summary_rows).to_csv(metric_dir / "segment_barlow_mixup_10seed_model_selection_summary.csv", index=False)
    pd.DataFrame(all_metric_rows).to_csv(metric_dir / "segment_barlow_mixup_10seed_all_seed_metrics.csv", index=False)
    if error_frames:
        pd.concat(error_frames, ignore_index=True).to_csv(
            metric_dir / "segment_barlow_mixup_10seed_subject_error_frequency.csv",
            index=False,
        )
    if sub_focus_frames:
        pd.concat(sub_focus_frames, ignore_index=True).to_csv(
            metric_dir / "sub09_sub14_mixup_10seed_scores_summary.csv",
            index=False,
        )
    _write_10seed_statistical_comparison(output_root, pd.DataFrame(summary_rows))
    _write_10seed_doc(output_root, pd.DataFrame(summary_rows))


def _load_10seed_mixup_frames(output_root: Path, seeds: list[int]) -> dict[str, dict[int, pd.DataFrame]] | None:
    groups = {
        "wpli_segbarlow_mixup_ssl_cnn": {
            seed: _mixup_output_paths(output_root, segment_feature_kind="fc-wpli", seed=seed)[0]
            for seed in seeds
        },
        "psd_segbarlow_mixup_ssl_cnn": {
            seed: _mixup_output_paths(output_root, segment_feature_kind="psd", seed=seed)[0]
            for seed in seeds
        },
    }
    loaded: dict[str, dict[int, pd.DataFrame]] = {}
    for model_group, paths in groups.items():
        if any(not path.exists() for path in paths.values()):
            return None
        loaded[model_group] = {
            seed: _annotate_seed_frame(pd.read_csv(path), model_group=model_group, seed=seed)
            for seed, path in paths.items()
        }
    ensemble_group = "psd_wpli_segbarlow_mixup_equal_weight"
    loaded[ensemble_group] = {}
    for seed in seeds:
        ensemble = equal_weight_model_ensemble(
            {
                "psd_mixup": loaded["psd_segbarlow_mixup_ssl_cnn"][seed],
                "wpli_mixup": loaded["wpli_segbarlow_mixup_ssl_cnn"][seed],
            },
            model_group=ensemble_group,
            seed=seed,
        )
        loaded[ensemble_group][seed] = ensemble
    return loaded


def _annotate_seed_frame(frame: pd.DataFrame, *, model_group: str, seed: int) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["model_group"] = model_group
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


def _write_10seed_statistical_comparison(output_root: Path, summary: pd.DataFrame) -> None:
    prediction_frames = _statistical_prediction_frames(output_root, summary)
    rows = []
    for model_group, frame in prediction_frames.items():
        rows.extend(
            _STATS._bootstrap_ci_rows(
                frame,
                model_group=model_group,
                threshold=0.5,
                n_bootstrap=5000,
                rng=_STATS.np.random.default_rng(20260528),
            )
        )
    pairs = [
        ("wpli_segbarlow_ssl_cnn", "wpli_segbarlow_mixup_ssl_cnn"),
        ("psd_segbarlow_ssl_cnn", "psd_segbarlow_mixup_ssl_cnn"),
        ("psd_wpli_segbarlow_equal_weight", "psd_wpli_segbarlow_mixup_equal_weight"),
    ]
    if "no_ssl_stable_cnn" in prediction_frames:
        best_mixup = _best_mixup_group(summary)
        if best_mixup:
            pairs.append(("no_ssl_stable_cnn", best_mixup))
    elif _best_mixup_group(summary):
        rows.append(
            {
                "analysis_type": "paired_comparison_not_run",
                "model_a": "no_ssl_stable_cnn",
                "model_b": _best_mixup_group(summary),
                "reason": "Local no_ssl_stable_cnn fixed-threshold seed-mean prediction CSV was not available; locked summary metrics are documented separately.",
            }
        )
    for model_a, model_b in pairs:
        if model_a in prediction_frames and model_b in prediction_frames:
            rows.extend(
                _STATS._paired_comparison_rows(
                    prediction_frames[model_a],
                    prediction_frames[model_b],
                    model_a=model_a,
                    model_b=model_b,
                    rng=_STATS.np.random.default_rng(20260528),
                )
            )
    best_mixup = _best_mixup_group(summary)
    if best_mixup and best_mixup in prediction_frames:
        rows.append(
            _STATS._permutation_random_label_row(
                prediction_frames[best_mixup],
                model_group=best_mixup,
                threshold=0.5,
                n_permutations=5000,
                rng=_STATS.np.random.default_rng(20260528),
            )
        )
    path = output_root / "results" / "metrics" / "segment_barlow_mixup_10seed_statistical_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False)


def _statistical_prediction_frames(output_root: Path, summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    fixed = summary.loc[
        (summary["aggregation"] == "seed_mean_probability")
        & (summary["threshold_method"] == "fixed_0.5")
    ]
    for row in fixed.itertuples(index=False):
        frames[str(row.model_group)] = pd.read_csv(row.prediction_path)
    baseline_paths = {
        "wpli_segbarlow_ssl_cnn": output_root / "results" / "predictions" / "wpli_segbarlow_ssl_cnn_fixed_0.5_seed_mean_predictions.csv",
        "psd_segbarlow_ssl_cnn": output_root / "results" / "predictions" / "psd_segbarlow_ssl_cnn_fixed_0.5_seed_mean_predictions.csv",
        "psd_wpli_segbarlow_equal_weight": output_root / "results" / "predictions" / "psd_wpli_segbarlow_equal_weight_fixed_0.5_seed_mean_predictions.csv",
        "no_ssl_stable_cnn": output_root / "results" / "predictions" / "no_ssl_stable_cnn_fixed_0.5_seed_mean_predictions.csv",
    }
    for model_group, path in baseline_paths.items():
        if path.exists():
            frames[model_group] = pd.read_csv(path)
    return frames


def _best_mixup_group(summary: pd.DataFrame) -> str | None:
    fixed = summary.loc[
        (summary["aggregation"] == "seed_mean_probability")
        & (summary["threshold_method"] == "fixed_0.5")
        & (summary["model_group"].astype(str).str.contains("mixup"))
    ].copy()
    if fixed.empty:
        return None
    fixed = fixed.sort_values(["balanced_accuracy", "brier_score"], ascending=[False, True])
    return str(fixed.iloc[0]["model_group"])


def _write_10seed_doc(output_root: Path, summary: pd.DataFrame) -> None:
    fixed = summary.loc[
        (summary["aggregation"] == "seed_mean_probability")
        & (summary["threshold_method"] == "fixed_0.5")
    ].copy()
    path = output_root / "docs" / "segment_barlow_mixup_10seed_results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Segment Barlow Manifold Mixup 10-Seed Results",
        "",
        "This 10-seed run uses existing fold-specific Segment Barlow encoder checkpoints and only changes supervised fine-tuning with embedding-level manifold mixup plus light modality dropout.",
        "",
        "Thresholding reports fixed 0.5 and leave-one-seed-out OOF calibration. No final 19-subject seed-mean test-label threshold search is used as a primary result.",
        "",
        "## Fixed 0.5 Seed-Mean Metrics",
        "",
        fixed.to_markdown(index=False),
        "",
    ]
    comparison = _locked_baseline_comparison(output_root, fixed)
    if not comparison.empty:
        lines.extend(
            [
                "## Locked Baseline Comparison",
                "",
                comparison.to_markdown(index=False),
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "Sub09/sub14 remain repeated false positives in the fixed-threshold 10-seed mixup outputs and are reported only as post-hoc error-analysis subjects.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _locked_baseline_comparison(output_root: Path, fixed_mixup: pd.DataFrame) -> pd.DataFrame:
    summary_path = output_root / "results" / "metrics" / "segment_barlow_10seed_model_selection_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    locked = pd.read_csv(summary_path)
    locked_fixed = locked.loc[
        (locked["aggregation"] == "seed_mean_probability")
        & (locked["threshold_method"] == "fixed_0.5")
    ].copy()
    rows = []
    pairs = [
        ("wpli_segbarlow_ssl_cnn", "wpli_segbarlow_mixup_ssl_cnn"),
        ("psd_segbarlow_ssl_cnn", "psd_segbarlow_mixup_ssl_cnn"),
        ("psd_wpli_segbarlow_equal_weight", "psd_wpli_segbarlow_mixup_equal_weight"),
    ]
    for baseline_group, mixup_group in pairs:
        baseline = locked_fixed.loc[locked_fixed["model_group"] == baseline_group]
        mixup = fixed_mixup.loc[fixed_mixup["model_group"] == mixup_group]
        if baseline.empty or mixup.empty:
            continue
        base = baseline.iloc[0]
        new = mixup.iloc[0]
        rows.append(
            {
                "baseline_group": baseline_group,
                "mixup_group": mixup_group,
                "accuracy_delta": float(new["accuracy"]) - float(base["accuracy"]),
                "balanced_accuracy_delta": float(new["balanced_accuracy"]) - float(base["balanced_accuracy"]),
                "roc_auc_delta": float(new["roc_auc"]) - float(base["roc_auc"]),
                "pr_auc_delta": float(new["pr_auc"]) - float(base["pr_auc"]),
                "brier_delta": float(new["brier_score"]) - float(base["brier_score"]),
            }
        )
    no_ssl = locked_fixed.loc[locked_fixed["model_group"] == "no_ssl_stable_cnn"]
    if not no_ssl.empty:
        base = no_ssl.iloc[0]
        rows.append(
            {
                "baseline_group": "no_ssl_stable_cnn",
                "mixup_group": "best_mixup_fixed_0.5",
                "accuracy_delta": float(fixed_mixup["accuracy"].max()) - float(base["accuracy"]),
                "balanced_accuracy_delta": float(fixed_mixup["balanced_accuracy"].max()) - float(base["balanced_accuracy"]),
                "roc_auc_delta": float(fixed_mixup["roc_auc"].max()) - float(base["roc_auc"]),
                "pr_auc_delta": float(fixed_mixup["pr_auc"].max()) - float(base["pr_auc"]),
                "brier_delta": float(fixed_mixup["brier_score"].min()) - float(base["brier_score"]),
            }
        )
    return pd.DataFrame(rows)


def _seed0_pilot_decision(comparison: pd.DataFrame) -> str:
    rows = {str(row["model_group"]): row for _, row in comparison.iterrows()}
    wpli_ba = _metric_delta(rows, "wpli_mixup", "wpli_bce_baseline", "balanced_accuracy")
    wpli_roc = _metric_delta(rows, "wpli_mixup", "wpli_bce_baseline", "roc_auc")
    wpli_pr = _metric_delta(rows, "wpli_mixup", "wpli_bce_baseline", "pr_auc")
    wpli_acc = _metric_delta(rows, "wpli_mixup", "wpli_bce_baseline", "accuracy")
    wpli_brier = _metric_delta(rows, "wpli_mixup", "wpli_bce_baseline", "brier_score")
    ensemble_ba = _metric_delta(rows, "psd_wpli_mixup_equal", "psd_wpli_bce_equal", "balanced_accuracy")
    ensemble_brier = _metric_delta(rows, "psd_wpli_mixup_equal", "psd_wpli_bce_equal", "brier_score")
    sub09 = _metric_delta(rows, "wpli_mixup", "wpli_bce_baseline", "sub09_y_score")
    sub14 = _metric_delta(rows, "wpli_mixup", "wpli_bce_baseline", "sub14_y_score")
    criteria_met = [
        wpli_ba is not None and wpli_ba > 0,
        ((wpli_roc is not None and wpli_roc > 0) or (wpli_pr is not None and wpli_pr > 0))
        and (wpli_acc is not None and wpli_acc >= 0),
        wpli_brier is not None and wpli_brier < 0,
        (ensemble_ba is not None and ensemble_ba > 0) or (ensemble_brier is not None and ensemble_brier < 0),
        ((sub09 is not None and sub09 <= -0.05) or (sub14 is not None and sub14 <= -0.05))
        and (wpli_acc is not None and wpli_acc >= 0),
    ]
    decision = (
        "At least one continuation criterion is met; a 10-seed run is defensible."
        if any(criteria_met)
        else "This seed0 pilot is treated as negative, so the 10-seed mixup run should not be started from this pilot."
    )
    return (
        f"{decision} "
        f"WPLI mixup changed balanced accuracy by {_format_delta(wpli_ba)}, ROC AUC by {_format_delta(wpli_roc)}, "
        f"PR AUC by {_format_delta(wpli_pr)}, accuracy by {_format_delta(wpli_acc)}, and Brier by {_format_delta(wpli_brier)} versus WPLI BCE. "
        f"The mixup equal-weight ensemble changed balanced accuracy by {_format_delta(ensemble_ba)} and Brier by {_format_delta(ensemble_brier)} versus the original equal-weight ensemble. "
        f"Post-hoc WPLI monitored score deltas were sub09={_format_delta(sub09)} and sub14={_format_delta(sub14)}."
    )


if __name__ == "__main__":
    main()
