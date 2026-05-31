from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.segment_ssl_dataset import load_segment_ssl_records, segment_records_for_scope
from eeg_recovery.training.ssl_checkpointing import (
    load_ssl_encoder_checkpoint,
    make_ssl_checkpoint_name,
    merge_branch_pretrained_states,
    prefix_branch_encoder_state_dict,
    save_ssl_encoder_checkpoint,
)
from eeg_recovery.training.train_feature_ssl import aggregate_seed_ensemble_predictions
from eeg_recovery.training.train_segment_ssl import (
    SegmentSSLTrainingConfig,
    extract_branch_encoder_state,
    run_segment_ssl_pretraining,
    segment_records_manifest_hash,
    write_segment_ssl_transfer_outputs,
)
from eeg_recovery.training.train_ssl import SSL_DATA_SCOPES
from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    load_supervised_feature_records,
    run_loso_supervised_with_history,
)


DEFAULT_SEEDS = [0, 1, 2, 3, 7, 13]
DUAL_EXPERIMENTS = {
    "dual-finetune": {"experiment_name": "D_dual_segment_barlow_finetune", "transfer_mode": "finetune", "lr_key": "main"},
    "dual-freeze": {"experiment_name": "E_dual_segment_barlow_freeze_encoder", "transfer_mode": "freeze-encoder", "lr_key": "main"},
    "dual-low-lr": {"experiment_name": "F_dual_segment_barlow_lower_lr", "transfer_mode": "finetune", "lr_key": "low"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run remaining dual PSD+wPLI Segment Barlow transfer experiments D/E/F.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--experiment-set",
        choices=("remaining", "dual-finetune", "dual-freeze", "dual-low-lr", "all-dual"),
        default="remaining",
    )
    parser.add_argument("--ssl-data-scope", choices=SSL_DATA_SCOPES, default="all-patient")
    parser.add_argument("--psd-checkpoint-dir", default=None)
    parser.add_argument("--wpli-checkpoint-dir", default=None)
    parser.add_argument("--ssl-checkpoint-dir", default=None)
    parser.add_argument("--save-ssl-encoders", action="store_true")
    parser.add_argument("--reuse-ssl-encoders", action="store_true")
    parser.add_argument("--force-retrain-ssl", action="store_true")
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--barlow-offdiag-weight", type=float, default=0.005)
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--supervised-lr-main", type=float, default=0.002)
    parser.add_argument("--supervised-lr-low", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--feature-mask-prob", type=float, default=0.03)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--lambda-latent", type=float, default=1.0)
    parser.add_argument("--lambda-local", type=float, default=0.1)
    parser.add_argument("--masked-latent-loss", choices=("cosine", "mse"), default="cosine")
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--checkpoint-tag", default=None)
    parser.add_argument("--allow-mismatched-ssl-seeds", action="store_true")
    parser.add_argument("--limit-segments", type=int, default=None, help="Optional cap for smoke tests only.")
    args = parser.parse_args()
    if args.reuse_only and not args.reuse_ssl_encoders:
        raise SystemExit("--reuse-only requires --reuse-ssl-encoders.")

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    labels = load_supervised_label_table(path_config)
    supervised_ids = labels["subject_id"].tolist()
    supervised_records = load_supervised_feature_records(path_config, labels, feature_kind="psd-fc-wpli")
    folds = make_loso_folds([record.subject_id for record in supervised_records])
    segment_root = output_root / "data" / "features" / "segment_level"
    psd_records = load_segment_ssl_records(segment_root / "psd", branches=("psd",))
    wpli_records = load_segment_ssl_records(segment_root / "fc", branches=("wpli",))
    experiments = _selected_experiments(args.experiment_set)

    transfer_summary_frames: list[pd.DataFrame] = []
    dual_error_frames: list[pd.DataFrame] = []
    dual_predictions_by_experiment: dict[str, list[pd.DataFrame]] = {name: [] for name in experiments}
    dual_metrics_by_experiment: dict[str, list[pd.DataFrame]] = {name: [] for name in experiments}

    for seed in args.seeds:
        pretrained_state_by_test_subject: dict[str, dict] = {}
        ssl_history_frames: list[pd.DataFrame] = []
        for fold in folds:
            psd_segments = segment_records_for_scope(
                psd_records,
                data_scope=args.ssl_data_scope,
                strict_loso_test_subject_id=fold.test_subject_id,
                supervised_subject_ids=supervised_ids,
            )
            wpli_segments = segment_records_for_scope(
                wpli_records,
                data_scope=args.ssl_data_scope,
                strict_loso_test_subject_id=fold.test_subject_id,
                supervised_subject_ids=supervised_ids,
            )
            if args.limit_segments is not None:
                psd_segments = psd_segments[: args.limit_segments]
                wpli_segments = wpli_segments[: args.limit_segments]
            psd_manifest_hash = segment_records_manifest_hash(psd_segments)
            wpli_manifest_hash = segment_records_manifest_hash(wpli_segments)
            psd_checkpoint, psd_history = _run_or_load_branch_checkpoint(
                args=args,
                path_config=path_config,
                branch="psd",
                feature_kind="psd",
                fold_segments=psd_segments,
                fold=fold,
                seed=seed,
                source_feature_manifest_hash=psd_manifest_hash,
            )
            wpli_checkpoint, wpli_history = _run_or_load_branch_checkpoint(
                args=args,
                path_config=path_config,
                branch="wpli",
                feature_kind="fc-wpli",
                fold_segments=wpli_segments,
                fold=fold,
                seed=seed,
                source_feature_manifest_hash=wpli_manifest_hash,
            )
            merged_state, _ = merge_branch_pretrained_states(
                psd_checkpoint,
                wpli_checkpoint,
                allow_mismatched_ssl_seeds=args.allow_mismatched_ssl_seeds,
            )
            pretrained_state_by_test_subject[fold.test_subject_id] = merged_state
            for history, branch in ((psd_history, "psd"), (wpli_history, "wpli")):
                history = history.copy()
                history.insert(0, "fold_index", fold.fold_index)
                history.insert(1, "test_subject_id", fold.test_subject_id)
                history["branch"] = branch
                history["ssl_data_scope"] = args.ssl_data_scope
                history["base_seed"] = seed
                ssl_history_frames.append(history)
            print(
                f"[dual-segbarlow] seed={seed} fold={fold.fold_index} test={fold.test_subject_id} "
                f"psd_segments={len(psd_segments)} wpli_segments={len(wpli_segments)}"
            )

        ssl_history_all = pd.concat(ssl_history_frames, ignore_index=True)
        for experiment_key in experiments:
            spec = DUAL_EXPERIMENTS[experiment_key]
            supervised_lr = args.supervised_lr_main if spec["lr_key"] == "main" else args.supervised_lr_low
            run_name = _dual_run_name(experiment_key, seed, supervised_lr)
            if args.output_tag:
                run_name = f"{run_name}_{args.output_tag}"
            supervised_config = SupervisedTrainingConfig(
                architecture="multimodal",
                feature_kind="psd-fc-wpli",
                fusion="gated",
                encoder_kind="cnn",
                device=args.device,
                epochs=args.supervised_epochs,
                patience=args.patience,
                lr=supervised_lr,
                weight_decay=args.weight_decay,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
                seed=seed,
                pretrained_transfer_mode=str(spec["transfer_mode"]),
            )
            predictions, metrics, loss_history = run_loso_supervised_with_history(
                supervised_records,
                supervised_config,
                pretrained_state_by_test_subject=pretrained_state_by_test_subject,
            )
            _annotate_dual_frames(
                (predictions, metrics, loss_history),
                run_name=run_name,
                experiment_name=str(spec["experiment_name"]),
                seed=seed,
                transfer_mode=str(spec["transfer_mode"]),
                supervised_lr=supervised_lr,
                args=args,
            )
            paths = write_segment_ssl_transfer_outputs(
                output_root=path_config.output_root,
                run_name=run_name,
                predictions=predictions,
                metrics=metrics,
                ssl_history=ssl_history_all,
                supervised_loss_history=loss_history,
            )
            print(f"Wrote predictions: {paths['predictions']}")
            print(f"Wrote metrics: {paths['metrics']}")
            transfer_summary_frames.append(metrics)
            dual_predictions_by_experiment[experiment_key].append(predictions)
            dual_metrics_by_experiment[experiment_key].append(metrics)

    for experiment_key, prediction_frames in dual_predictions_by_experiment.items():
        if len(prediction_frames) < 2:
            continue
        spec = DUAL_EXPERIMENTS[experiment_key]
        supervised_lr = args.supervised_lr_main if spec["lr_key"] == "main" else args.supervised_lr_low
        ensemble_predictions, ensemble_metrics = aggregate_seed_ensemble_predictions(prediction_frames)
        ensemble_run_name = _dual_run_name(experiment_key, f"ensemble{len(prediction_frames)}", supervised_lr)
        if args.output_tag:
            ensemble_run_name = f"{ensemble_run_name}_{args.output_tag}"
        _annotate_dual_frames(
            (ensemble_predictions, ensemble_metrics),
            run_name=ensemble_run_name,
            experiment_name=str(spec["experiment_name"]),
            seed="ensemble",
            transfer_mode=str(spec["transfer_mode"]),
            supervised_lr=supervised_lr,
            args=args,
        )
        _write_prediction_metric_pair(output_root, ensemble_run_name, ensemble_predictions, ensemble_metrics)
        transfer_summary_frames.append(ensemble_metrics)

    if transfer_summary_frames:
        transfer_summary = pd.concat(transfer_summary_frames, ignore_index=True)
        transfer_summary["n_subjects"] = len(supervised_records)
        transfer_summary_path = output_root / "results" / "metrics" / "dual_segment_barlow_psd_wpli_transfer_summary.csv"
        transfer_summary_path.parent.mkdir(parents=True, exist_ok=True)
        transfer_summary.to_csv(transfer_summary_path, index=False)
        print(f"Wrote dual transfer summary: {transfer_summary_path}")

    for experiment_key, prediction_frames in dual_predictions_by_experiment.items():
        if prediction_frames:
            error_frame = _error_frequency_from_frames(
                prediction_frames,
                experiment_name=str(DUAL_EXPERIMENTS[experiment_key]["experiment_name"]),
            )
            dual_error_frames.append(error_frame)
    if dual_error_frames:
        dual_errors = pd.concat(dual_error_frames, ignore_index=True)
        dual_error_path = output_root / "results" / "metrics" / "dual_segment_barlow_error_frequency.csv"
        dual_errors.to_csv(dual_error_path, index=False)
        print(f"Wrote dual error frequency: {dual_error_path}")

    ablation_summary, ablation_errors = _build_ablation_outputs(
        output_root,
        dual_predictions_by_experiment=dual_predictions_by_experiment,
        dual_metrics_by_experiment=dual_metrics_by_experiment,
    )
    ablation_summary_path = output_root / "results" / "metrics" / "segment_barlow_ablation_comparison_summary.csv"
    ablation_summary.to_csv(ablation_summary_path, index=False)
    print(f"Wrote ablation summary: {ablation_summary_path}")
    if not ablation_errors.empty:
        ablation_error_path = output_root / "results" / "metrics" / "segment_barlow_ablation_error_frequency.csv"
        ablation_errors.to_csv(ablation_error_path, index=False)
        print(f"Wrote ablation error frequency: {ablation_error_path}")


def _selected_experiments(experiment_set: str) -> list[str]:
    if experiment_set in {"remaining", "all-dual"}:
        return ["dual-finetune", "dual-freeze", "dual-low-lr"]
    return [experiment_set]


def _run_or_load_branch_checkpoint(
    *,
    args,
    path_config,
    branch: str,
    feature_kind: str,
    fold_segments,
    fold,
    seed: int,
    source_feature_manifest_hash: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    checkpoint_dir = _checkpoint_dir(args, path_config, branch)
    checkpoint_path = checkpoint_dir / make_ssl_checkpoint_name(
        method="segssl",
        branch=branch,
        ssl_objective="barlow",
        seed=seed,
        fold_index=fold.fold_index,
        test_subject_id=fold.test_subject_id,
        embedding_dim=args.embedding_dim,
        pretrain_epochs=args.pretrain_epochs,
        ssl_data_scope=args.ssl_data_scope,
        tag=args.checkpoint_tag,
    )
    ssl_config = SegmentSSLTrainingConfig(
        objective="barlow",
        feature_kind=feature_kind,
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        lr=args.pretrain_lr,
        feature_mask_prob=args.feature_mask_prob,
        noise_std=args.noise_std,
        barlow_offdiag_weight=args.barlow_offdiag_weight,
        lambda_latent=args.lambda_latent,
        lambda_local=args.lambda_local,
        masked_latent_loss=args.masked_latent_loss,
        device=args.device,
        seed=seed + fold.fold_index,
    )
    expected_source_feature_manifest_hash = source_feature_manifest_hash
    if args.reuse_ssl_encoders and checkpoint_path.exists() and not args.force_retrain_ssl:
        expected_source_feature_manifest_hash = _source_manifest_hash_for_checkpoint_reuse(
            fold_segments,
            output_root=Path(path_config.output_root),
            checkpoint_path=checkpoint_path,
            current_hash=source_feature_manifest_hash,
        )
    expected_metadata = _checkpoint_metadata(
        args=args,
        branch=branch,
        feature_kind=feature_kind,
        fold_segments=fold_segments,
        fold=fold,
        seed=seed,
        ssl_config=ssl_config,
        source_feature_manifest_hash=expected_source_feature_manifest_hash,
    )
    if args.reuse_ssl_encoders and checkpoint_path.exists() and not args.force_retrain_ssl:
        checkpoint = load_ssl_encoder_checkpoint(checkpoint_path, expected_metadata)
        return checkpoint, _checkpoint_reuse_history(
            branch=branch,
            feature_kind=feature_kind,
            fold_segments=fold_segments,
            ssl_config=ssl_config,
            checkpoint_path=checkpoint_path,
        )
    if args.reuse_only:
        raise FileNotFoundError(f"Missing reusable {branch} Segment Barlow checkpoint: {checkpoint_path}")

    pretrained_state, history = run_segment_ssl_pretraining(fold_segments, ssl_config)
    encoder_state = extract_branch_encoder_state(pretrained_state, branch)
    metadata = dict(expected_metadata)
    metadata["created_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint = {
        "checkpoint_type": "segment_ssl_encoder",
        "metadata": metadata,
        "encoder_state_dict": encoder_state,
        "prefixed_state_dict": prefix_branch_encoder_state_dict(branch, encoder_state),
    }
    if args.save_ssl_encoders:
        save_ssl_encoder_checkpoint(
            checkpoint_path,
            metadata,
            encoder_state,
            prefixed_state_dict=checkpoint["prefixed_state_dict"],
        )
    history = history.copy()
    history["checkpoint_reused"] = False
    history["checkpoint_path"] = str(checkpoint_path)
    return checkpoint, history


def _checkpoint_dir(args, path_config, branch: str) -> Path:
    if branch == "psd" and args.psd_checkpoint_dir:
        return Path(args.psd_checkpoint_dir)
    if branch == "wpli" and args.wpli_checkpoint_dir:
        return Path(args.wpli_checkpoint_dir)
    if args.ssl_checkpoint_dir:
        return Path(args.ssl_checkpoint_dir)
    return Path(path_config.output_root) / "results" / "checkpoints" / "ssl_encoders"


def _checkpoint_metadata(
    *,
    args,
    branch: str,
    feature_kind: str,
    fold_segments,
    fold,
    seed: int,
    ssl_config: SegmentSSLTrainingConfig,
    source_feature_manifest_hash: str,
) -> dict[str, object]:
    return {
        "checkpoint_type": "segment_ssl_encoder",
        "branch": branch,
        "ssl_objective": "barlow",
        "segment_ssl_method": "segment_barlow",
        "base_seed": seed,
        "effective_seed": ssl_config.seed,
        "fold_index": fold.fold_index,
        "test_subject_id": fold.test_subject_id,
        "excluded_subject_id": fold.test_subject_id,
        "ssl_data_scope": args.ssl_data_scope,
        "historical_unlabeled_pretraining": args.ssl_data_scope in {"all-patient", "all-patient-health"},
        "segment_feature_kind": feature_kind,
        "supervised_feature_kind": "psd-fc-wpli",
        "feature_kind": feature_kind,
        "encoder_kind": "cnn",
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "projection_dim": args.projection_dim,
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_lr": args.pretrain_lr,
        "batch_size": args.pretrain_batch_size,
        "feature_mask_prob": args.feature_mask_prob,
        "noise_std": args.noise_std,
        "lambda_latent": args.lambda_latent,
        "lambda_local": args.lambda_local,
        "masked_latent_loss": args.masked_latent_loss,
        "barlow_offdiag_weight": args.barlow_offdiag_weight,
        "n_ssl_segments": len(fold_segments),
        "source_feature_manifest_hash": source_feature_manifest_hash,
    }


def _checkpoint_reuse_history(
    *,
    branch: str,
    feature_kind: str,
    fold_segments,
    ssl_config: SegmentSSLTrainingConfig,
    checkpoint_path: Path,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "epoch": 0,
                "objective": "barlow",
                "feature_kind": feature_kind,
                "branch": branch,
                "n_segments": len(fold_segments),
                "batch_size": ssl_config.batch_size,
                "embedding_dim": ssl_config.embedding_dim,
                "projection_dim": ssl_config.projection_dim,
                "learning_rate": ssl_config.lr,
                "feature_mask_prob": ssl_config.feature_mask_prob,
                "noise_std": ssl_config.noise_std,
                "lambda_latent": ssl_config.lambda_latent,
                "lambda_local": ssl_config.lambda_local,
                "loss": pd.NA,
                "alignment_loss": pd.NA,
                "masked_latent_loss": pd.NA,
                "local_reconstruction_loss": pd.NA,
                "barlow_on_diag_loss": pd.NA,
                "barlow_off_diag_loss": pd.NA,
                "checkpoint_reused": True,
                "checkpoint_path": str(checkpoint_path),
            }
        ]
    )


def _dual_source_manifest_hash(psd_segments, wpli_segments) -> str:
    return segment_records_manifest_hash([*psd_segments, *wpli_segments])


def _source_manifest_hash_for_checkpoint_reuse(
    records,
    *,
    output_root: Path,
    checkpoint_path: Path,
    current_hash: str,
) -> str:
    observed_hash = _checkpoint_source_manifest_hash(checkpoint_path)
    if not observed_hash or observed_hash == current_hash:
        return current_hash
    for source_root in _legacy_source_roots_from_checkpoint_manifest(checkpoint_path):
        legacy_hash = _segment_records_manifest_hash_with_source_root(
            records,
            output_root=output_root,
            source_root=source_root,
        )
        if legacy_hash == observed_hash:
            return legacy_hash
    return current_hash


def _checkpoint_source_manifest_hash(checkpoint_path: Path) -> str | None:
    try:
        checkpoint = load_ssl_encoder_checkpoint(checkpoint_path)
    except (EOFError, OSError, RuntimeError, ValueError):
        return None
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("source_feature_manifest_hash")
    return value if isinstance(value, str) and value else None


def _legacy_source_roots_from_checkpoint_manifest(checkpoint_path: Path) -> list[Path]:
    manifest_path = checkpoint_path.parent / "segment_ssl_checkpoint_manifest.csv"
    if not manifest_path.exists():
        return []
    try:
        manifest = pd.read_csv(manifest_path, usecols=["checkpoint_path"])
    except (OSError, ValueError):
        return []

    roots: list[Path] = []
    seen: set[str] = set()
    for checkpoint_text in manifest["checkpoint_path"].dropna().astype(str):
        if Path(checkpoint_text).name != checkpoint_path.name:
            continue
        root = _project_root_from_checkpoint_text(checkpoint_text)
        if root is None:
            continue
        root_key = str(root)
        if root_key not in seen:
            roots.append(root)
            seen.add(root_key)
    return roots


def _project_root_from_checkpoint_text(checkpoint_text: str) -> Path | None:
    normalized = checkpoint_text.replace("/", "\\")
    marker = "\\results\\checkpoints\\ssl_encoders\\"
    index = normalized.lower().find(marker)
    if index < 0:
        return None
    return Path(normalized[:index])


def _segment_records_manifest_hash_with_source_root(records, *, output_root: Path, source_root: Path) -> str:
    output_root = Path(output_root).resolve()
    source_root = Path(source_root)
    manifest_rows = []
    for record in records:
        record_source_path = Path(record.source_path).resolve()
        try:
            source_path = source_root / record_source_path.relative_to(output_root)
        except ValueError:
            source_path = record_source_path
        manifest_rows.append(
            {
                "group": record.group,
                "subject_id": record.subject_id,
                "subject_key": record.subject_key,
                "stage": record.stage,
                "state": record.state,
                "segment_index": record.segment_index,
                "branches": sorted(record.features),
                "source_path": str(source_path),
            }
        )
    payload = json.dumps(
        sorted(
            manifest_rows,
            key=lambda row: (
                row["group"],
                row["subject_key"],
                row["stage"],
                row["state"],
                row["segment_index"],
                row["source_path"],
            ),
        ),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dual_run_name(experiment_key: str, seed: int | str, supervised_lr: float) -> str:
    mode = "freeze" if experiment_key == "dual-freeze" else "finetune"
    return f"dual_segbarlow_psd_wpli_{mode}_lr{_number_token(supervised_lr)}_seed{seed}"


def _number_token(value: float) -> str:
    return str(value).replace(".", "_")


def _annotate_dual_frames(
    frames,
    *,
    run_name: str,
    experiment_name: str,
    seed: int | str,
    transfer_mode: str,
    supervised_lr: float,
    args,
) -> None:
    for frame in frames:
        frame["run_name"] = run_name
        frame["experiment_name"] = experiment_name
        frame["seed"] = seed
        frame["transfer_mode"] = transfer_mode
        frame["supervised_lr"] = supervised_lr
        frame["weight_decay"] = args.weight_decay
        frame["pretrained_psd"] = True
        frame["pretrained_wpli"] = True
        frame["psd_ssl_objective"] = "barlow"
        frame["wpli_ssl_objective"] = "barlow"
        frame["strict_loso_ssl"] = True
        frame["ssl_data_scope"] = args.ssl_data_scope
        frame["pretrain_epochs"] = args.pretrain_epochs
        frame["pretrain_lr"] = args.pretrain_lr
        frame["ssl_projection_dim"] = args.projection_dim


def _write_prediction_metric_pair(
    output_root: Path,
    run_name: str,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    prediction_path = output_root / "results" / "predictions" / f"dl_loso_predictions_{run_name}.csv"
    metric_path = output_root / "results" / "metrics" / f"dl_model_comparison_{run_name}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    metrics.to_csv(metric_path, index=False)


def compute_per_subject_error_frequency(prediction_files) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in prediction_files]
    return _error_frequency_from_frames(frames, experiment_name=";".join(str(path) for path in prediction_files))


def _error_frequency_from_frames(frames, *, experiment_name: str) -> pd.DataFrame:
    predictions = pd.concat([frame.copy() for frame in frames], ignore_index=True)
    required = {"subject_id", "y_true", "y_pred", "y_score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame missing required column(s): {', '.join(sorted(missing))}.")
    predictions["is_error"] = predictions["y_true"].astype(int) != predictions["y_pred"].astype(int)
    grouped = (
        predictions.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            n_runs=("is_error", "size"),
            n_errors=("is_error", "sum"),
            mean_y_score=("y_score", "mean"),
            std_y_score=("y_score", "std"),
        )
        .assign(
            error_rate=lambda frame: frame["n_errors"] / frame["n_runs"],
            experiment_name=experiment_name,
            experiments=experiment_name,
        )
        .sort_values(["error_rate", "subject_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return grouped


def _build_ablation_outputs(
    output_root: Path,
    *,
    dual_predictions_by_experiment: dict[str, list[pd.DataFrame]],
    dual_metrics_by_experiment: dict[str, list[pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiments = _existing_ablation_experiments(output_root)
    for key, frames in dual_metrics_by_experiment.items():
        experiments[str(DUAL_EXPERIMENTS[key]["experiment_name"])] = {
            "metrics": frames,
            "predictions": dual_predictions_by_experiment.get(key, []),
        }
    rows = []
    error_frames = []
    for experiment_name, payload in experiments.items():
        metrics_frames = payload["metrics"]
        prediction_frames = payload["predictions"]
        if not metrics_frames:
            rows.append({"experiment_name": experiment_name, "warning": "missing metrics", "n_seeds": 0})
            continue
        metrics = pd.concat(metrics_frames, ignore_index=True)
        seed_metrics = metrics[metrics["seed"].astype(str) != "ensemble"] if "seed" in metrics.columns else metrics
        row = {"experiment_name": experiment_name, "n_seeds": int(len(seed_metrics)), "warning": ""}
        for metric in ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "sensitivity", "specificity"):
            if metric in seed_metrics.columns:
                values = seed_metrics[metric].dropna().astype(float)
                row[f"{metric}_mean"] = float(values.mean()) if not values.empty else pd.NA
                row[f"{metric}_std"] = float(values.std(ddof=0)) if not values.empty else pd.NA
                row[f"{metric}_min"] = float(values.min()) if not values.empty else pd.NA
                row[f"{metric}_max"] = float(values.max()) if not values.empty else pd.NA
        if prediction_frames:
            ensemble_predictions, ensemble_metrics = aggregate_seed_ensemble_predictions(prediction_frames)
            for metric in ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc"):
                if metric in ensemble_metrics.columns:
                    row[f"ensemble_{metric}"] = float(ensemble_metrics[metric].iloc[0])
            error_frames.append(_error_frequency_from_frames(prediction_frames, experiment_name=experiment_name))
            row["per_subject_error_frequency"] = "segment_barlow_ablation_error_frequency.csv"
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = _add_ablation_deltas(summary)
    errors = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    return summary, errors


def _add_ablation_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    if "accuracy_mean" not in summary.columns:
        return summary
    baselines = {
        "no_ssl": "A_no_ssl_cnn",
        "psd_only": "B_psd_segment_barlow_only",
        "wpli_only": "C_fc_wpli_segment_barlow_only",
    }
    for suffix, name in baselines.items():
        match = summary.loc[summary["experiment_name"] == name]
        if match.empty or pd.isna(match["accuracy_mean"].iloc[0]):
            summary[f"delta_accuracy_mean_vs_{suffix}"] = pd.NA
            summary[f"delta_std_vs_{suffix}"] = pd.NA
            continue
        base_acc = float(match["accuracy_mean"].iloc[0])
        base_std = float(match["accuracy_std"].iloc[0]) if "accuracy_std" in match and not pd.isna(match["accuracy_std"].iloc[0]) else pd.NA
        summary[f"delta_accuracy_mean_vs_{suffix}"] = summary["accuracy_mean"].astype(float) - base_acc
        if "accuracy_std" in summary.columns and base_std is not pd.NA:
            summary[f"delta_std_vs_{suffix}"] = summary["accuracy_std"].astype(float) - base_std
    valid_accuracy = summary["accuracy_mean"].dropna()
    if not valid_accuracy.empty:
        summary["whether_best_mean"] = summary["accuracy_mean"] == valid_accuracy.max()
    if "accuracy_std" in summary.columns:
        valid_std = summary["accuracy_std"].dropna()
        if not valid_std.empty:
            summary["whether_best_stability"] = summary["accuracy_std"] == valid_std.min()
    return summary


def _existing_ablation_experiments(output_root: Path) -> dict[str, dict[str, list[pd.DataFrame]]]:
    specs = {
        "A_no_ssl_cnn": {
            "metric_patterns": [output_root / "results" / "metrics" / "dl_model_comparison_no_ssl_patient_psd_fc_wpli_seed*_ep100.csv"],
            "prediction_patterns": [output_root / "results" / "predictions" / "dl_loso_predictions_no_ssl_patient_psd_fc_wpli_seed*_ep100.csv"],
        },
        "B_psd_segment_barlow_only": {
            "metric_patterns": [
                output_root / "docs" / "experiment_exports" / "20260527_local_psd_segssl" / "metrics" / "dl_model_comparison_segssl_barlow_all-patient_psd-fc-wpli_finetune_seed*_pre20_proj32_mask0_03_noise0_02_mlp1_0_sup100.csv",
            ],
            "prediction_patterns": [
                output_root / "docs" / "experiment_exports" / "20260527_local_psd_segssl" / "predictions" / "dl_loso_predictions_segssl_barlow_all-patient_psd-fc-wpli_finetune_seed*_pre20_proj32_mask0_03_noise0_02_mlp1_0_sup100.csv",
            ],
        },
        "C_fc_wpli_segment_barlow_only": {
            "metric_patterns": [output_root / "results" / "metrics" / "dl_model_comparison_segssl_barlow_all-patient_fc-wpli_finetune_seed*_pre20_proj32_mask0_03_noise0_02_mlp1_0_sup100.csv"],
            "prediction_patterns": [output_root / "results" / "predictions" / "dl_loso_predictions_segssl_barlow_all-patient_fc-wpli_finetune_seed*_pre20_proj32_mask0_03_noise0_02_mlp1_0_sup100.csv"],
        },
    }
    experiments: dict[str, dict[str, list[pd.DataFrame]]] = {}
    for experiment_name, spec in specs.items():
        metric_files = _glob_seed_files(spec["metric_patterns"])
        prediction_files = _glob_seed_files(spec["prediction_patterns"])
        experiments[experiment_name] = {
            "metrics": [_read_with_experiment(path, experiment_name) for path in metric_files],
            "predictions": [_read_with_experiment(path, experiment_name) for path in prediction_files],
        }
    return experiments


def _glob_seed_files(patterns: list[Path]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in pattern.parent.glob(pattern.name) if "ensemble" not in path.name and "smoke" not in path.name)
    return sorted(files, key=lambda path: path.name)


def _read_with_experiment(path: Path, experiment_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["experiment_name"] = experiment_name
    if "seed" not in frame.columns:
        frame["seed"] = _seed_from_name(path.name)
    return frame


def _seed_from_name(name: str) -> str:
    marker = "seed"
    if marker not in name:
        return ""
    return name.split(marker, 1)[1].split("_", 1)[0]


if __name__ == "__main__":
    main()
