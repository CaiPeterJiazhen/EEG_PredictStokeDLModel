from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.models.multimodal_model import branches_for_feature_kind
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.segment_ssl_dataset import (
    load_segment_ssl_records,
    merge_segment_modalities,
    segment_records_for_scope,
)
from eeg_recovery.training.train_feature_ssl import aggregate_seed_ensemble_predictions
from eeg_recovery.training.train_segment_ssl import (
    SEGMENT_SSL_OBJECTIVES,
    SegmentSSLTrainingConfig,
    extract_branch_encoder_state,
    per_subject_error_frequency,
    run_segment_ssl_pretraining,
    segment_records_manifest_hash,
    segment_ssl_transfer_run_name,
    write_segment_ssl_only_cache_history,
    summarize_seed_metrics,
    write_segment_ssl_transfer_outputs,
)
from eeg_recovery.training.ssl_checkpointing import (
    load_reusable_ssl_encoder_checkpoint,
    make_ssl_checkpoint_name,
    prefix_branch_encoder_state_dict,
    save_ssl_encoder_checkpoint,
    update_ssl_checkpoint_manifest,
)
from eeg_recovery.training.train_ssl import SSL_DATA_SCOPES
from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    load_supervised_feature_records,
    run_loso_supervised_with_history,
)


DEFAULT_SEEDS = [0, 1, 2, 3, 7, 13]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run segment-level negative-free SSL and transfer encoder weights to the PSD+FC-wPLI gated CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--data-scope", choices=SSL_DATA_SCOPES, default="all-patient")
    parser.add_argument("--objective", choices=sorted(SEGMENT_SSL_OBJECTIVES), nargs="+", default=["barlow", "vicreg"])
    parser.add_argument(
        "--segment-feature-kind",
        choices=("psd", "fc", "fc-wpli", "psd-fc-wpli"),
        default="psd",
        help="Segment cache modalities used for SSL pretraining.",
    )
    parser.add_argument("--supervised-feature-kind", choices=("psd-fc-wpli",), default="psd-fc-wpli")
    parser.add_argument("--transfer-mode", choices=("finetune", "freeze-encoder"), default="finetune")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--feature-mask-prob", type=float, choices=(0.01, 0.03, 0.05), default=0.03)
    parser.add_argument("--noise-std", type=float, choices=(0.01, 0.02), default=0.02)
    parser.add_argument("--lambda-latent", type=float, default=1.0)
    parser.add_argument("--lambda-local", type=float, default=0.1)
    parser.add_argument("--masked-latent-loss", choices=("cosine", "mse"), default="cosine")
    parser.add_argument("--vicreg-invariance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-variance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-covariance-weight", type=float, default=1.0)
    parser.add_argument("--barlow-offdiag-weight", type=float, default=0.005)
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--supervised-lr", type=float, default=0.002)
    parser.add_argument("--supervised-weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--limit-segments", type=int, default=None, help="Optional cap for smoke tests only.")
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--save-ssl-encoders", action="store_true")
    parser.add_argument("--reuse-ssl-encoders", action="store_true")
    parser.add_argument("--ssl-checkpoint-dir", default=None)
    parser.add_argument("--force-retrain-ssl", action="store_true")
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--checkpoint-tag", default=None)
    parser.add_argument(
        "--ssl-only-cache",
        action="store_true",
        help="Run fold-specific Segment SSL pretraining/checkpoint caching only; skip supervised transfer.",
    )
    args = parser.parse_args()
    if args.reuse_only and not args.reuse_ssl_encoders:
        raise SystemExit("--reuse-only requires --reuse-ssl-encoders.")
    if args.reuse_only and args.force_retrain_ssl:
        raise SystemExit("--reuse-only cannot be combined with --force-retrain-ssl.")

    path_config = load_path_config(args.config)
    labels = load_supervised_label_table(path_config)
    supervised_ids = labels["subject_id"].tolist()
    supervised_records = load_supervised_feature_records(
        path_config,
        labels,
        feature_kind=args.supervised_feature_kind,
    )
    segment_records = _load_segment_records_for_feature_kind(
        Path(path_config.output_root),
        args.segment_feature_kind,
    )

    folds = make_loso_folds([record.subject_id for record in supervised_records])
    summary_frames: list[pd.DataFrame] = []
    prediction_groups: dict[tuple[str, str], list[pd.DataFrame]] = {}
    metrics_groups: dict[tuple[str, str], list[pd.DataFrame]] = {}

    for objective in args.objective:
        for seed in args.seeds:
            pretrained_state_by_test_subject = {}
            ssl_history_frames: list[pd.DataFrame] = []
            for fold in folds:
                fold_segments = segment_records_for_scope(
                    segment_records,
                    data_scope=args.data_scope,
                    strict_loso_test_subject_id=fold.test_subject_id,
                    supervised_subject_ids=supervised_ids,
                )
                if args.limit_segments is not None:
                    if args.limit_segments < 1:
                        raise SystemExit("--limit-segments must be at least 1 when provided.")
                    fold_segments = fold_segments[: args.limit_segments]
                ssl_config = SegmentSSLTrainingConfig(
                    objective=objective,
                    feature_kind=args.segment_feature_kind,
                    epochs=args.pretrain_epochs,
                    batch_size=args.pretrain_batch_size,
                    embedding_dim=args.embedding_dim,
                    projection_dim=args.projection_dim,
                    dropout=args.dropout,
                    lr=args.pretrain_lr,
                    feature_mask_prob=args.feature_mask_prob,
                    noise_std=args.noise_std,
                    vicreg_invariance_weight=args.vicreg_invariance_weight,
                    vicreg_variance_weight=args.vicreg_variance_weight,
                    vicreg_covariance_weight=args.vicreg_covariance_weight,
                    barlow_offdiag_weight=args.barlow_offdiag_weight,
                    lambda_latent=args.lambda_latent,
                    lambda_local=args.lambda_local,
                    masked_latent_loss=args.masked_latent_loss,
                    device=args.device,
                    seed=seed + fold.fold_index,
                )
                pretrained_state, ssl_history = _run_or_load_segment_ssl_pretraining(
                    fold_segments=fold_segments,
                    ssl_config=ssl_config,
                    path_config=path_config,
                    args=args,
                    objective=objective,
                    seed=seed,
                    fold=fold,
                )
                ssl_history.insert(0, "fold_index", fold.fold_index)
                ssl_history.insert(1, "test_subject_id", fold.test_subject_id)
                ssl_history["ssl_data_scope"] = args.data_scope
                ssl_history["historical_unlabeled_pretraining"] = args.data_scope in {"all-patient", "all-patient-health"}
                ssl_history_frames.append(ssl_history)
                pretrained_state_by_test_subject[fold.test_subject_id] = pretrained_state
                print(
                    f"[segssl] objective={objective} seed={seed} fold={fold.fold_index} "
                    f"test={fold.test_subject_id} segments={len(fold_segments)}"
                )

            ssl_history_all = pd.concat(ssl_history_frames, ignore_index=True)
            if args.ssl_only_cache or args.supervised_epochs == 0:
                run_name = segment_ssl_transfer_run_name(
                    objective=objective,
                    feature_kind=args.segment_feature_kind,
                    data_scope=args.data_scope,
                    transfer_mode=args.transfer_mode,
                    seed=seed,
                    pretrain_epochs=args.pretrain_epochs,
                    projection_dim=args.projection_dim,
                    feature_mask_prob=args.feature_mask_prob,
                    noise_std=args.noise_std,
                    lambda_latent=args.lambda_latent,
                    supervised_epochs=0,
                )
                if args.output_tag:
                    run_name = f"{run_name}_{args.output_tag}"
                ssl_history_all["run_name"] = run_name
                ssl_history_all["ssl_only_cache"] = True
                safe_history_path = write_segment_ssl_only_cache_history(
                    output_root=path_config.output_root,
                    run_name=run_name,
                    ssl_history=ssl_history_all,
                )
                print(f"Wrote SSL-only cache history: {safe_history_path}")
                continue

            supervised_config = SupervisedTrainingConfig(
                architecture="multimodal",
                feature_kind=args.supervised_feature_kind,
                fusion="gated",
                encoder_kind="cnn",
                device=args.device,
                epochs=args.supervised_epochs,
                patience=args.patience,
                lr=args.supervised_lr,
                weight_decay=args.supervised_weight_decay,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
                seed=seed,
                pretrained_transfer_mode=args.transfer_mode,
            )
            predictions, metrics, loss_history = run_loso_supervised_with_history(
                supervised_records,
                supervised_config,
                pretrained_state_by_test_subject=pretrained_state_by_test_subject,
            )
            run_name = segment_ssl_transfer_run_name(
                objective=objective,
                feature_kind=args.segment_feature_kind,
                data_scope=args.data_scope,
                transfer_mode=args.transfer_mode,
                seed=seed,
                pretrain_epochs=args.pretrain_epochs,
                projection_dim=args.projection_dim,
                feature_mask_prob=args.feature_mask_prob,
                noise_std=args.noise_std,
                lambda_latent=args.lambda_latent,
                supervised_epochs=args.supervised_epochs,
            )
            if args.output_tag:
                run_name = f"{run_name}_{args.output_tag}"
            for frame in (predictions, metrics, loss_history, ssl_history_all):
                frame["run_name"] = run_name
                frame["segment_ssl"] = True
                frame["segment_ssl_objective"] = objective
                frame["segment_ssl_feature_kind"] = args.segment_feature_kind
                frame["ssl_data_scope"] = args.data_scope
                frame["historical_unlabeled_pretraining"] = args.data_scope in {"all-patient", "all-patient-health"}
                frame["seed"] = seed
                frame["lambda_latent"] = args.lambda_latent
                frame["lambda_local"] = args.lambda_local
                frame["ssl_projection_dim"] = args.projection_dim
                frame["ssl_feature_mask_prob"] = args.feature_mask_prob
                frame["ssl_noise_std"] = args.noise_std
            paths = write_segment_ssl_transfer_outputs(
                output_root=path_config.output_root,
                run_name=run_name,
                predictions=predictions,
                metrics=metrics,
                ssl_history=ssl_history_all,
                supervised_loss_history=loss_history,
            )
            summary_frames.append(metrics)
            prediction_groups.setdefault((objective, args.data_scope), []).append(predictions)
            metrics_groups.setdefault((objective, args.data_scope), []).append(metrics)
            print(f"Wrote predictions: {paths['predictions']}")
            print(f"Wrote metrics: {paths['metrics']}")
            print(f"Wrote SSL history: {paths['ssl_history']}")

    for (objective, data_scope), prediction_frames in prediction_groups.items():
        if len(prediction_frames) < 2:
            continue
        ensemble_predictions, ensemble_metrics = aggregate_seed_ensemble_predictions(prediction_frames)
        ensemble_run_name = segment_ssl_transfer_run_name(
            objective=objective,
            feature_kind=args.segment_feature_kind,
            data_scope=data_scope,
            transfer_mode=args.transfer_mode,
            seed=f"ensemble{len(prediction_frames)}",
            pretrain_epochs=args.pretrain_epochs,
            projection_dim=args.projection_dim,
            feature_mask_prob=args.feature_mask_prob,
            noise_std=args.noise_std,
            lambda_latent=args.lambda_latent,
            supervised_epochs=args.supervised_epochs,
        )
        if args.output_tag:
            ensemble_run_name = f"{ensemble_run_name}_{args.output_tag}"
        for frame in (ensemble_predictions, ensemble_metrics):
            frame["run_name"] = ensemble_run_name
            frame["segment_ssl"] = True
            frame["segment_ssl_objective"] = objective
            frame["segment_ssl_feature_kind"] = args.segment_feature_kind
            frame["ssl_data_scope"] = data_scope
            frame["seed"] = "ensemble"
        prediction_path = Path(path_config.output_root) / "results" / "predictions" / f"dl_loso_predictions_{ensemble_run_name}.csv"
        metric_path = Path(path_config.output_root) / "results" / "metrics" / f"dl_model_comparison_{ensemble_run_name}.csv"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        ensemble_predictions.to_csv(prediction_path, index=False)
        ensemble_metrics.to_csv(metric_path, index=False)
        summary_frames.append(ensemble_metrics)

        seed_summary = summarize_seed_metrics(metrics_groups[(objective, data_scope)])
        errors = per_subject_error_frequency(prediction_frames)
        segment_tag = _path_token(args.segment_feature_kind)
        seed_summary_path = Path(path_config.output_root) / "results" / "metrics" / f"segssl_seed_summary_{objective}_{data_scope}_{segment_tag}.csv"
        errors_path = Path(path_config.output_root) / "results" / "metrics" / f"segssl_subject_error_frequency_{objective}_{data_scope}_{segment_tag}.csv"
        seed_summary.to_csv(seed_summary_path, index=False)
        errors.to_csv(errors_path, index=False)
        print(f"Wrote seed summary: {seed_summary_path}")
        print(f"Wrote per-subject error frequency: {errors_path}")

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        summary_path = Path(path_config.output_root) / "results" / "metrics" / f"segssl_transfer_summary_{_path_token(args.segment_feature_kind)}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Wrote summary: {summary_path}")


def _load_segment_records_for_feature_kind(output_root: Path, feature_kind: str):
    branches = branches_for_feature_kind(feature_kind)
    segment_root = output_root / "data" / "features" / "segment_level"
    if branches == ("psd",):
        return load_segment_ssl_records(segment_root / "psd", branches=("psd",))
    if "psd" not in branches:
        return load_segment_ssl_records(segment_root / "fc", branches=branches)

    fc_branches = tuple(branch for branch in branches if branch != "psd")
    psd_records = load_segment_ssl_records(segment_root / "psd", branches=("psd",))
    fc_records = load_segment_ssl_records(segment_root / "fc", branches=fc_branches)
    merged = merge_segment_modalities(
        psd_records,
        fc_records,
        primary_branch="psd",
        secondary_branch=fc_branches,
    )
    if not merged:
        raise SystemExit(
            "No matching PSD and FC segment caches were found. "
            "Run scripts/17_compute_segment_features.py with --feature-kind psd-fc-wpli first."
        )
    return merged


def _path_token(value: str) -> str:
    return value.replace("-", "_")


def _run_or_load_segment_ssl_pretraining(
    *,
    fold_segments,
    ssl_config: SegmentSSLTrainingConfig,
    path_config,
    args,
    objective: str,
    seed: int,
    fold,
) -> tuple[dict, pd.DataFrame]:
    branches = branches_for_feature_kind(args.segment_feature_kind)
    checkpoint_dir = Path(args.ssl_checkpoint_dir) if args.ssl_checkpoint_dir else Path(path_config.output_root) / "results" / "checkpoints" / "ssl_encoders"
    expected_metadata = {
        branch: _ssl_checkpoint_metadata(
            args=args,
            objective=objective,
            branch=branch,
            seed=seed,
            fold=fold,
            fold_segments=fold_segments,
            ssl_config=ssl_config,
        )
        for branch in branches
    }
    checkpoint_paths = {
        branch: checkpoint_dir
        / make_ssl_checkpoint_name(
            method="segssl",
            branch=branch,
            ssl_objective=objective,
            seed=seed,
            fold_index=fold.fold_index,
            test_subject_id=fold.test_subject_id,
            embedding_dim=args.embedding_dim,
            pretrain_epochs=args.pretrain_epochs,
            ssl_data_scope=args.data_scope,
            tag=args.checkpoint_tag,
        )
        for branch in branches
    }

    if args.reuse_ssl_encoders and not args.force_retrain_ssl:
        loaded_state: dict = {}
        missing_branches = []
        for branch, checkpoint_path in checkpoint_paths.items():
            try:
                checkpoint = load_reusable_ssl_encoder_checkpoint(
                    checkpoint_path,
                    expected_metadata=expected_metadata[branch],
                    reuse_only=False,
                )
            except ValueError as exc:
                if not (args.ssl_only_cache and args.save_ssl_encoders):
                    raise
                print(
                    f"[segssl] existing checkpoint is not reusable and will be regenerated "
                    f"during explicit ssl-only cache refresh: {checkpoint_path} ({exc})"
                )
                checkpoint = None
            if checkpoint is None:
                missing_branches.append(branch)
                continue
            encoder_state = checkpoint.get("encoder_state_dict")
            if not isinstance(encoder_state, dict):
                raise ValueError(f"SSL checkpoint {checkpoint_path} is missing encoder_state_dict.")
            loaded_state.update(prefix_branch_encoder_state_dict(branch, encoder_state))
        if not missing_branches and loaded_state:
            for branch, checkpoint_path in checkpoint_paths.items():
                update_ssl_checkpoint_manifest(
                    checkpoint_dir,
                    checkpoint_path,
                    expected_metadata[branch],
                    status="reused",
                )
            return loaded_state, _checkpoint_reuse_history(
                objective=objective,
                feature_kind=args.segment_feature_kind,
                fold_segments=fold_segments,
                ssl_config=ssl_config,
                checkpoint_paths=checkpoint_paths,
            )
        if args.reuse_only:
            missing = ", ".join(missing_branches)
            raise FileNotFoundError(f"Missing reusable Segment SSL checkpoint(s) for branch(es): {missing}")
    elif args.reuse_only:
        missing = [branch for branch, path in checkpoint_paths.items() if not path.exists()]
        if missing:
            joined = ", ".join(missing)
            raise FileNotFoundError(f"Missing reusable Segment SSL checkpoint(s) for branch(es): {joined}")

    pretrained_state, ssl_history = run_segment_ssl_pretraining(fold_segments, ssl_config)
    if args.save_ssl_encoders:
        for branch, checkpoint_path in checkpoint_paths.items():
            encoder_state = extract_branch_encoder_state(pretrained_state, branch)
            metadata = dict(expected_metadata[branch])
            metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            save_ssl_encoder_checkpoint(
                checkpoint_path,
                metadata,
                encoder_state,
                prefixed_state_dict=prefix_branch_encoder_state_dict(branch, encoder_state),
            )
            update_ssl_checkpoint_manifest(
                checkpoint_dir,
                checkpoint_path,
                metadata,
                status="saved",
            )
    return pretrained_state, ssl_history


def _ssl_checkpoint_metadata(
    *,
    args,
    objective: str,
    branch: str,
    seed: int,
    fold,
    fold_segments,
    ssl_config: SegmentSSLTrainingConfig,
) -> dict[str, object]:
    return {
        "checkpoint_type": "segment_ssl_encoder",
        "segment_ssl_method": f"segment_{objective}",
        "ssl_objective": objective,
        "branch": branch,
        "base_seed": seed,
        "effective_seed": ssl_config.seed,
        "fold_index": fold.fold_index,
        "test_subject_id": fold.test_subject_id,
        "excluded_subject_id": fold.test_subject_id,
        "ssl_data_scope": args.data_scope,
        "historical_unlabeled_pretraining": args.data_scope in {"all-patient", "all-patient-health"},
        "segment_feature_kind": args.segment_feature_kind,
        "supervised_feature_kind": args.supervised_feature_kind,
        "feature_kind": args.segment_feature_kind,
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
        "source_feature_manifest_hash": segment_records_manifest_hash(fold_segments),
    }


def _checkpoint_reuse_history(
    *,
    objective: str,
    feature_kind: str,
    fold_segments,
    ssl_config: SegmentSSLTrainingConfig,
    checkpoint_paths: dict,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "epoch": 0,
                "objective": objective,
                "feature_kind": feature_kind,
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
                "vicreg_invariance_loss": pd.NA,
                "vicreg_variance_loss": pd.NA,
                "vicreg_covariance_loss": pd.NA,
                "checkpoint_reused": True,
                "checkpoint_paths": ";".join(str(path) for path in checkpoint_paths.values()),
            }
        ]
    )


if __name__ == "__main__":
    main()
