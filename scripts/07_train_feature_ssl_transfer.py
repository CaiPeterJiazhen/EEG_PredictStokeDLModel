from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.io.index import build_eeg_file_index
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.train_feature_ssl import (
    FEATURE_SSL_OBJECTIVES,
    FeatureSSLTrainingConfig,
    aggregate_seed_ensemble_predictions,
    build_feature_ssl_pair_records_from_feature_records,
    compute_feature_records_for_eeg_records,
    feature_ssl_pairs_for_scope,
    feature_ssl_transfer_run_name,
    run_feature_ssl_pretraining,
    widest_feature_ssl_scope,
    write_feature_ssl_transfer_outputs,
)
from eeg_recovery.training.train_ssl import SSL_DATA_SCOPES, select_ssl_records
from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    load_supervised_feature_records,
    run_loso_supervised_with_history,
)


DEFAULT_TRANSFER_SCOPES = (
    "supervised-baseline",
    "all-patient-baseline",
    "all-patient",
    "all-patient-health",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feature-space SSL pretraining and transfer to the PSD+FC-wPLI CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "configs" / "paths.example.yaml",
        help="Path YAML containing external EEG/workbook paths and output_root.",
    )
    parser.add_argument(
        "--data-scopes",
        nargs="+",
        choices=SSL_DATA_SCOPES,
        default=list(DEFAULT_TRANSFER_SCOPES),
        help="Feature SSL data scopes to evaluate.",
    )
    parser.add_argument("--architecture", choices=("multimodal",), default="multimodal")
    parser.add_argument("--feature-kind", choices=("psd-fc-wpli",), default="psd-fc-wpli")
    parser.add_argument("--fusion", choices=("concat", "gated"), default="gated")
    parser.add_argument("--encoder", choices=("cnn", "linear"), default="cnn")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pretrain-epochs", type=int, nargs="+", default=[20])
    parser.add_argument("--pretrain-batch-size", type=int, default=8)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--ssl-objective", nargs="+", choices=sorted(FEATURE_SSL_OBJECTIVES), default=["ntxent"])
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.2])
    parser.add_argument("--noise-std", type=float, nargs="+", default=[0.02])
    parser.add_argument("--feature-mask-prob", type=float, nargs="+", default=[0.05])
    parser.add_argument("--vicreg-invariance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-variance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-covariance-weight", type=float, default=1.0)
    parser.add_argument("--vicreg-variance-target", type=float, default=1.0)
    parser.add_argument("--barlow-offdiag-weight", type=float, default=0.005)
    parser.add_argument("--byol-momentum", type=float, default=0.99)
    parser.add_argument("--byol-predictor-hidden-dim", type=int, default=64)
    parser.add_argument("--transfer-modes", nargs="+", choices=("finetune", "freeze-encoder"), default=["finetune"])
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--supervised-lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None, help="Single seed alias. Prefer --seeds for sweeps.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--summary-name",
        default="feature_ssl_psd_fc_wpli_transfer_summary.csv",
        help="Summary CSV filename under results/metrics.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional filename-safe tag appended to per-run output names.",
    )
    parser.add_argument(
        "--limit-ssl-pairs",
        type=int,
        default=None,
        help="Optional cap after scope selection for smoke tests. Omit for real runs.",
    )
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else ([args.seed] if args.seed is not None else [2])

    config = load_path_config(args.config)
    labels = load_supervised_label_table(config)
    supervised_ids = labels["subject_id"].tolist()
    supervised_records = load_supervised_feature_records(
        config,
        labels,
        feature_kind=args.feature_kind,
    )
    eeg_records = build_eeg_file_index(
        config.patient_eeg_root,
        config.health_eeg_root,
        supervised_subject_ids=supervised_ids,
        validate_supervised_baseline=True,
    )
    source_scope = widest_feature_ssl_scope(args.data_scopes)
    eeg_records = select_ssl_records(eeg_records, data_scope=source_scope)
    feature_records = compute_feature_records_for_eeg_records(
        config,
        eeg_records,
        feature_kind=args.feature_kind,
    )
    all_ssl_pairs = build_feature_ssl_pair_records_from_feature_records(eeg_records, feature_records)

    summary_frames: list[pd.DataFrame] = []
    ensemble_prediction_groups: dict[tuple[object, ...], list[pd.DataFrame]] = {}
    folds = make_loso_folds([record.subject_id for record in supervised_records])
    for data_scope, transfer_mode, pretrain_epochs, ssl_objective, temperature, noise_std, feature_mask_prob in product(
        args.data_scopes,
        args.transfer_modes,
        args.pretrain_epochs,
        args.ssl_objective,
        args.temperature,
        args.noise_std,
        args.feature_mask_prob,
    ):
        for seed in seeds:
            pretrained_state_by_test_subject = {}
            ssl_history_frames: list[pd.DataFrame] = []
            for fold in folds:
                ssl_pairs = feature_ssl_pairs_for_scope(
                    all_ssl_pairs,
                    data_scope=data_scope,
                    strict_loso_test_subject_id=fold.test_subject_id,
                )
                if args.limit_ssl_pairs is not None:
                    if args.limit_ssl_pairs < 1:
                        raise SystemExit("--limit-ssl-pairs must be at least 1 when provided.")
                    ssl_pairs = ssl_pairs[: args.limit_ssl_pairs]
                ssl_config = FeatureSSLTrainingConfig(
                    data_scope=data_scope,
                    feature_kind=args.feature_kind,
                    fusion=args.fusion,
                    encoder_kind=args.encoder,
                    epochs=pretrain_epochs,
                    batch_size=args.pretrain_batch_size,
                    embedding_dim=args.embedding_dim,
                    projection_dim=args.projection_dim,
                    dropout=args.dropout,
                    lr=args.pretrain_lr,
                    ssl_objective=ssl_objective,
                    temperature=temperature,
                    noise_std=noise_std,
                    feature_mask_prob=feature_mask_prob,
                    vicreg_invariance_weight=args.vicreg_invariance_weight,
                    vicreg_variance_weight=args.vicreg_variance_weight,
                    vicreg_covariance_weight=args.vicreg_covariance_weight,
                    vicreg_variance_target=args.vicreg_variance_target,
                    barlow_offdiag_weight=args.barlow_offdiag_weight,
                    byol_momentum=args.byol_momentum,
                    byol_predictor_hidden_dim=args.byol_predictor_hidden_dim,
                    device=args.device,
                    seed=seed + fold.fold_index,
                )
                pretrained_state, ssl_history = run_feature_ssl_pretraining(ssl_pairs, ssl_config)
                ssl_history.insert(0, "fold_index", fold.fold_index)
                ssl_history.insert(1, "test_subject_id", fold.test_subject_id)
                ssl_history_frames.append(ssl_history)
                pretrained_state_by_test_subject[fold.test_subject_id] = pretrained_state
                print(
                    f"[{data_scope}] seed={seed} objective={ssl_objective} transfer={transfer_mode} pre={pretrain_epochs} "
                    f"temp={temperature} noise={noise_std} mask={feature_mask_prob} "
                    f"fold={fold.fold_index} test={fold.test_subject_id} ssl_pairs={len(ssl_pairs)}"
                )

            supervised_config = SupervisedTrainingConfig(
                architecture=args.architecture,
                feature_kind=args.feature_kind,
                fusion=args.fusion,
                encoder_kind=args.encoder,
                device=args.device,
                epochs=args.supervised_epochs,
                patience=args.patience,
                lr=args.supervised_lr,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
                seed=seed,
                pretrained_transfer_mode=transfer_mode,
            )
            predictions, metrics, supervised_loss = run_loso_supervised_with_history(
                supervised_records,
                supervised_config,
                pretrained_state_by_test_subject=pretrained_state_by_test_subject,
            )
            run_name = feature_ssl_transfer_run_name(
                data_scope=data_scope,
                feature_kind=args.feature_kind,
                fusion=args.fusion,
                encoder_kind=args.encoder,
                transfer_mode=transfer_mode,
                seed=seed,
                ssl_objective=ssl_objective,
                pretrain_epochs=pretrain_epochs,
                temperature=temperature,
                noise_std=noise_std,
                feature_mask_prob=feature_mask_prob,
                pretrain_batch_size=args.pretrain_batch_size,
                supervised_epochs=args.supervised_epochs,
                embedding_dim=args.embedding_dim,
                projection_dim=args.projection_dim,
                dropout=args.dropout,
                pretrain_lr=args.pretrain_lr,
                supervised_lr=args.supervised_lr,
            )
            if args.output_tag:
                run_name = f"{run_name}_{args.output_tag}"
            ssl_history_all = pd.concat(ssl_history_frames, ignore_index=True)
            for frame in (predictions, metrics, supervised_loss, ssl_history_all):
                frame["ssl_data_scope"] = data_scope
                frame["ssl_objective"] = ssl_objective
                frame["pretrain_epochs"] = pretrain_epochs
                frame["pretrain_lr"] = args.pretrain_lr
                frame["ssl_projection_dim"] = args.projection_dim
                frame["embedding_dim"] = args.embedding_dim
                frame["ssl_temperature"] = temperature
                frame["ssl_noise_std"] = noise_std
                frame["ssl_feature_mask_prob"] = feature_mask_prob
                frame["barlow_offdiag_weight"] = args.barlow_offdiag_weight
                frame["dropout"] = args.dropout
                frame["supervised_lr"] = args.supervised_lr
                frame["transfer_mode"] = transfer_mode
                frame["seed"] = seed
                frame["run_name"] = run_name
            paths = write_feature_ssl_transfer_outputs(
                output_root=config.output_root,
                run_name=run_name,
                predictions=predictions,
                metrics=metrics,
                ssl_history=ssl_history_all,
                supervised_loss_history=supervised_loss,
            )
            summary_frames.append(metrics)
            ensemble_key = (
                data_scope,
                transfer_mode,
                pretrain_epochs,
                ssl_objective,
                temperature,
                noise_std,
                feature_mask_prob,
            )
            ensemble_prediction_groups.setdefault(ensemble_key, []).append(predictions)
            print(f"[{data_scope}] Wrote predictions: {paths['predictions']}")
            print(f"[{data_scope}] Wrote metrics: {paths['metrics']}")
            print(f"[{data_scope}] Wrote SSL history: {paths['ssl_history']}")

    for (
        data_scope,
        transfer_mode,
        pretrain_epochs,
        ssl_objective,
        temperature,
        noise_std,
        feature_mask_prob,
    ), prediction_frames in ensemble_prediction_groups.items():
        if len(prediction_frames) < 2:
            continue
        ensemble_predictions, ensemble_metrics = aggregate_seed_ensemble_predictions(prediction_frames)
        ensemble_run_name = feature_ssl_transfer_run_name(
            data_scope=data_scope,
            feature_kind=args.feature_kind,
            fusion=args.fusion,
            encoder_kind=args.encoder,
            transfer_mode=transfer_mode,
            seed=f"ensemble{len(prediction_frames)}",
            ssl_objective=ssl_objective,
            pretrain_epochs=pretrain_epochs,
            temperature=temperature,
            noise_std=noise_std,
            feature_mask_prob=feature_mask_prob,
            pretrain_batch_size=args.pretrain_batch_size,
            supervised_epochs=args.supervised_epochs,
            embedding_dim=args.embedding_dim,
            projection_dim=args.projection_dim,
            dropout=args.dropout,
            pretrain_lr=args.pretrain_lr,
            supervised_lr=args.supervised_lr,
        )
        if args.output_tag:
            ensemble_run_name = f"{ensemble_run_name}_{args.output_tag}"
        for frame in (ensemble_predictions, ensemble_metrics):
            frame["model"] = f"seed_ensemble_{args.architecture}_{args.feature_kind}_{args.fusion}_{args.encoder}"
            frame["architecture"] = args.architecture
            frame["feature_kind"] = args.feature_kind
            frame["fusion"] = args.fusion
            frame["encoder_kind"] = args.encoder
            frame["pretrained"] = True
            frame["pretrained_transfer_mode"] = transfer_mode
            frame["status"] = "trained"
            frame["skip_reason"] = ""
            frame["ssl_data_scope"] = data_scope
            frame["ssl_objective"] = ssl_objective
            frame["pretrain_epochs"] = pretrain_epochs
            frame["pretrain_lr"] = args.pretrain_lr
            frame["ssl_projection_dim"] = args.projection_dim
            frame["embedding_dim"] = args.embedding_dim
            frame["ssl_temperature"] = temperature
            frame["ssl_noise_std"] = noise_std
            frame["ssl_feature_mask_prob"] = feature_mask_prob
            frame["barlow_offdiag_weight"] = args.barlow_offdiag_weight
            frame["dropout"] = args.dropout
            frame["supervised_lr"] = args.supervised_lr
            frame["transfer_mode"] = transfer_mode
            frame["seed"] = "ensemble"
            frame["run_name"] = ensemble_run_name
        prediction_path = (
            Path(config.output_root)
            / "results"
            / "predictions"
            / f"dl_loso_predictions_{ensemble_run_name}.csv"
        )
        metric_path = (
            Path(config.output_root)
            / "results"
            / "metrics"
            / f"dl_model_comparison_{ensemble_run_name}.csv"
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        ensemble_predictions.to_csv(prediction_path, index=False)
        ensemble_metrics.to_csv(metric_path, index=False)
        summary_frames.append(ensemble_metrics)
        print(f"[{data_scope}] Wrote ensemble predictions: {prediction_path}")
        print(f"[{data_scope}] Wrote ensemble metrics: {metric_path}")

    summary = pd.concat(summary_frames, ignore_index=True)
    summary_path = Path(config.output_root) / "results" / "metrics" / args.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
