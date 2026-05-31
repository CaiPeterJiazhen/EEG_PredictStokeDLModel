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
    _safe_run_name,
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
    parser.add_argument("--encoder", choices=("cnn", "linear", "gncnn", "rescnn"), default="cnn")
    parser.add_argument(
        "--eeg-summary-features-enabled",
        action="store_true",
        help="Add a fold-scaled interpretable EEG-summary branch derived from the existing PSD/WPLI files.",
    )
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
    parser.add_argument(
        "--branch-barlow-weight",
        type=float,
        default=0.5,
        help="Weight for per-branch Barlow loss when --ssl-objective branch-barlow is used.",
    )
    parser.add_argument(
        "--latent-modality-mask-prob",
        type=float,
        default=0.5,
        help="Branch-embedding mask probability for the masked-barlow SSL objective.",
    )
    parser.add_argument("--byol-momentum", type=float, default=0.99)
    parser.add_argument("--byol-predictor-hidden-dim", type=int, default=64)
    parser.add_argument("--transfer-modes", nargs="+", choices=("finetune", "freeze-encoder"), default=["finetune"])
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--supervised-lr", type=float, default=1e-3)
    parser.add_argument("--supervised-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--loss-name",
        choices=("bce", "weighted_bce", "asymmetric_focal", "asymmetric_focal_fp_margin"),
        default="bce",
        help="Supervised fine-tuning loss.",
    )
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--negative-class-weight", type=float, default=1.5)
    parser.add_argument("--focal-gamma-pos", type=float, default=1.0)
    parser.add_argument("--focal-gamma-neg", type=float, default=2.0)
    parser.add_argument("--fp-margin", type=float, default=0.60)
    parser.add_argument("--fp-penalty-weight", type=float, default=0.25)
    parser.add_argument(
        "--encoder-lr-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to pretrained encoder parameters during supervised fine-tuning.",
    )
    parser.add_argument(
        "--transfer-head-lr-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to branch gates/projections during supervised fine-tuning.",
    )
    parser.add_argument(
        "--freeze-pretrained-encoder-epochs",
        type=int,
        default=0,
        help="For finetune transfer, keep pretrained encoder parameters frozen for the first N supervised epochs.",
    )
    parser.add_argument(
        "--ssl-bridge-enabled",
        action="store_true",
        help="Use a frozen copy of the SSL encoder beside the trainable CNN during supervised fine-tuning.",
    )
    parser.add_argument(
        "--ssl-consistency-weight",
        type=float,
        default=0.0,
        help="Weight for normalized trainable-vs-frozen SSL embedding consistency during bridge fine-tuning.",
    )
    parser.add_argument(
        "--ssl-two-head-enabled",
        action="store_true",
        help="Use trainable-CNN and frozen-SSL supervised heads with fixed logit fusion.",
    )
    parser.add_argument(
        "--ssl-aux-head-weight",
        type=float,
        default=0.5,
        help="Training-only auxiliary weight for the individual trainable and frozen SSL heads.",
    )
    parser.add_argument(
        "--ssl-fusion-weight",
        type=float,
        default=0.5,
        help="Fixed inference logit-fusion weight assigned to the frozen SSL head.",
    )
    parser.add_argument(
        "--ssl-residual-enabled",
        action="store_true",
        help="Use low-gate trainable CNN + frozen SSL + bridge residual probability fusion.",
    )
    parser.add_argument(
        "--ssl-residual-weight",
        type=float,
        default=0.14,
        help="Fixed probability-fusion weight assigned to the frozen SSL residual head.",
    )
    parser.add_argument(
        "--bridge-residual-weight",
        type=float,
        default=0.04,
        help="Fixed probability-fusion weight assigned to the trainable/frozen bridge residual head.",
    )
    parser.add_argument(
        "--ssl-residual-random-main",
        action="store_true",
        help="Keep the supervised main CNN randomly initialized and load Barlow weights only into residual SSL branches.",
    )
    parser.add_argument(
        "--ssl-residual-main-loss-only",
        action="store_true",
        help="Train the primary supervised loss on the main CNN head while using residual heads for auxiliary/inference fusion.",
    )
    parser.add_argument(
        "--ssl-residual-preserve-main",
        action="store_true",
        help="Train a fold-internal no-SSL main CNN first, freeze it, then train only Barlow residual heads.",
    )
    parser.add_argument(
        "--ssl-residual-select-weights-on-val",
        action="store_true",
        help="After residual training, select residual fusion weights on the fold-internal validation subjects only.",
    )
    parser.add_argument(
        "--ssl-residual-weight-candidates",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4],
        help="Candidate frozen-SSL residual weights for fold-internal validation selection.",
    )
    parser.add_argument(
        "--bridge-residual-weight-candidates",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.04, 0.08],
        help="Candidate bridge residual weights for fold-internal validation selection.",
    )
    parser.add_argument(
        "--ssl-residual-weight-selection-metric",
        choices=["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"],
        default="balanced_accuracy",
        help="Validation metric used to choose residual fusion weights.",
    )
    parser.add_argument(
        "--ssl-residual-logit-preservation-weight",
        type=float,
        default=0.0,
        help="Training loss weight for preserving the no-SSL/main residual-path logits.",
    )
    parser.add_argument(
        "--ssl-residual-logit-delta-enabled",
        action="store_true",
        help="Use a frozen main CNN plus bounded Barlow bridge correction in logit space.",
    )
    parser.add_argument(
        "--ssl-residual-logit-delta-scale",
        type=float,
        default=0.5,
        help="Maximum absolute logit correction from the Barlow bridge when logit-delta residual is enabled.",
    )
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument(
        "--embedding-adapter-dim",
        type=int,
        default=0,
        help="Hidden size for a zero-initialized residual adapter on the fused CNN embedding. 0 disables it.",
    )
    parser.add_argument(
        "--embedding-adapter-scale",
        type=float,
        default=1.0,
        help="Scale applied to the residual embedding adapter output.",
    )
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
        eeg_summary_features_enabled=args.eeg_summary_features_enabled,
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
                    branch_barlow_weight=args.branch_barlow_weight,
                    latent_modality_mask_prob=args.latent_modality_mask_prob,
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
                weight_decay=args.supervised_weight_decay,
                loss_name=args.loss_name,
                positive_class_weight=args.positive_class_weight,
                negative_class_weight=args.negative_class_weight,
                focal_gamma_pos=args.focal_gamma_pos,
                focal_gamma_neg=args.focal_gamma_neg,
                fp_margin=args.fp_margin,
                fp_penalty_weight=args.fp_penalty_weight,
                embedding_dim=args.embedding_dim,
                embedding_adapter_dim=args.embedding_adapter_dim,
                embedding_adapter_scale=args.embedding_adapter_scale,
                dropout=args.dropout,
                seed=seed,
                pretrained_transfer_mode=transfer_mode,
                encoder_lr_multiplier=args.encoder_lr_multiplier,
                transfer_head_lr_multiplier=args.transfer_head_lr_multiplier,
                freeze_pretrained_encoder_epochs=args.freeze_pretrained_encoder_epochs,
                ssl_bridge_enabled=args.ssl_bridge_enabled,
                ssl_consistency_weight=args.ssl_consistency_weight,
                ssl_two_head_enabled=args.ssl_two_head_enabled,
                ssl_aux_head_weight=args.ssl_aux_head_weight,
                ssl_fusion_weight=args.ssl_fusion_weight,
                ssl_residual_enabled=args.ssl_residual_enabled,
                ssl_residual_weight=args.ssl_residual_weight,
                bridge_residual_weight=args.bridge_residual_weight,
                ssl_residual_random_main=args.ssl_residual_random_main,
                ssl_residual_main_loss_only=args.ssl_residual_main_loss_only,
                ssl_residual_preserve_main=args.ssl_residual_preserve_main,
                ssl_residual_select_weights_on_val=args.ssl_residual_select_weights_on_val,
                ssl_residual_weight_candidates=tuple(args.ssl_residual_weight_candidates),
                bridge_residual_weight_candidates=tuple(args.bridge_residual_weight_candidates),
                ssl_residual_weight_selection_metric=args.ssl_residual_weight_selection_metric,
                ssl_residual_logit_preservation_weight=args.ssl_residual_logit_preservation_weight,
                ssl_residual_logit_delta_enabled=args.ssl_residual_logit_delta_enabled,
                ssl_residual_logit_delta_scale=args.ssl_residual_logit_delta_scale,
                eeg_summary_features_enabled=args.eeg_summary_features_enabled,
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
                encoder_lr_multiplier=args.encoder_lr_multiplier,
                transfer_head_lr_multiplier=args.transfer_head_lr_multiplier,
                freeze_pretrained_encoder_epochs=args.freeze_pretrained_encoder_epochs,
                ssl_bridge_enabled=args.ssl_bridge_enabled,
                ssl_consistency_weight=args.ssl_consistency_weight,
                ssl_two_head_enabled=args.ssl_two_head_enabled,
                ssl_aux_head_weight=args.ssl_aux_head_weight,
                ssl_fusion_weight=args.ssl_fusion_weight,
                ssl_residual_enabled=args.ssl_residual_enabled,
                ssl_residual_weight=args.ssl_residual_weight,
                bridge_residual_weight=args.bridge_residual_weight,
                ssl_residual_random_main=args.ssl_residual_random_main,
                ssl_residual_main_loss_only=args.ssl_residual_main_loss_only,
                ssl_residual_preserve_main=args.ssl_residual_preserve_main,
                ssl_residual_select_weights_on_val=args.ssl_residual_select_weights_on_val,
                ssl_residual_logit_preservation_weight=args.ssl_residual_logit_preservation_weight,
                ssl_residual_logit_delta_enabled=args.ssl_residual_logit_delta_enabled,
                ssl_residual_logit_delta_scale=args.ssl_residual_logit_delta_scale,
                embedding_adapter_dim=args.embedding_adapter_dim,
                embedding_adapter_scale=args.embedding_adapter_scale,
                latent_modality_mask_prob=args.latent_modality_mask_prob,
            )
            if args.output_tag:
                run_name = f"{run_name}_{args.output_tag}"
            if args.eeg_summary_features_enabled:
                run_name = f"{run_name}_eegsummary"
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
                frame["latent_modality_mask_prob"] = args.latent_modality_mask_prob
                frame["dropout"] = args.dropout
                frame["supervised_lr"] = args.supervised_lr
                frame["loss_name"] = args.loss_name
                frame["positive_class_weight"] = args.positive_class_weight
                frame["negative_class_weight"] = args.negative_class_weight
                frame["focal_gamma_pos"] = args.focal_gamma_pos
                frame["focal_gamma_neg"] = args.focal_gamma_neg
                frame["fp_margin"] = args.fp_margin
                frame["fp_penalty_weight"] = args.fp_penalty_weight
                frame["encoder_lr_multiplier"] = args.encoder_lr_multiplier
                frame["transfer_head_lr_multiplier"] = args.transfer_head_lr_multiplier
                frame["freeze_pretrained_encoder_epochs"] = args.freeze_pretrained_encoder_epochs
                frame["ssl_bridge_enabled"] = args.ssl_bridge_enabled
                frame["ssl_two_head_enabled"] = args.ssl_two_head_enabled
                frame["ssl_aux_head_weight"] = args.ssl_aux_head_weight
                frame["ssl_fusion_weight"] = args.ssl_fusion_weight
                frame["ssl_residual_enabled"] = args.ssl_residual_enabled
                frame["ssl_residual_weight"] = args.ssl_residual_weight
                frame["bridge_residual_weight"] = args.bridge_residual_weight
                frame["ssl_residual_random_main"] = args.ssl_residual_random_main
                frame["ssl_residual_main_loss_only"] = args.ssl_residual_main_loss_only
                frame["ssl_residual_preserve_main"] = args.ssl_residual_preserve_main
                frame["ssl_residual_select_weights_on_val"] = args.ssl_residual_select_weights_on_val
                frame["ssl_residual_weight_selection_metric"] = args.ssl_residual_weight_selection_metric
                frame["ssl_residual_logit_preservation_weight"] = args.ssl_residual_logit_preservation_weight
                frame["ssl_residual_logit_delta_enabled"] = args.ssl_residual_logit_delta_enabled
                frame["ssl_residual_logit_delta_scale"] = args.ssl_residual_logit_delta_scale
                frame["embedding_adapter_dim"] = args.embedding_adapter_dim
                frame["embedding_adapter_scale"] = args.embedding_adapter_scale
                frame["eeg_summary_features_enabled"] = args.eeg_summary_features_enabled
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
                args.latent_modality_mask_prob,
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
        latent_modality_mask_prob,
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
            encoder_lr_multiplier=args.encoder_lr_multiplier,
            transfer_head_lr_multiplier=args.transfer_head_lr_multiplier,
            freeze_pretrained_encoder_epochs=args.freeze_pretrained_encoder_epochs,
            ssl_bridge_enabled=args.ssl_bridge_enabled,
            ssl_consistency_weight=args.ssl_consistency_weight,
            ssl_two_head_enabled=args.ssl_two_head_enabled,
            ssl_aux_head_weight=args.ssl_aux_head_weight,
            ssl_fusion_weight=args.ssl_fusion_weight,
            ssl_residual_enabled=args.ssl_residual_enabled,
            ssl_residual_weight=args.ssl_residual_weight,
            bridge_residual_weight=args.bridge_residual_weight,
            ssl_residual_random_main=args.ssl_residual_random_main,
            ssl_residual_main_loss_only=args.ssl_residual_main_loss_only,
            ssl_residual_preserve_main=args.ssl_residual_preserve_main,
            ssl_residual_select_weights_on_val=args.ssl_residual_select_weights_on_val,
            ssl_residual_logit_preservation_weight=args.ssl_residual_logit_preservation_weight,
            ssl_residual_logit_delta_enabled=args.ssl_residual_logit_delta_enabled,
            ssl_residual_logit_delta_scale=args.ssl_residual_logit_delta_scale,
            embedding_adapter_dim=args.embedding_adapter_dim,
            embedding_adapter_scale=args.embedding_adapter_scale,
            latent_modality_mask_prob=latent_modality_mask_prob,
        )
        if args.output_tag:
            ensemble_run_name = f"{ensemble_run_name}_{args.output_tag}"
        if args.eeg_summary_features_enabled:
            ensemble_run_name = f"{ensemble_run_name}_eegsummary"
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
            frame["latent_modality_mask_prob"] = latent_modality_mask_prob
            frame["dropout"] = args.dropout
            frame["supervised_lr"] = args.supervised_lr
            frame["loss_name"] = args.loss_name
            frame["positive_class_weight"] = args.positive_class_weight
            frame["negative_class_weight"] = args.negative_class_weight
            frame["focal_gamma_pos"] = args.focal_gamma_pos
            frame["focal_gamma_neg"] = args.focal_gamma_neg
            frame["fp_margin"] = args.fp_margin
            frame["fp_penalty_weight"] = args.fp_penalty_weight
            frame["encoder_lr_multiplier"] = args.encoder_lr_multiplier
            frame["transfer_head_lr_multiplier"] = args.transfer_head_lr_multiplier
            frame["freeze_pretrained_encoder_epochs"] = args.freeze_pretrained_encoder_epochs
            frame["ssl_bridge_enabled"] = args.ssl_bridge_enabled
            frame["ssl_two_head_enabled"] = args.ssl_two_head_enabled
            frame["ssl_aux_head_weight"] = args.ssl_aux_head_weight
            frame["ssl_fusion_weight"] = args.ssl_fusion_weight
            frame["ssl_residual_enabled"] = args.ssl_residual_enabled
            frame["ssl_residual_weight"] = args.ssl_residual_weight
            frame["bridge_residual_weight"] = args.bridge_residual_weight
            frame["ssl_residual_random_main"] = args.ssl_residual_random_main
            frame["ssl_residual_main_loss_only"] = args.ssl_residual_main_loss_only
            frame["ssl_residual_preserve_main"] = args.ssl_residual_preserve_main
            frame["ssl_residual_select_weights_on_val"] = args.ssl_residual_select_weights_on_val
            frame["ssl_residual_weight_selection_metric"] = args.ssl_residual_weight_selection_metric
            frame["ssl_residual_logit_preservation_weight"] = args.ssl_residual_logit_preservation_weight
            frame["ssl_residual_logit_delta_enabled"] = args.ssl_residual_logit_delta_enabled
            frame["ssl_residual_logit_delta_scale"] = args.ssl_residual_logit_delta_scale
            frame["embedding_adapter_dim"] = args.embedding_adapter_dim
            frame["embedding_adapter_scale"] = args.embedding_adapter_scale
            frame["eeg_summary_features_enabled"] = args.eeg_summary_features_enabled
            frame["transfer_mode"] = transfer_mode
            frame["seed"] = "ensemble"
            frame["run_name"] = ensemble_run_name
        prediction_path, metric_path = _ensemble_output_paths(config.output_root, ensemble_run_name)
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


def _ensemble_output_paths(output_root: str | Path, ensemble_run_name: str) -> tuple[Path, Path]:
    root = Path(output_root)
    safe = _safe_run_name(ensemble_run_name)
    return (
        root / "results" / "predictions" / f"dl_loso_predictions_{safe}.csv",
        root / "results" / "metrics" / f"dl_model_comparison_{safe}.csv",
    )


if __name__ == "__main__":
    main()
