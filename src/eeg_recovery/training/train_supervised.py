from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from eeg_recovery.config import PathConfig
from eeg_recovery.features.eeg_summary import compute_subject_eeg_summary_features
from eeg_recovery.features.qeeg_slowing import compute_subject_qeeg_slowing_features
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.models.dual_state_model import DualStateEEGModel
from eeg_recovery.models.fusion_3d_model import Fusion3DEEGModel
from eeg_recovery.models.multimodal_model import (
    EEGSummaryGatedMultimodalEEGModel,
    FrozenMainSSLLogitDeltaMultimodalEEGModel,
    FrozenMainSSLResidualMultimodalEEGModel,
    MultimodalEEGModel,
    QEEGGuidedMultimodalEEGModel,
    SSLBridgeMultimodalEEGModel,
    SSLResidualMultimodalEEGModel,
    SSLTwoHeadMultimodalEEGModel,
    branches_for_feature_kind,
)
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.optimizers import SAM, build_adamw_optimizer, build_swa_model, update_swa_batch_norm
from eeg_recovery.training.schedulers import EarlyStopping, build_reduce_on_plateau


@dataclass(frozen=True)
class SupervisedFeatureRecord:
    subject_id: str
    label: int
    eo: np.ndarray
    ec: np.ndarray
    modalities: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    eeg_summary: np.ndarray | None = None
    eeg_summary_feature_names: tuple[str, ...] = ()
    qeeg_features: np.ndarray | None = None
    qeeg_feature_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    architecture: str = "dual_state"
    feature_kind: str = "psd"
    fusion: str = "concat"
    encoder_kind: str = "cnn"
    device: str = "auto"
    epochs: int = 50
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer_name: str = "adam"
    use_swa: bool = False
    swa_start_epoch: int = 50
    swa_lr: float = 5e-4
    finetune_schedule: str = "standard"
    encoder_lr: float | None = None
    head_lr: float | None = None
    freeze_encoder_epochs: int = 0
    freeze_conv_backbone: bool = False
    sam_rho: float = 0.05
    loss_name: str = "bce"
    focal_gamma_pos: float = 1.0
    focal_gamma_neg: float = 2.0
    fp_margin: float = 0.60
    fp_penalty_weight: float = 0.25
    positive_class_weight: float = 1.0
    negative_class_weight: float = 1.5
    embedding_dim: int = 16
    dropout: float = 0.1
    seed: int = 42
    pretrained_transfer_mode: str = "finetune"
    mixup_enabled: bool = False
    mixup_alpha: float = 0.4
    mixup_weight: float = 1.0
    mixup_layer: str = "embedding"
    modality_dropout_prob: float = 0.0
    state_dropout_prob: float = 0.0
    encoder_lr_multiplier: float = 1.0
    transfer_head_lr_multiplier: float = 1.0
    freeze_pretrained_encoder_epochs: int = 0
    ssl_bridge_enabled: bool = False
    ssl_consistency_weight: float = 0.0
    ssl_two_head_enabled: bool = False
    ssl_aux_head_weight: float = 0.5
    ssl_fusion_weight: float = 0.5
    ssl_residual_enabled: bool = False
    ssl_residual_weight: float = 0.14
    bridge_residual_weight: float = 0.04
    ssl_residual_random_main: bool = False
    ssl_residual_main_loss_only: bool = False
    ssl_residual_preserve_main: bool = False
    ssl_residual_select_weights_on_val: bool = False
    ssl_residual_weight_candidates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)
    bridge_residual_weight_candidates: tuple[float, ...] = (0.0, 0.02, 0.04, 0.08)
    ssl_residual_weight_selection_metric: str = "balanced_accuracy"
    ssl_residual_logit_preservation_weight: float = 0.0
    ssl_residual_logit_delta_enabled: bool = False
    ssl_residual_logit_delta_scale: float = 0.5
    embedding_adapter_dim: int = 0
    embedding_adapter_scale: float = 1.0
    eeg_summary_features_enabled: bool = False
    eeg_summary_feature_names: tuple[str, ...] = ()
    eeg_summary_embedding_dim: int = 0
    qeeg_features_enabled: bool = False
    qeeg_feature_names: tuple[str, ...] = ("qeeg_ec_global_slow_fast_bsi",)
    qeeg_primary_only: bool = True
    qeeg_hidden_dim: int = 4
    qeeg_clip_value: float | None = 3.0


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(device)


def load_supervised_feature_records(
    config: PathConfig,
    label_table: pd.DataFrame,
    feature_kind: str = "psd",
    eeg_summary_features_enabled: bool = False,
    qeeg_features_enabled: bool = False,
    qeeg_feature_names: tuple[str, ...] = ("qeeg_ec_global_slow_fast_bsi",),
) -> list[SupervisedFeatureRecord]:
    """Load full EO/EC tensors for supervised deep-learning LOSO training."""

    feature_kind = _normalize_feature_kind(feature_kind)
    rows = label_table.loc[:, ["subject_id", "label"]].copy()
    rows["subject_id"] = rows["subject_id"].map(normalize_subject_id)
    records: list[SupervisedFeatureRecord] = []
    branches = branches_for_feature_kind(feature_kind)

    for row in rows.itertuples(index=False):
        subject_id = normalize_subject_id(row.subject_id)
        modalities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for branch in branches:
            eo = _load_branch_array(config.output_root, subject_id, "EO", branch)
            ec = _load_branch_array(config.output_root, subject_id, "EC", branch)
            _validate_feature_shape(eo, branch, subject_id, "EO")
            _validate_feature_shape(ec, branch, subject_id, "EC")
            modalities[branch] = (
                eo.astype(np.float32, copy=False),
                ec.astype(np.float32, copy=False),
            )
        first_eo, first_ec = modalities[branches[0]]
        eeg_summary = None
        eeg_summary_feature_names: tuple[str, ...] = ()
        if eeg_summary_features_enabled:
            summary = compute_subject_eeg_summary_features(config.output_root, subject_id)
            eeg_summary = summary.values
            eeg_summary_feature_names = summary.feature_names
        qeeg_features = None
        selected_qeeg_feature_names: tuple[str, ...] = ()
        if qeeg_features_enabled:
            qeeg_vector = compute_subject_qeeg_slowing_features(config.output_root, subject_id)
            qeeg_features = _select_named_feature_values(
                qeeg_vector.values,
                qeeg_vector.feature_names,
                qeeg_feature_names,
                feature_family="qEEG",
            )
            selected_qeeg_feature_names = tuple(qeeg_feature_names)
        records.append(
            SupervisedFeatureRecord(
                subject_id=subject_id,
                label=int(row.label),
                eo=first_eo,
                ec=first_ec,
                modalities=modalities,
                eeg_summary=eeg_summary,
                eeg_summary_feature_names=eeg_summary_feature_names,
                qeeg_features=qeeg_features,
                qeeg_feature_names=selected_qeeg_feature_names,
            )
        )
    return records


def run_loso_supervised(
    records: Iterable[SupervisedFeatureRecord],
    training_config: SupervisedTrainingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, metrics, _ = run_loso_supervised_with_history(records, training_config)
    return predictions, metrics


def run_loso_supervised_with_history(
    records: Iterable[SupervisedFeatureRecord],
    training_config: SupervisedTrainingConfig | None = None,
    *,
    pretrained_state_by_test_subject: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train a small supervised neural model with patient-level LOSO folds."""

    resolved = training_config or SupervisedTrainingConfig()
    _validate_architecture_feature_kind(resolved)
    _seed_everything(resolved.seed)
    device = resolve_device(resolved.device)
    records = list(records)
    if len(records) < 2:
        raise ValueError("Supervised LOSO requires at least 2 records.")
    if resolved.eeg_summary_features_enabled and not resolved.eeg_summary_feature_names:
        summary_names = _eeg_summary_feature_names_from_records(records)
        resolved = replace(resolved, eeg_summary_feature_names=summary_names)
        _validate_architecture_feature_kind(resolved)
    if resolved.qeeg_features_enabled and not resolved.qeeg_feature_names:
        qeeg_names = _qeeg_feature_names_from_records(records)
        resolved = replace(resolved, qeeg_feature_names=qeeg_names)
        _validate_architecture_feature_kind(resolved)
    if resolved.ssl_bridge_enabled and not pretrained_state_by_test_subject:
        raise ValueError("ssl_bridge_enabled requires pretrained state by LOSO test subject.")
    if resolved.ssl_two_head_enabled and not pretrained_state_by_test_subject:
        raise ValueError("ssl_two_head_enabled requires pretrained state by LOSO test subject.")
    if resolved.ssl_residual_enabled and not pretrained_state_by_test_subject:
        raise ValueError("ssl_residual_enabled requires pretrained state by LOSO test subject.")

    subject_ids = [record.subject_id for record in records]
    record_by_subject = {record.subject_id: record for record in records}
    folds = make_loso_folds(subject_ids)
    rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    model_name = f"{resolved.architecture}_{resolved.feature_kind}_{resolved.fusion}_{resolved.encoder_kind}"
    if resolved.ssl_bridge_enabled:
        model_name = f"{model_name}_sslbridge"
    if resolved.ssl_two_head_enabled:
        model_name = f"{model_name}_ssltwohead"
        if resolved.ssl_residual_enabled:
            model_name = f"{model_name}_sslresidual"
    if resolved.ssl_residual_enabled and resolved.ssl_residual_logit_delta_enabled:
        model_name = f"{model_name}_logitdelta"
    if resolved.eeg_summary_features_enabled:
        model_name = f"{model_name}_eegsummary"
    if resolved.qeeg_features_enabled:
        model_name = f"{model_name}_qeeg"

    for fold in folds:
        selected_residual_weights: dict[str, float | str] = {
            "selected_ssl_residual_weight": float(resolved.ssl_residual_weight),
            "selected_bridge_residual_weight": float(resolved.bridge_residual_weight),
            "selected_trainable_weight": float(1.0 - resolved.ssl_residual_weight - resolved.bridge_residual_weight),
            "selected_weight_metric": resolved.ssl_residual_weight_selection_metric,
            "selected_weight_metric_value": float("nan"),
            "selected_weight_val_brier_score": float("nan"),
        }
        train_subjects = list(fold.train_subject_ids)
        fit_subjects, val_subjects = _train_validation_subjects(train_subjects, fold.fold_index)
        scaler = _fit_state_scaler(record_by_subject[subject_id] for subject_id in fit_subjects)
        model = _build_model(resolved).to(device)
        pretrained_loaded = _load_fold_pretrained_state(
            model,
            fold.test_subject_id,
            pretrained_state_by_test_subject,
        )
        transfer_mode = resolved.pretrained_transfer_mode if pretrained_loaded else "none"
        if pretrained_loaded:
            _apply_pretrained_transfer_mode(model, resolved.pretrained_transfer_mode)
        if resolved.finetune_schedule == "staged_sam_swa":
            _configure_finetune_stage(model, resolved, stage="stage1")
        elif pretrained_loaded and _uses_pretrained_encoder_warmup(resolved):
            _set_encoder_requires_grad(model, False)
        current_stage = "stage1" if resolved.finetune_schedule == "staged_sam_swa" else "standard"
        optimizer_config = _optimizer_config_for_finetune_stage(resolved, current_stage)
        optimizer = _build_transfer_optimizer(model, optimizer_config)
        scheduler = build_reduce_on_plateau(optimizer, mode="min", patience=max(1, resolved.patience // 2))
        early_stopping = EarlyStopping(patience=resolved.patience, mode="min")
        swa_model = build_swa_model(model) if resolved.use_swa else None
        swa_updates = 0

        train_batch = _make_batch(
            (record_by_subject[subject_id] for subject_id in fit_subjects),
            scaler,
            device,
            resolved.architecture,
        )
        val_batch = _make_batch(
            (record_by_subject[subject_id] for subject_id in val_subjects),
            scaler,
            device,
            resolved.architecture,
        )
        if resolved.ssl_residual_preserve_main:
            main_model, main_loss_rows = _train_preserved_main_fold_model(
                train_batch,
                val_batch,
                resolved,
                fold_index=fold.fold_index,
                test_subject_id=fold.test_subject_id,
                fit_subjects=fit_subjects,
                val_subjects=val_subjects,
            )
            _load_frozen_main_state(model, main_model.state_dict())
            loss_rows.extend(main_loss_rows)
            optimizer_config = _optimizer_config_for_finetune_stage(resolved, current_stage)
            optimizer = _build_transfer_optimizer(model, optimizer_config)
            scheduler = build_reduce_on_plateau(optimizer, mode="min", patience=max(1, resolved.patience // 2))
            early_stopping = EarlyStopping(patience=resolved.patience, mode="min")

        for epoch in range(1, resolved.epochs + 1):
            if (
                resolved.finetune_schedule == "staged_sam_swa"
                and current_stage == "stage1"
                and epoch == _staged_freeze_epochs(resolved) + 1
            ):
                current_stage = "stage2"
                _configure_finetune_stage(model, resolved, stage="stage2")
                optimizer_config = _optimizer_config_for_finetune_stage(resolved, current_stage)
                optimizer = _build_transfer_optimizer(model, optimizer_config)
                scheduler = build_reduce_on_plateau(optimizer, mode="min", patience=max(1, resolved.patience // 2))
            if pretrained_loaded and _uses_pretrained_encoder_warmup(resolved) and epoch == resolved.freeze_pretrained_encoder_epochs + 1:
                _set_encoder_requires_grad(model, True)
                optimizer_config = _optimizer_config_for_finetune_stage(resolved, current_stage)
                optimizer = _build_transfer_optimizer(model, optimizer_config)
                scheduler = build_reduce_on_plateau(optimizer, mode="min", patience=max(1, resolved.patience // 2))
            if swa_model is not None and epoch >= resolved.swa_start_epoch:
                for group in optimizer.param_groups:
                    group["lr"] = resolved.swa_lr
            model.train()
            if isinstance(optimizer, SAM):
                def closure() -> torch.Tensor:
                    optimizer.zero_grad()
                    closure_loss = _compute_supervised_loss(model, train_batch, resolved, training=True)
                    closure_loss.backward()
                    return closure_loss

                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                loss = _compute_supervised_loss(model, train_batch, resolved, training=True)
                loss.backward()
                optimizer.step()
            if swa_model is not None and epoch >= resolved.swa_start_epoch:
                swa_model.update_parameters(model)
                swa_updates += 1

            train_loss = float(loss.detach().cpu().item())
            val_loss = _evaluate_loss(model, val_batch, resolved)
            scheduler.step(val_loss)
            is_best_epoch = early_stopping.best_score is None or val_loss < early_stopping.best_score - early_stopping.min_delta
            stopped_early = early_stopping.step(val_loss, model)
            loss_rows.append(
                {
                    "model": model_name,
                    "fold_index": fold.fold_index,
                    "test_subject_id": fold.test_subject_id,
                    "fit_subject_ids": ";".join(fit_subjects),
                    "val_subject_ids": ";".join(val_subjects),
                    "epoch": epoch,
                    "training_phase": "ssl_residual" if resolved.ssl_residual_preserve_main else "supervised",
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "learning_rates": _format_optimizer_learning_rates(optimizer),
                    "weight_decay": resolved.weight_decay,
                    "optimizer_name": optimizer_config.optimizer_name,
                    "finetune_schedule": resolved.finetune_schedule,
                    "finetune_stage": current_stage,
                    "use_swa": resolved.use_swa,
                    "swa_start_epoch": resolved.swa_start_epoch,
                    "swa_lr": resolved.swa_lr,
                    "sam_rho": resolved.sam_rho,
                    "encoder_lr": resolved.encoder_lr,
                    "head_lr": resolved.head_lr,
                    "freeze_encoder_epochs": resolved.freeze_encoder_epochs,
                    "freeze_conv_backbone": resolved.freeze_conv_backbone,
                    "loss_name": resolved.loss_name,
                    "positive_class_weight": resolved.positive_class_weight,
                    "negative_class_weight": resolved.negative_class_weight,
                    "focal_gamma_pos": resolved.focal_gamma_pos,
                    "focal_gamma_neg": resolved.focal_gamma_neg,
                    "fp_margin": resolved.fp_margin,
                    "fp_penalty_weight": resolved.fp_penalty_weight,
                    "is_best_epoch": bool(is_best_epoch),
                    "stopped_early": bool(stopped_early),
                    "architecture": resolved.architecture,
                    "feature_kind": resolved.feature_kind,
                    "fusion": resolved.fusion,
                    "encoder_kind": resolved.encoder_kind,
                    "pretrained": bool(pretrained_loaded),
                    "pretrained_transfer_mode": transfer_mode,
                    "mixup_enabled": resolved.mixup_enabled,
                    "mixup_alpha": resolved.mixup_alpha,
                    "mixup_weight": resolved.mixup_weight,
                    "mixup_layer": resolved.mixup_layer,
                    "modality_dropout_prob": resolved.modality_dropout_prob,
                    "state_dropout_prob": resolved.state_dropout_prob,
                    "encoder_lr_multiplier": resolved.encoder_lr_multiplier,
                    "transfer_head_lr_multiplier": resolved.transfer_head_lr_multiplier,
                    "freeze_pretrained_encoder_epochs": resolved.freeze_pretrained_encoder_epochs,
                    "encoder_trainable": _encoder_trainable(model),
                    "ssl_bridge_enabled": resolved.ssl_bridge_enabled,
                    "ssl_consistency_weight": resolved.ssl_consistency_weight,
                    "ssl_two_head_enabled": resolved.ssl_two_head_enabled,
                    "ssl_aux_head_weight": resolved.ssl_aux_head_weight,
                    "ssl_fusion_weight": resolved.ssl_fusion_weight,
                    "ssl_residual_enabled": resolved.ssl_residual_enabled,
                    "ssl_residual_weight": resolved.ssl_residual_weight,
                    "bridge_residual_weight": resolved.bridge_residual_weight,
                    "ssl_residual_random_main": resolved.ssl_residual_random_main,
                    "ssl_residual_main_loss_only": resolved.ssl_residual_main_loss_only,
                    "ssl_residual_preserve_main": resolved.ssl_residual_preserve_main,
                    "ssl_residual_select_weights_on_val": resolved.ssl_residual_select_weights_on_val,
                    "ssl_residual_logit_preservation_weight": resolved.ssl_residual_logit_preservation_weight,
                    "ssl_residual_logit_delta_enabled": resolved.ssl_residual_logit_delta_enabled,
                    "ssl_residual_logit_delta_scale": resolved.ssl_residual_logit_delta_scale,
                    **selected_residual_weights,
                    "embedding_adapter_dim": resolved.embedding_adapter_dim,
                    "embedding_adapter_scale": resolved.embedding_adapter_scale,
                    "eeg_summary_features_enabled": resolved.eeg_summary_features_enabled,
                    "eeg_summary_embedding_dim": resolved.eeg_summary_embedding_dim,
                }
            )
            if stopped_early:
                break
        evaluation_model: nn.Module = model
        if swa_model is not None and swa_updates > 0:
            update_swa_batch_norm([train_batch], swa_model)
            evaluation_model = swa_model
        else:
            early_stopping.restore_best_weights(model)
        if resolved.ssl_residual_select_weights_on_val:
            selected_residual_weights = _select_ssl_residual_weights_on_val(evaluation_model, val_batch, resolved)
            for loss_row in loss_rows:
                if loss_row.get("fold_index") == fold.fold_index:
                    loss_row.update(selected_residual_weights)

        test_record = record_by_subject[fold.test_subject_id]
        test_batch = _make_batch([test_record], scaler, device, resolved.architecture)
        evaluation_model.eval()
        with torch.no_grad():
            aux_scores: dict[str, float] = {}
            if resolved.ssl_residual_enabled:
                probabilities_and_aux = evaluation_model(test_batch, return_aux=True)
                if not isinstance(probabilities_and_aux, tuple) or len(probabilities_and_aux) != 2:
                    raise ValueError("SSL residual model must return auxiliary probabilities at test time.")
                probabilities, aux = probabilities_and_aux
                probability = float(probabilities.detach().cpu().numpy()[0, 0])
                aux_key_to_column = {
                    "trainable_probabilities": "trainable_y_score",
                    "ssl_probabilities": "ssl_residual_y_score",
                    "bridge_probabilities": "bridge_residual_y_score",
                    "logit_delta": "ssl_logit_delta",
                }
                for aux_key, column in aux_key_to_column.items():
                    if aux_key in aux:
                        aux_scores[column] = float(aux[aux_key].detach().cpu().numpy()[0, 0])
            elif resolved.eeg_summary_features_enabled:
                probabilities_and_aux = evaluation_model(test_batch, return_aux=True)
                if not isinstance(probabilities_and_aux, tuple) or len(probabilities_and_aux) != 2:
                    raise ValueError("EEG summary model must return auxiliary outputs at test time.")
                probabilities, aux = probabilities_and_aux
                probability = float(probabilities.detach().cpu().numpy()[0, 0])
                aux_scores.update(_eeg_summary_aux_scores(aux, resolved))
            else:
                probability = float(_model_probabilities(evaluation_model, test_batch).detach().cpu().numpy()[0, 0])
        row = {
                "model": model_name,
                "fold_index": fold.fold_index,
                "subject_id": fold.test_subject_id,
                "y_true": int(test_record.label),
                "y_score": probability,
                "y_pred": int(probability >= 0.5),
                "architecture": resolved.architecture,
                "feature_kind": resolved.feature_kind,
                "fusion": resolved.fusion,
                "encoder_kind": resolved.encoder_kind,
                "pretrained": bool(pretrained_loaded),
                "pretrained_transfer_mode": transfer_mode,
                "status": "trained",
                "skip_reason": "",
                "learning_rate": resolved.lr,
                "weight_decay": resolved.weight_decay,
                "loss_name": resolved.loss_name,
                "positive_class_weight": resolved.positive_class_weight,
                "negative_class_weight": resolved.negative_class_weight,
                "focal_gamma_pos": resolved.focal_gamma_pos,
                "focal_gamma_neg": resolved.focal_gamma_neg,
                "fp_margin": resolved.fp_margin,
                "fp_penalty_weight": resolved.fp_penalty_weight,
                "embedding_dim": resolved.embedding_dim,
                "dropout": resolved.dropout,
                "seed": resolved.seed,
                "mixup_enabled": resolved.mixup_enabled,
                "mixup_alpha": resolved.mixup_alpha,
                "mixup_weight": resolved.mixup_weight,
                "mixup_layer": resolved.mixup_layer,
                "modality_dropout_prob": resolved.modality_dropout_prob,
                "state_dropout_prob": resolved.state_dropout_prob,
                "encoder_lr_multiplier": resolved.encoder_lr_multiplier,
                "transfer_head_lr_multiplier": resolved.transfer_head_lr_multiplier,
                "freeze_pretrained_encoder_epochs": resolved.freeze_pretrained_encoder_epochs,
                "ssl_bridge_enabled": resolved.ssl_bridge_enabled,
                "ssl_consistency_weight": resolved.ssl_consistency_weight,
                "ssl_two_head_enabled": resolved.ssl_two_head_enabled,
                "ssl_aux_head_weight": resolved.ssl_aux_head_weight,
                "ssl_fusion_weight": resolved.ssl_fusion_weight,
                "ssl_residual_enabled": resolved.ssl_residual_enabled,
                "ssl_residual_weight": resolved.ssl_residual_weight,
                "bridge_residual_weight": resolved.bridge_residual_weight,
                "ssl_residual_random_main": resolved.ssl_residual_random_main,
                "ssl_residual_main_loss_only": resolved.ssl_residual_main_loss_only,
                "ssl_residual_preserve_main": resolved.ssl_residual_preserve_main,
                "ssl_residual_select_weights_on_val": resolved.ssl_residual_select_weights_on_val,
                "ssl_residual_logit_preservation_weight": resolved.ssl_residual_logit_preservation_weight,
                "ssl_residual_logit_delta_enabled": resolved.ssl_residual_logit_delta_enabled,
                "ssl_residual_logit_delta_scale": resolved.ssl_residual_logit_delta_scale,
                **selected_residual_weights,
                "embedding_adapter_dim": resolved.embedding_adapter_dim,
                "embedding_adapter_scale": resolved.embedding_adapter_scale,
                "eeg_summary_features_enabled": resolved.eeg_summary_features_enabled,
                "eeg_summary_embedding_dim": resolved.eeg_summary_embedding_dim,
            }
        row.update(aux_scores)
        rows.append(row)

    sample_predictions = pd.DataFrame(rows)
    predictions = aggregate_patient_probabilities(sample_predictions)
    for column in (
        "model",
        "architecture",
        "feature_kind",
        "fusion",
        "encoder_kind",
        "pretrained",
        "pretrained_transfer_mode",
        "status",
        "skip_reason",
        "ssl_bridge_enabled",
        "ssl_consistency_weight",
        "ssl_two_head_enabled",
        "ssl_aux_head_weight",
        "ssl_fusion_weight",
        "ssl_residual_enabled",
        "ssl_residual_weight",
        "bridge_residual_weight",
        "ssl_residual_random_main",
        "ssl_residual_main_loss_only",
        "ssl_residual_preserve_main",
        "ssl_residual_select_weights_on_val",
        "ssl_residual_logit_preservation_weight",
        "ssl_residual_logit_delta_enabled",
        "ssl_residual_logit_delta_scale",
        "embedding_adapter_dim",
        "embedding_adapter_scale",
        "eeg_summary_features_enabled",
        "eeg_summary_embedding_dim",
    ):
        predictions[column] = sample_predictions[column].iloc[0]
    for column in (
        "selected_ssl_residual_weight",
        "selected_bridge_residual_weight",
        "selected_trainable_weight",
        "selected_weight_metric",
        "selected_weight_metric_value",
        "selected_weight_val_brier_score",
    ):
        if column in sample_predictions.columns and column not in predictions.columns:
            aggregation = "first" if column == "selected_weight_metric" else "mean"
            predictions[column] = (
                sample_predictions.groupby("subject_id")[column].agg(aggregation).reindex(predictions["subject_id"]).to_numpy()
            )
    metric_values = binary_classification_metrics(
        predictions["y_true"].to_numpy(dtype=int),
        predictions["y_score"].to_numpy(dtype=float),
    )
    metrics = pd.DataFrame(
        [
            {
                "model": model_name,
                "architecture": resolved.architecture,
                "feature_kind": resolved.feature_kind,
                "fusion": resolved.fusion,
                "encoder_kind": resolved.encoder_kind,
                "pretrained": bool(
                    pretrained_state_by_test_subject is not None
                    and len(pretrained_state_by_test_subject) > 0
                ),
                "pretrained_transfer_mode": resolved.pretrained_transfer_mode
                if pretrained_state_by_test_subject
                else "none",
                "status": "trained",
                "skip_reason": "",
                "learning_rate": resolved.lr,
                "weight_decay": resolved.weight_decay,
                "loss_name": resolved.loss_name,
                "positive_class_weight": resolved.positive_class_weight,
                "negative_class_weight": resolved.negative_class_weight,
                "focal_gamma_pos": resolved.focal_gamma_pos,
                "focal_gamma_neg": resolved.focal_gamma_neg,
                "fp_margin": resolved.fp_margin,
                "fp_penalty_weight": resolved.fp_penalty_weight,
                "embedding_dim": resolved.embedding_dim,
                "dropout": resolved.dropout,
                "seed": resolved.seed,
                "mixup_enabled": resolved.mixup_enabled,
                "mixup_alpha": resolved.mixup_alpha,
                "mixup_weight": resolved.mixup_weight,
                "mixup_layer": resolved.mixup_layer,
                "modality_dropout_prob": resolved.modality_dropout_prob,
                "state_dropout_prob": resolved.state_dropout_prob,
                "encoder_lr_multiplier": resolved.encoder_lr_multiplier,
                "transfer_head_lr_multiplier": resolved.transfer_head_lr_multiplier,
                "freeze_pretrained_encoder_epochs": resolved.freeze_pretrained_encoder_epochs,
                "ssl_bridge_enabled": resolved.ssl_bridge_enabled,
                "ssl_consistency_weight": resolved.ssl_consistency_weight,
                "ssl_two_head_enabled": resolved.ssl_two_head_enabled,
                "ssl_aux_head_weight": resolved.ssl_aux_head_weight,
                "ssl_fusion_weight": resolved.ssl_fusion_weight,
                "ssl_residual_enabled": resolved.ssl_residual_enabled,
                "ssl_residual_weight": resolved.ssl_residual_weight,
                "bridge_residual_weight": resolved.bridge_residual_weight,
                "ssl_residual_random_main": resolved.ssl_residual_random_main,
                "ssl_residual_main_loss_only": resolved.ssl_residual_main_loss_only,
                "ssl_residual_preserve_main": resolved.ssl_residual_preserve_main,
                "ssl_residual_select_weights_on_val": resolved.ssl_residual_select_weights_on_val,
                "ssl_residual_logit_preservation_weight": resolved.ssl_residual_logit_preservation_weight,
                "ssl_residual_logit_delta_enabled": resolved.ssl_residual_logit_delta_enabled,
                "ssl_residual_logit_delta_scale": resolved.ssl_residual_logit_delta_scale,
                "selected_ssl_residual_weight": float(sample_predictions["selected_ssl_residual_weight"].mean()),
                "selected_bridge_residual_weight": float(sample_predictions["selected_bridge_residual_weight"].mean()),
                "selected_trainable_weight": float(sample_predictions["selected_trainable_weight"].mean()),
                "selected_weight_metric": resolved.ssl_residual_weight_selection_metric,
                "selected_weight_metric_value": float(sample_predictions["selected_weight_metric_value"].mean()),
                "selected_weight_val_brier_score": float(sample_predictions["selected_weight_val_brier_score"].mean()),
                "embedding_adapter_dim": resolved.embedding_adapter_dim,
                "embedding_adapter_scale": resolved.embedding_adapter_scale,
                "eeg_summary_features_enabled": resolved.eeg_summary_features_enabled,
                "eeg_summary_embedding_dim": resolved.eeg_summary_embedding_dim,
                **metric_values,
            }
        ]
    )
    loss_history = pd.DataFrame(loss_rows)
    return predictions, metrics, loss_history


def _build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: SupervisedTrainingConfig,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)


def _build_transfer_optimizer(
    model: nn.Module,
    config: SupervisedTrainingConfig,
) -> torch.optim.Optimizer:
    encoder_lr = config.encoder_lr if config.encoder_lr is not None else config.lr * config.encoder_lr_multiplier
    head_lr = config.head_lr if config.head_lr is not None else config.lr
    groups: dict[str, dict[str, Any]] = {
        "encoder": {"params": [], "lr": encoder_lr},
        "transfer_head": {"params": [], "lr": head_lr * config.transfer_head_lr_multiplier},
        "classifier": {"params": [], "lr": head_lr},
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_encoder_parameter_name(name):
            groups["encoder"]["params"].append(parameter)
        elif name.startswith("branch_models."):
            groups["transfer_head"]["params"].append(parameter)
        else:
            groups["classifier"]["params"].append(parameter)

    param_groups = [
        {"name": name, "params": group["params"], "lr": group["lr"]}
        for name, group in groups.items()
        if group["params"]
    ]
    if not param_groups:
        raise ValueError("No trainable model parameters remain after applying transfer mode.")
    if config.optimizer_name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
    if config.optimizer_name == "adamw":
        return build_adamw_optimizer(param_groups, weight_decay=config.weight_decay)
    if config.optimizer_name == "sam_adamw":
        return SAM(
            param_groups,
            torch.optim.AdamW,
            rho=config.sam_rho,
            weight_decay=config.weight_decay,
        )
    raise ValueError("optimizer_name must be 'adam', 'adamw', or 'sam_adamw'.")


def _train_preserved_main_fold_model(
    train_batch: dict[str, torch.Tensor],
    val_batch: dict[str, torch.Tensor],
    config: SupervisedTrainingConfig,
    *,
    fold_index: int,
    test_subject_id: str,
    fit_subjects: list[str],
    val_subjects: list[str],
) -> tuple[nn.Module, list[dict[str, Any]]]:
    main_config = replace(
        config,
        ssl_bridge_enabled=False,
        ssl_consistency_weight=0.0,
        ssl_two_head_enabled=False,
        ssl_aux_head_weight=0.0,
        ssl_fusion_weight=0.5,
        ssl_residual_enabled=False,
        ssl_residual_random_main=False,
        ssl_residual_main_loss_only=False,
        ssl_residual_preserve_main=False,
        ssl_residual_select_weights_on_val=False,
        ssl_residual_logit_preservation_weight=0.0,
        ssl_residual_logit_delta_enabled=False,
        ssl_residual_logit_delta_scale=0.5,
        embedding_adapter_dim=0,
        embedding_adapter_scale=1.0,
    )
    model = _build_model(main_config)
    device = next(iter(train_batch.values())).device
    model = model.to(device)
    optimizer = _build_transfer_optimizer(model, main_config)
    scheduler = build_reduce_on_plateau(optimizer, mode="min", patience=max(1, main_config.patience // 2))
    early_stopping = EarlyStopping(patience=main_config.patience, mode="min")
    rows: list[dict[str, Any]] = []
    model_name = f"{main_config.architecture}_{main_config.feature_kind}_{main_config.fusion}_{main_config.encoder_kind}_preserved_main"

    for epoch in range(1, main_config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = _compute_supervised_loss(model, train_batch, main_config, training=True)
        loss.backward()
        optimizer.step()

        train_loss = float(loss.detach().cpu().item())
        val_loss = _evaluate_loss(model, val_batch, main_config)
        scheduler.step(val_loss)
        is_best_epoch = early_stopping.best_score is None or val_loss < early_stopping.best_score - early_stopping.min_delta
        stopped_early = early_stopping.step(val_loss, model)
        rows.append(
            {
                "model": model_name,
                "fold_index": fold_index,
                "test_subject_id": test_subject_id,
                "fit_subject_ids": ";".join(fit_subjects),
                "val_subject_ids": ";".join(val_subjects),
                "epoch": epoch,
                "training_phase": "preserved_main",
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "learning_rates": _format_optimizer_learning_rates(optimizer),
                "weight_decay": main_config.weight_decay,
                "loss_name": main_config.loss_name,
                "positive_class_weight": main_config.positive_class_weight,
                "negative_class_weight": main_config.negative_class_weight,
                "focal_gamma_pos": main_config.focal_gamma_pos,
                "focal_gamma_neg": main_config.focal_gamma_neg,
                "fp_margin": main_config.fp_margin,
                "fp_penalty_weight": main_config.fp_penalty_weight,
                "is_best_epoch": bool(is_best_epoch),
                "stopped_early": bool(stopped_early),
                "architecture": main_config.architecture,
                "feature_kind": main_config.feature_kind,
                "fusion": main_config.fusion,
                "encoder_kind": main_config.encoder_kind,
                "pretrained": False,
                "pretrained_transfer_mode": "none",
                "encoder_trainable": _encoder_trainable(model),
                "ssl_residual_preserve_main": True,
            }
        )
        if stopped_early:
            break

    early_stopping.restore_best_weights(model)
    model.eval()
    return model, rows


def _load_frozen_main_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    if not callable(getattr(model, "load_frozen_main_state", None)):
        raise ValueError("ssl_residual_preserve_main requires a model with load_frozen_main_state.")
    incompatible = model.load_frozen_main_state(dict(state))
    if incompatible.unexpected_keys:
        joined = ", ".join(incompatible.unexpected_keys)
        raise ValueError(f"Preserved main state contains unexpected key(s): {joined}")


def _select_ssl_residual_weights_on_val(
    model: nn.Module,
    val_batch: dict[str, torch.Tensor],
    config: SupervisedTrainingConfig,
) -> dict[str, float | str]:
    if not callable(getattr(model, "set_residual_weights", None)):
        raise ValueError("ssl_residual_select_weights_on_val requires a model with set_residual_weights.")
    model.eval()
    with torch.no_grad():
        probabilities_and_aux = model(val_batch, return_aux=True)
    if not isinstance(probabilities_and_aux, tuple) or len(probabilities_and_aux) != 2:
        raise ValueError("SSL residual weight selection requires auxiliary probabilities.")
    _, aux = probabilities_and_aux
    required_aux = {"trainable_probabilities", "ssl_probabilities", "bridge_probabilities"}
    missing = required_aux - set(aux)
    if missing:
        raise ValueError(f"SSL residual weight selection missing auxiliary key(s): {', '.join(sorted(missing))}.")

    targets = val_batch["y"].detach().cpu().numpy().reshape(-1).astype(int)
    main_scores = aux["trainable_probabilities"].detach().cpu().numpy().reshape(-1)
    ssl_scores = aux["ssl_probabilities"].detach().cpu().numpy().reshape(-1)
    bridge_scores = aux["bridge_probabilities"].detach().cpu().numpy().reshape(-1)
    best: dict[str, float | str] | None = None
    best_key: tuple[float, float, float] | None = None
    metric_name = config.ssl_residual_weight_selection_metric

    for ssl_weight in config.ssl_residual_weight_candidates:
        for bridge_weight in config.bridge_residual_weight_candidates:
            if ssl_weight + bridge_weight > 1:
                continue
            trainable_weight = 1.0 - ssl_weight - bridge_weight
            scores = trainable_weight * main_scores + ssl_weight * ssl_scores + bridge_weight * bridge_scores
            metrics = binary_classification_metrics(targets, scores)
            metric_value = float(metrics[metric_name])
            if not np.isfinite(metric_value):
                continue
            brier = float(metrics["brier_score"])
            direction = -1.0 if metric_name == "brier_score" else 1.0
            candidate_key = (direction * metric_value, -brier, -abs(ssl_weight + bridge_weight))
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best = {
                    "selected_ssl_residual_weight": float(ssl_weight),
                    "selected_bridge_residual_weight": float(bridge_weight),
                    "selected_trainable_weight": float(trainable_weight),
                    "selected_weight_metric": metric_name,
                    "selected_weight_metric_value": metric_value,
                    "selected_weight_val_brier_score": brier,
                }

    if best is None:
        raise ValueError("No finite validation metric was available for SSL residual weight selection.")
    model.set_residual_weights(
        ssl_residual_weight=float(best["selected_ssl_residual_weight"]),
        bridge_residual_weight=float(best["selected_bridge_residual_weight"]),
    )
    return best


def aggregate_patient_probabilities(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Aggregate sample-level predictions to patient-level mean probabilities."""

    required = {"subject_id", "y_true", "y_score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required column(s): {', '.join(sorted(missing))}")
    aggregations: dict[str, tuple[str, str]] = {
        "y_true": ("y_true", "first"),
        "y_score": ("y_score", "mean"),
        "fold_index": ("fold_index", "first") if "fold_index" in predictions.columns else ("subject_id", "size"),
    }
    for column in predictions.columns:
        if column.endswith("_y_score") and column != "y_score":
            aggregations[column] = (column, "mean")
        if column.startswith("modality_weight_") or column.startswith("eeg_summary_importance_"):
            aggregations[column] = (column, "mean")

    grouped = (
        predictions.groupby("subject_id", as_index=False)
        .agg(**aggregations)
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    grouped["y_true"] = grouped["y_true"].astype(int)
    grouped["y_pred"] = (grouped["y_score"] >= threshold).astype(int)
    return grouped


def write_dl_outputs(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_root: str | Path,
    run_name: str | None = None,
) -> tuple[Path, Path]:
    root = Path(output_root)
    if run_name is None:
        prediction_name = "dl_loso_predictions.csv"
        metric_name = "dl_model_comparison.csv"
    else:
        token = _safe_filename_token(run_name)
        prediction_name = f"dl_loso_predictions_{token}.csv"
        metric_name = f"dl_model_comparison_{token}.csv"
    prediction_path = root / "results" / "predictions" / prediction_name
    metric_path = root / "results" / "metrics" / metric_name
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(prediction_path, index=False)
    metrics_df.to_csv(metric_path, index=False)
    return prediction_path, metric_path


def write_loss_history_outputs(
    loss_history_df: pd.DataFrame,
    output_root: str | Path,
    run_name: str | None = None,
) -> tuple[Path, Path]:
    """Write per-fold epoch loss values and a loss-vs-epoch curve."""

    if loss_history_df.empty:
        raise ValueError("loss_history_df is empty.")
    if "model" not in loss_history_df.columns:
        raise ValueError("loss_history_df must include a model column.")
    model_names = loss_history_df["model"].dropna().astype(str).unique()
    if len(model_names) != 1:
        raise ValueError("loss_history_df must contain exactly one model.")

    root = Path(output_root)
    output_token = _safe_filename_token(run_name or model_names[0])
    history_path = root / "results" / "training_logs" / f"dl_loss_history_{output_token}.csv"
    figure_path = root / "results" / "figures" / f"dl_loss_curve_{output_token}.png"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    loss_history_df.to_csv(history_path, index=False)
    _plot_loss_history(loss_history_df, figure_path)
    return history_path, figure_path


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    if not token:
        raise ValueError("model name cannot be converted to a safe filename.")
    return token


def _plot_loss_history(loss_history_df: pd.DataFrame, figure_path: Path) -> None:
    required = {"model", "fold_index", "epoch", "train_loss", "val_loss"}
    missing = required - set(loss_history_df.columns)
    if missing:
        raise ValueError(f"loss_history_df missing required column(s): {', '.join(sorted(missing))}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = loss_history_df.sort_values(["fold_index", "epoch"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, fold_rows in ordered.groupby("fold_index"):
        ax.plot(fold_rows["epoch"], fold_rows["train_loss"], color="#1f77b4", alpha=0.18, linewidth=0.8)
        ax.plot(fold_rows["epoch"], fold_rows["val_loss"], color="#ff7f0e", alpha=0.18, linewidth=0.8)

    epoch_means = ordered.groupby("epoch", as_index=False).agg(
        train_loss=("train_loss", "mean"),
        val_loss=("val_loss", "mean"),
    )
    ax.plot(epoch_means["epoch"], epoch_means["train_loss"], color="#1f77b4", linewidth=2.5, label="Train loss mean")
    ax.plot(epoch_means["epoch"], epoch_means["val_loss"], color="#ff7f0e", linewidth=2.5, label="Validation loss mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCELoss")
    ax.set_title(str(ordered["model"].iloc[0]))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def _build_model(config: SupervisedTrainingConfig) -> nn.Module:
    _validate_architecture_feature_kind(config)
    if config.architecture == "dual_state":
        return DualStateEEGModel(
            feature_kind=config.feature_kind,
            fusion=config.fusion,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            encoder_kind=config.encoder_kind,
        )
    if config.architecture == "fusion_3d":
        return Fusion3DEEGModel(
            feature_kind=config.feature_kind,
            fusion=config.fusion,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            encoder_kind=config.encoder_kind,
        )
    if config.architecture == "multimodal":
        enabled_ssl_modes = sum(
            int(value)
            for value in (
                config.ssl_bridge_enabled,
                config.ssl_two_head_enabled,
                config.ssl_residual_enabled,
            )
        )
        if enabled_ssl_modes > 1:
            raise ValueError("ssl_bridge_enabled, ssl_two_head_enabled, and ssl_residual_enabled are mutually exclusive.")
        if config.eeg_summary_features_enabled:
            if enabled_ssl_modes:
                raise ValueError("eeg_summary_features_enabled is supported on the standard multimodal CNN path only.")
            return EEGSummaryGatedMultimodalEEGModel(
                feature_kind=config.feature_kind,
                fusion=config.fusion,
                embedding_dim=config.embedding_dim,
                dropout=config.dropout,
                encoder_kind=config.encoder_kind,
                summary_input_dim=len(config.eeg_summary_feature_names),
                summary_feature_names=tuple(config.eeg_summary_feature_names),
            )
        if config.qeeg_features_enabled:
            if enabled_ssl_modes:
                raise ValueError("qeeg_features_enabled is supported on the standard multimodal CNN path only.")
            return QEEGGuidedMultimodalEEGModel(
                feature_kind=config.feature_kind,
                fusion=config.fusion,
                embedding_dim=config.embedding_dim,
                dropout=config.dropout,
                encoder_kind=config.encoder_kind,
                qeeg_input_dim=len(config.qeeg_feature_names),
                qeeg_hidden_dim=config.qeeg_hidden_dim,
                qeeg_clip_value=config.qeeg_clip_value,
                primary_qeeg_only=config.qeeg_primary_only,
            )
        if config.ssl_bridge_enabled:
            return SSLBridgeMultimodalEEGModel(
                feature_kind=config.feature_kind,
                fusion=config.fusion,
                embedding_dim=config.embedding_dim,
                dropout=config.dropout,
                encoder_kind=config.encoder_kind,
            )
        if config.ssl_two_head_enabled:
            return SSLTwoHeadMultimodalEEGModel(
                feature_kind=config.feature_kind,
                fusion=config.fusion,
                embedding_dim=config.embedding_dim,
                dropout=config.dropout,
                encoder_kind=config.encoder_kind,
                ssl_fusion_weight=config.ssl_fusion_weight,
            )
        if config.ssl_residual_enabled:
            if config.ssl_residual_preserve_main:
                if config.ssl_residual_logit_delta_enabled:
                    return FrozenMainSSLLogitDeltaMultimodalEEGModel(
                        feature_kind=config.feature_kind,
                        fusion=config.fusion,
                        embedding_dim=config.embedding_dim,
                        dropout=config.dropout,
                        encoder_kind=config.encoder_kind,
                        logit_delta_scale=config.ssl_residual_logit_delta_scale,
                    )
                return FrozenMainSSLResidualMultimodalEEGModel(
                    feature_kind=config.feature_kind,
                    fusion=config.fusion,
                    embedding_dim=config.embedding_dim,
                    dropout=config.dropout,
                    encoder_kind=config.encoder_kind,
                    ssl_residual_weight=config.ssl_residual_weight,
                    bridge_residual_weight=config.bridge_residual_weight,
                )
            return SSLResidualMultimodalEEGModel(
                feature_kind=config.feature_kind,
                fusion=config.fusion,
                embedding_dim=config.embedding_dim,
                dropout=config.dropout,
                encoder_kind=config.encoder_kind,
                ssl_residual_weight=config.ssl_residual_weight,
                bridge_residual_weight=config.bridge_residual_weight,
                preload_trainable_branch=not config.ssl_residual_random_main,
            )
        return MultimodalEEGModel(
            feature_kind=config.feature_kind,
            fusion=config.fusion,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            encoder_kind=config.encoder_kind,
            embedding_adapter_dim=config.embedding_adapter_dim,
            embedding_adapter_scale=config.embedding_adapter_scale,
        )
    raise ValueError("architecture must be 'dual_state', 'fusion_3d', or 'multimodal'.")


def _load_fold_pretrained_state(
    model: nn.Module,
    test_subject_id: str,
    pretrained_state_by_test_subject: Mapping[str, Mapping[str, torch.Tensor]] | None,
) -> bool:
    if not pretrained_state_by_test_subject:
        return False
    normalized_lookup = {
        normalize_subject_id(subject_id): state
        for subject_id, state in pretrained_state_by_test_subject.items()
    }
    state = normalized_lookup.get(normalize_subject_id(test_subject_id))
    if state is None:
        return False
    if callable(getattr(model, "load_ssl_bridge_state", None)):
        incompatible = model.load_ssl_bridge_state(dict(state))
    else:
        incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        joined = ", ".join(incompatible.unexpected_keys)
        raise ValueError(f"Pretrained state contains unexpected key(s): {joined}")
    return True


def _apply_pretrained_transfer_mode(model: nn.Module, transfer_mode: str) -> None:
    if transfer_mode == "finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
        if callable(getattr(model, "freeze_ssl_branch_models", None)):
            model.freeze_ssl_branch_models()
        return
    if transfer_mode == "freeze-encoder":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = not _is_encoder_parameter_name(name)
        if callable(getattr(model, "freeze_ssl_branch_models", None)):
            model.freeze_ssl_branch_models()
        return
    raise ValueError("pretrained_transfer_mode must be 'finetune' or 'freeze-encoder'.")


def _configure_finetune_stage(
    model: nn.Module,
    config: SupervisedTrainingConfig,
    *,
    stage: str,
) -> None:
    if stage not in {"standard", "stage1", "stage2"}:
        raise ValueError("stage must be 'standard', 'stage1', or 'stage2'.")
    if stage == "standard":
        return
    if stage == "stage1":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = not _is_encoder_parameter_name(name)
        if callable(getattr(model, "freeze_ssl_branch_models", None)):
            model.freeze_ssl_branch_models()
        return
    for name, parameter in model.named_parameters():
        if not _is_encoder_parameter_name(name):
            parameter.requires_grad = True
        elif config.freeze_conv_backbone:
            parameter.requires_grad = _is_encoder_final_projection_parameter_name(name)
        else:
            parameter.requires_grad = True
    if callable(getattr(model, "freeze_ssl_branch_models", None)):
        model.freeze_ssl_branch_models()


def _optimizer_config_for_finetune_stage(
    config: SupervisedTrainingConfig,
    stage: str,
) -> SupervisedTrainingConfig:
    if config.finetune_schedule == "staged_sam_swa":
        if stage == "stage1":
            return replace(
                config,
                optimizer_name="adamw",
                encoder_lr=config.encoder_lr,
                head_lr=config.lr,
            )
        if stage == "stage2":
            return replace(
                config,
                optimizer_name=config.optimizer_name,
                lr=config.head_lr if config.head_lr is not None else config.lr,
            )
    if config.finetune_schedule == "swa_only":
        return replace(config, optimizer_name="adamw" if config.optimizer_name == "adam" else config.optimizer_name)
    if config.finetune_schedule == "sam_only":
        return replace(config, optimizer_name="sam_adamw")
    return config


def _staged_freeze_epochs(config: SupervisedTrainingConfig) -> int:
    if config.freeze_encoder_epochs > 0:
        return config.freeze_encoder_epochs
    return config.freeze_pretrained_encoder_epochs


def _uses_pretrained_encoder_warmup(config: SupervisedTrainingConfig) -> bool:
    return (
        config.finetune_schedule != "staged_sam_swa"
        and config.pretrained_transfer_mode == "finetune"
        and config.freeze_pretrained_encoder_epochs > 0
    )


def _set_encoder_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for name, parameter in model.named_parameters():
        if _is_encoder_parameter_name(name):
            parameter.requires_grad = requires_grad


def _encoder_trainable(model: nn.Module) -> bool:
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if _is_encoder_parameter_name(name)
    ]
    return bool(encoder_parameters) and all(parameter.requires_grad for parameter in encoder_parameters)


def _format_optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> str:
    return ";".join(
        f"{group.get('name', index)}={float(group['lr']):.12g}"
        for index, group in enumerate(optimizer.param_groups)
    )


def _is_encoder_parameter_name(name: str) -> bool:
    if name.startswith("branch_models.") and ".encoder." in name:
        return True
    return name.startswith("encoder.")


def _is_encoder_final_projection_parameter_name(name: str) -> bool:
    final_tokens = (".fc", ".final", ".projection", ".head", ".linear")
    return any(token in name for token in final_tokens)


def _validate_architecture_feature_kind(config: SupervisedTrainingConfig) -> None:
    if config.encoder_kind not in {"cnn", "linear", "gncnn", "rescnn"}:
        raise ValueError("encoder_kind must be 'cnn', 'linear', 'gncnn', or 'rescnn'.")
    if config.optimizer_name not in {"adam", "adamw", "sam_adamw"}:
        raise ValueError("optimizer_name must be 'adam', 'adamw', or 'sam_adamw'.")
    if config.finetune_schedule not in {"standard", "swa_only", "sam_only", "staged_sam_swa"}:
        raise ValueError("finetune_schedule must be 'standard', 'swa_only', 'sam_only', or 'staged_sam_swa'.")
    if config.swa_start_epoch < 1:
        raise ValueError("swa_start_epoch must be at least 1.")
    if config.swa_lr <= 0:
        raise ValueError("swa_lr must be positive.")
    if config.sam_rho <= 0:
        raise ValueError("sam_rho must be positive.")
    if config.encoder_lr is not None and config.encoder_lr <= 0:
        raise ValueError("encoder_lr must be positive when provided.")
    if config.head_lr is not None and config.head_lr <= 0:
        raise ValueError("head_lr must be positive when provided.")
    if config.freeze_encoder_epochs < 0:
        raise ValueError("freeze_encoder_epochs must be non-negative.")
    if config.pretrained_transfer_mode not in {"finetune", "freeze-encoder"}:
        raise ValueError("pretrained_transfer_mode must be 'finetune' or 'freeze-encoder'.")
    if config.loss_name not in {"bce", "weighted_bce", "asymmetric_focal", "asymmetric_focal_fp_margin"}:
        raise ValueError(
            "loss_name must be 'bce', 'weighted_bce', 'asymmetric_focal', "
            "or 'asymmetric_focal_fp_margin'."
        )
    if config.positive_class_weight <= 0 or config.negative_class_weight <= 0:
        raise ValueError("positive_class_weight and negative_class_weight must be positive.")
    if config.focal_gamma_pos < 0 or config.focal_gamma_neg < 0:
        raise ValueError("focal gamma values must be non-negative.")
    if not 0 <= config.fp_margin <= 1:
        raise ValueError("fp_margin must be in [0, 1].")
    if config.fp_penalty_weight < 0:
        raise ValueError("fp_penalty_weight must be non-negative.")
    if config.mixup_alpha <= 0:
        raise ValueError("mixup_alpha must be positive.")
    if config.mixup_weight < 0:
        raise ValueError("mixup_weight must be non-negative.")
    if config.mixup_layer != "embedding":
        raise ValueError("mixup_layer must be 'embedding'.")
    if not 0 <= config.modality_dropout_prob <= 1:
        raise ValueError("modality_dropout_prob must be in [0, 1].")
    if not 0 <= config.state_dropout_prob <= 1:
        raise ValueError("state_dropout_prob must be in [0, 1].")
    if config.encoder_lr_multiplier <= 0:
        raise ValueError("encoder_lr_multiplier must be positive.")
    if config.transfer_head_lr_multiplier <= 0:
        raise ValueError("transfer_head_lr_multiplier must be positive.")
    if config.freeze_pretrained_encoder_epochs < 0:
        raise ValueError("freeze_pretrained_encoder_epochs must be non-negative.")
    if config.ssl_consistency_weight < 0:
        raise ValueError("ssl_consistency_weight must be non-negative.")
    if config.ssl_consistency_weight > 0 and not config.ssl_bridge_enabled:
        raise ValueError("ssl_consistency_weight requires ssl_bridge_enabled.")
    enabled_ssl_modes = sum(
        int(value)
        for value in (
            config.ssl_bridge_enabled,
            config.ssl_two_head_enabled,
            config.ssl_residual_enabled,
        )
    )
    if enabled_ssl_modes > 1:
        raise ValueError("ssl_bridge_enabled, ssl_two_head_enabled, and ssl_residual_enabled are mutually exclusive.")
    if config.ssl_aux_head_weight < 0:
        raise ValueError("ssl_aux_head_weight must be non-negative.")
    if not 0 <= config.ssl_fusion_weight <= 1:
        raise ValueError("ssl_fusion_weight must be in [0, 1].")
    if not 0 <= config.ssl_residual_weight <= 1:
        raise ValueError("ssl_residual_weight must be in [0, 1].")
    if not 0 <= config.bridge_residual_weight <= 1:
        raise ValueError("bridge_residual_weight must be in [0, 1].")
    if config.ssl_residual_weight + config.bridge_residual_weight > 1:
        raise ValueError("ssl_residual_weight + bridge_residual_weight must be <= 1.")
    if config.ssl_residual_random_main and not config.ssl_residual_enabled:
        raise ValueError("ssl_residual_random_main requires ssl_residual_enabled.")
    if config.ssl_residual_main_loss_only and not config.ssl_residual_enabled:
        raise ValueError("ssl_residual_main_loss_only requires ssl_residual_enabled.")
    if config.ssl_residual_preserve_main and not config.ssl_residual_enabled:
        raise ValueError("ssl_residual_preserve_main requires ssl_residual_enabled.")
    if config.ssl_residual_preserve_main and config.architecture != "multimodal":
        raise ValueError("ssl_residual_preserve_main requires multimodal architecture.")
    if config.ssl_residual_preserve_main and config.ssl_residual_random_main:
        raise ValueError("ssl_residual_preserve_main cannot be combined with ssl_residual_random_main.")
    if config.ssl_residual_preserve_main and config.ssl_residual_main_loss_only:
        raise ValueError("ssl_residual_preserve_main cannot be combined with ssl_residual_main_loss_only.")
    if config.ssl_residual_select_weights_on_val and not config.ssl_residual_enabled:
        raise ValueError("ssl_residual_select_weights_on_val requires ssl_residual_enabled.")
    if config.ssl_residual_logit_preservation_weight < 0:
        raise ValueError("ssl_residual_logit_preservation_weight must be non-negative.")
    if config.ssl_residual_logit_preservation_weight > 0 and not config.ssl_residual_enabled:
        raise ValueError("ssl_residual_logit_preservation_weight requires ssl_residual_enabled.")
    if config.ssl_residual_logit_delta_enabled and not config.ssl_residual_enabled:
        raise ValueError("ssl_residual_logit_delta_enabled requires ssl_residual_enabled.")
    if config.ssl_residual_logit_delta_enabled and not config.ssl_residual_preserve_main:
        raise ValueError("ssl_residual_logit_delta_enabled requires ssl_residual_preserve_main.")
    if config.ssl_residual_logit_delta_enabled and config.ssl_residual_select_weights_on_val:
        raise ValueError("ssl_residual_logit_delta_enabled cannot be combined with validation weight selection.")
    if config.ssl_residual_logit_delta_scale < 0:
        raise ValueError("ssl_residual_logit_delta_scale must be non-negative.")
    if not config.ssl_residual_weight_candidates:
        raise ValueError("ssl_residual_weight_candidates must not be empty.")
    if not config.bridge_residual_weight_candidates:
        raise ValueError("bridge_residual_weight_candidates must not be empty.")
    for candidate in (*config.ssl_residual_weight_candidates, *config.bridge_residual_weight_candidates):
        if not 0 <= candidate <= 1:
            raise ValueError("SSL residual weight candidates must be in [0, 1].")
    valid_selection_metrics = {"accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"}
    if config.ssl_residual_weight_selection_metric not in valid_selection_metrics:
        raise ValueError(
            "ssl_residual_weight_selection_metric must be one of "
            f"{', '.join(sorted(valid_selection_metrics))}."
        )
    if config.embedding_adapter_dim < 0:
        raise ValueError("embedding_adapter_dim must be non-negative.")
    if config.embedding_adapter_scale < 0:
        raise ValueError("embedding_adapter_scale must be non-negative.")
    if config.embedding_adapter_dim > 0 and enabled_ssl_modes:
        raise ValueError("embedding_adapter_dim is only supported for the standard multimodal CNN path.")
    if config.eeg_summary_features_enabled and config.architecture != "multimodal":
        raise ValueError("eeg_summary_features_enabled requires multimodal architecture.")
    if config.eeg_summary_features_enabled and enabled_ssl_modes:
        raise ValueError("eeg_summary_features_enabled is supported on the standard multimodal CNN path only.")
    if config.eeg_summary_embedding_dim < 0:
        raise ValueError("eeg_summary_embedding_dim must be non-negative.")
    if config.qeeg_features_enabled and config.architecture != "multimodal":
        raise ValueError("qeeg_features_enabled requires multimodal architecture.")
    if config.qeeg_features_enabled and enabled_ssl_modes:
        raise ValueError("qeeg_features_enabled is supported on the standard multimodal CNN path only.")
    if config.qeeg_features_enabled and not config.qeeg_feature_names:
        raise ValueError("qeeg_features_enabled requires at least one qEEG feature name.")
    if config.qeeg_primary_only and len(config.qeeg_feature_names) != 1:
        raise ValueError("primary qEEG mode is locked to exactly one qEEG feature.")
    if config.qeeg_hidden_dim < 1:
        raise ValueError("qeeg_hidden_dim must be positive.")
    if config.qeeg_clip_value is not None and config.qeeg_clip_value <= 0:
        raise ValueError("qeeg_clip_value must be positive when provided.")
    branches = branches_for_feature_kind(config.feature_kind)
    if config.eeg_summary_features_enabled and not {"psd", "wpli"}.issubset(set(branches)):
        raise ValueError("eeg_summary_features_enabled requires feature_kind to include both PSD and WPLI.")
    if config.qeeg_features_enabled and not {"psd", "wpli"}.issubset(set(branches)):
        raise ValueError("qeeg_features_enabled requires feature_kind to include both PSD and WPLI.")
    if len(branches) > 1 and config.architecture != "multimodal":
        raise ValueError(
            f"feature_kind={config.feature_kind} uses multiple feature branches "
            f"{branches} and requires architecture=multimodal; got architecture={config.architecture}."
        )


def _model_probabilities(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "x" in batch:
        return model(batch["x"])
    if "eo" in batch and "ec" in batch:
        return model(batch["eo"], batch["ec"])
    return model(batch)


def _evaluate_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    config: SupervisedTrainingConfig,
) -> float:
    model.eval()
    with torch.no_grad():
        loss = _compute_supervised_loss(model, batch, config, training=False)
    return float(loss.detach().cpu().item())


def _compute_supervised_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    config: SupervisedTrainingConfig,
    *,
    training: bool,
) -> torch.Tensor:
    probabilities = _model_probabilities_for_loss(model, batch, config, training=training)
    loss = _supervised_binary_loss(probabilities, batch["y"], config)
    if training and config.mixup_enabled:
        loss = loss + float(config.mixup_weight) * _manifold_mixup_loss(model, batch, config)
    if training and config.ssl_consistency_weight > 0:
        loss = loss + float(config.ssl_consistency_weight) * _ssl_embedding_consistency_loss(model, batch)
    if training and (config.ssl_two_head_enabled or config.ssl_residual_enabled) and config.ssl_aux_head_weight > 0:
        loss = loss + float(config.ssl_aux_head_weight) * _ssl_two_head_auxiliary_loss(model, batch, config)
    if training and config.ssl_residual_logit_preservation_weight > 0:
        loss = loss + float(config.ssl_residual_logit_preservation_weight) * _ssl_residual_logit_preservation_loss(
            model,
            batch,
        )
    if not torch.isfinite(loss):
        raise ValueError("supervised loss produced a non-finite value.")
    return loss


def _model_probabilities_for_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    config: SupervisedTrainingConfig,
    *,
    training: bool,
) -> torch.Tensor:
    if training and config.ssl_residual_enabled and config.ssl_residual_main_loss_only:
        probabilities_and_aux = model(batch, return_aux=True)
        if not isinstance(probabilities_and_aux, tuple) or len(probabilities_and_aux) != 2:
            raise ValueError("ssl_residual_main_loss_only requires auxiliary trainable probabilities.")
        _, aux = probabilities_and_aux
        if "trainable_probabilities" not in aux:
            raise ValueError("ssl_residual_main_loss_only missing trainable_probabilities.")
        return aux["trainable_probabilities"]
    if training and config.modality_dropout_prob > 0 and _supports_embedding_classifier(model):
        embedding = _extract_model_embedding(model, batch)
        embedding = _apply_modality_dropout_to_embeddings(
            embedding,
            feature_kind=config.feature_kind,
            embedding_dim=config.embedding_dim,
            dropout_prob=config.modality_dropout_prob,
        )
        return torch.sigmoid(model.classifier(embedding))
    return _model_probabilities(model, batch)


def _supports_embedding_classifier(model: nn.Module) -> bool:
    return callable(getattr(model, "extract_embedding", None)) and hasattr(model, "classifier")


def _extract_model_embedding(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if not _supports_embedding_classifier(model):
        raise ValueError("Manifold mixup requires a model with extract_embedding(batch) and classifier.")
    embedding = model.extract_embedding(batch)
    if isinstance(embedding, tuple):
        embedding = embedding[0]
    if not isinstance(embedding, torch.Tensor):
        raise ValueError("extract_embedding(batch) must return a tensor or a tuple whose first item is a tensor.")
    return embedding


def _sample_mixup_lambda(alpha: float, device: torch.device) -> torch.Tensor:
    if alpha <= 0:
        raise ValueError("mixup alpha must be positive.")
    concentration = torch.as_tensor(float(alpha), dtype=torch.float32, device=device)
    return torch.distributions.Beta(concentration, concentration).sample().clamp(0.0, 1.0)


def _mixup_embeddings_and_targets(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float | None = None,
    lam: torch.Tensor | float | None = None,
    permutation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.shape[0] != targets.shape[0]:
        raise ValueError("embeddings and targets must have the same batch size.")
    if embeddings.shape[0] < 2:
        raise ValueError("mixup requires at least two samples.")
    if lam is None:
        if alpha is None:
            raise ValueError("Either alpha or lam must be provided.")
        lam_tensor = _sample_mixup_lambda(alpha, embeddings.device)
    else:
        lam_tensor = torch.as_tensor(lam, dtype=embeddings.dtype, device=embeddings.device).clamp(0.0, 1.0)
    if permutation is None:
        permutation = torch.randperm(embeddings.shape[0], device=embeddings.device)
    else:
        permutation = permutation.to(device=embeddings.device)
    lam_tensor = lam_tensor.to(dtype=embeddings.dtype, device=embeddings.device)
    mixed_embeddings = lam_tensor * embeddings + (1.0 - lam_tensor) * embeddings[permutation]
    mixed_targets = lam_tensor * targets + (1.0 - lam_tensor) * targets[permutation]
    return mixed_embeddings, mixed_targets


def _manifold_mixup_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    config: SupervisedTrainingConfig,
) -> torch.Tensor:
    targets = batch["y"]
    if targets.shape[0] < 2:
        return torch.zeros((), dtype=targets.dtype, device=targets.device)
    embedding = _extract_model_embedding(model, batch)
    if config.modality_dropout_prob > 0:
        embedding = _apply_modality_dropout_to_embeddings(
            embedding,
            feature_kind=config.feature_kind,
            embedding_dim=config.embedding_dim,
            dropout_prob=config.modality_dropout_prob,
        )
    mixed_embedding, mixed_targets = _mixup_embeddings_and_targets(
        embedding,
        targets.to(dtype=embedding.dtype, device=embedding.device),
        alpha=config.mixup_alpha,
    )
    probabilities = torch.sigmoid(model.classifier(mixed_embedding))
    loss = F.binary_cross_entropy(probabilities, mixed_targets)
    if not torch.isfinite(loss):
        raise ValueError("manifold mixup loss produced a non-finite value.")
    return loss


def _ssl_embedding_consistency_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if not callable(getattr(model, "extract_bridge_embeddings", None)):
        raise ValueError("ssl consistency requires a bridge model with extract_bridge_embeddings(batch).")
    trainable_embedding, ssl_embedding = model.extract_bridge_embeddings(batch)
    trainable_embedding = F.normalize(trainable_embedding.float(), dim=1)
    ssl_embedding = F.normalize(ssl_embedding.detach().float(), dim=1)
    loss = F.mse_loss(trainable_embedding, ssl_embedding)
    if not torch.isfinite(loss):
        raise ValueError("ssl consistency loss produced a non-finite value.")
    return loss


def _ssl_two_head_auxiliary_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    config: SupervisedTrainingConfig,
) -> torch.Tensor:
    probabilities_and_aux = model(batch, return_aux=True)
    if not isinstance(probabilities_and_aux, tuple) or len(probabilities_and_aux) != 2:
        raise ValueError("ssl two-head loss requires a model that returns auxiliary head probabilities.")
    _, aux = probabilities_and_aux
    required = {"trainable_probabilities", "ssl_probabilities"}
    missing = required - set(aux)
    if missing:
        raise ValueError(f"ssl two-head loss missing auxiliary output(s): {', '.join(sorted(missing))}")
    head_probabilities = [aux["trainable_probabilities"], aux["ssl_probabilities"]]
    if "bridge_probabilities" in aux:
        head_probabilities.append(aux["bridge_probabilities"])
    loss = torch.stack(
        [_supervised_binary_loss(probabilities, batch["y"], config) for probabilities in head_probabilities]
    ).mean()
    if not torch.isfinite(loss):
        raise ValueError("ssl two-head auxiliary loss produced a non-finite value.")
    return loss


def _ssl_residual_logit_preservation_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    probabilities_and_aux = model(batch, return_aux=True)
    if not isinstance(probabilities_and_aux, tuple) or len(probabilities_and_aux) != 2:
        raise ValueError("ssl residual logit preservation requires auxiliary probabilities.")
    probabilities, aux = probabilities_and_aux
    if "trainable_probabilities" not in aux:
        raise ValueError("ssl residual logit preservation missing trainable_probabilities.")
    loss = F.mse_loss(
        _probability_to_logit(probabilities),
        _probability_to_logit(aux["trainable_probabilities"]).detach(),
    )
    if not torch.isfinite(loss):
        raise ValueError("ssl residual logit preservation loss produced a non-finite value.")
    return loss


def _probability_to_logit(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = probabilities.clamp(eps, 1.0 - eps)
    return torch.logit(probabilities)


def _eeg_summary_aux_scores(
    aux: dict[str, torch.Tensor],
    config: SupervisedTrainingConfig,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    if "modality_weights" in aux:
        weights = aux["modality_weights"].detach().cpu().numpy()[0]
        for name, value in zip((*branches_for_feature_kind(config.feature_kind), "eeg_summary"), weights):
            scores[f"modality_weight_{name}"] = float(value)
    if "eeg_summary_feature_importance" in aux:
        importances = aux["eeg_summary_feature_importance"].detach().cpu().numpy()
        for name, value in zip(config.eeg_summary_feature_names, importances):
            scores[f"eeg_summary_importance_{_safe_aux_column_token(name)}"] = float(value)
    return scores


def _safe_aux_column_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "feature"


def _apply_modality_dropout_to_embeddings(
    embeddings: torch.Tensor,
    *,
    feature_kind: str,
    embedding_dim: int,
    dropout_prob: float,
) -> torch.Tensor:
    if dropout_prob <= 0:
        return embeddings
    branches = branches_for_feature_kind(feature_kind)
    if len(branches) <= 1:
        return embeddings
    n_branches = len(branches)
    if embeddings.ndim != 2:
        raise ValueError("modality dropout expects a 2D embedding tensor.")
    if embeddings.shape[1] != embedding_dim * n_branches:
        raise ValueError(
            "embedding width does not match embedding_dim x number of modality branches "
            f"({embeddings.shape[1]} != {embedding_dim} x {n_branches})."
        )
    drop = torch.rand(
        (embeddings.shape[0], n_branches),
        dtype=embeddings.dtype,
        device=embeddings.device,
    ) < float(dropout_prob)
    keep = ~drop
    all_dropped = ~keep.any(dim=1)
    if all_dropped.any():
        replacement = torch.randint(0, n_branches, size=(int(all_dropped.sum().item()),), device=embeddings.device)
        keep[all_dropped] = False
        keep[all_dropped, replacement] = True
    mask = keep.to(dtype=embeddings.dtype).repeat_interleave(embedding_dim, dim=1)
    dropped = embeddings * mask
    if not torch.isfinite(dropped).all():
        raise ValueError("modality dropout produced non-finite embeddings.")
    return dropped


def _supervised_binary_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    config: SupervisedTrainingConfig,
) -> torch.Tensor:
    if config.loss_name == "bce":
        return F.binary_cross_entropy(probabilities, targets)
    if config.loss_name == "weighted_bce":
        return _weighted_binary_cross_entropy(probabilities, targets, config)
    focal = _asymmetric_focal_binary_cross_entropy(
        probabilities,
        targets,
        positive_class_weight=config.positive_class_weight,
        negative_class_weight=config.negative_class_weight,
        gamma_pos=config.focal_gamma_pos,
        gamma_neg=config.focal_gamma_neg,
    )
    if config.loss_name == "asymmetric_focal":
        return focal
    if config.loss_name == "asymmetric_focal_fp_margin":
        return focal + _hard_false_positive_penalty(
            probabilities,
            targets,
            margin=config.fp_margin,
            penalty_weight=config.fp_penalty_weight,
        )
    raise ValueError(f"Unsupported loss_name: {config.loss_name}")


def _weighted_binary_cross_entropy(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    config: SupervisedTrainingConfig,
) -> torch.Tensor:
    weights = torch.where(
        targets >= 0.5,
        torch.as_tensor(config.positive_class_weight, dtype=probabilities.dtype, device=probabilities.device),
        torch.as_tensor(config.negative_class_weight, dtype=probabilities.dtype, device=probabilities.device),
    )
    return F.binary_cross_entropy(probabilities, targets, weight=weights)


def _asymmetric_focal_binary_cross_entropy(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_class_weight: float,
    negative_class_weight: float,
    gamma_pos: float,
    gamma_neg: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    p = probabilities.clamp(min=eps, max=1.0 - eps)
    y = targets.to(dtype=p.dtype, device=p.device)
    pos_weight = torch.as_tensor(positive_class_weight, dtype=p.dtype, device=p.device)
    neg_weight = torch.as_tensor(negative_class_weight, dtype=p.dtype, device=p.device)
    loss_pos = -pos_weight * torch.pow(1.0 - p, gamma_pos) * torch.log(p) * y
    loss_neg = -neg_weight * torch.pow(p, gamma_neg) * torch.log(1.0 - p) * (1.0 - y)
    loss = (loss_pos + loss_neg).mean()
    if not torch.isfinite(loss):
        raise ValueError("asymmetric focal loss produced a non-finite value.")
    return loss


def _hard_false_positive_penalty(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float,
    penalty_weight: float,
) -> torch.Tensor:
    p = probabilities
    y = targets.to(dtype=p.dtype, device=p.device)
    negative_mask = (y < 0.5).to(dtype=p.dtype)
    excess = torch.relu(p - margin)
    penalty = torch.as_tensor(penalty_weight, dtype=p.dtype, device=p.device) * (
        negative_mask * excess.pow(2)
    ).mean()
    if not torch.isfinite(penalty):
        raise ValueError("hard false-positive penalty produced a non-finite value.")
    return penalty


def _make_batch(
    records: Iterable[SupervisedFeatureRecord],
    scaler: tuple[np.ndarray, np.ndarray] | dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    architecture: str,
) -> dict[str, torch.Tensor]:
    records = list(records)
    y = np.array([[record.label] for record in records], dtype=np.float32)
    if architecture == "multimodal":
        if not isinstance(scaler, dict):
            raise ValueError("Multimodal batches require branch-specific scalers.")
        batch = {"y": torch.as_tensor(y, device=device)}
        for branch, (mean, std) in scaler.items():
            if branch in {"__eeg_summary__", "__qeeg__"}:
                continue
            eo, ec = _branch_state_arrays(records, branch)
            batch[f"{branch}_eo"] = torch.as_tensor(((eo - mean) / std).astype(np.float32), device=device)
            batch[f"{branch}_ec"] = torch.as_tensor(((ec - mean) / std).astype(np.float32), device=device)
        if any(record.eeg_summary is not None for record in records):
            if "__eeg_summary__" not in scaler:
                raise ValueError("EEG summary records require an EEG summary scaler.")
            mean, std = scaler["__eeg_summary__"]
            summary = _eeg_summary_arrays(records)
            batch["eeg_summary"] = torch.as_tensor(((summary - mean) / std).astype(np.float32), device=device)
        if any(record.qeeg_features is not None for record in records):
            if "__qeeg__" not in scaler:
                raise ValueError("qEEG records require a qEEG scaler.")
            mean, std = scaler["__qeeg__"]
            qeeg = _qeeg_feature_arrays(records)
            batch["qeeg_features"] = torch.as_tensor(((qeeg - mean) / std).astype(np.float32), device=device)
        return batch

    if isinstance(scaler, dict):
        if len(scaler) != 1:
            raise ValueError("dual_state and fusion_3d architectures require exactly one feature branch.")
        branch = next(iter(scaler))
        mean, std = scaler[branch]
        eo, ec = _branch_state_arrays(records, branch)
    else:
        mean, std = scaler
        eo = np.stack([record.eo for record in records]).astype(np.float32)
        ec = np.stack([record.ec for record in records]).astype(np.float32)
    eo = ((eo - mean) / std).astype(np.float32)
    ec = ((ec - mean) / std).astype(np.float32)
    if architecture == "fusion_3d":
        x = np.stack([eo, ec], axis=1)
        return {"x": torch.as_tensor(x, device=device), "y": torch.as_tensor(y, device=device)}
    return {
        "eo": torch.as_tensor(eo, device=device),
        "ec": torch.as_tensor(ec, device=device),
        "y": torch.as_tensor(y, device=device),
    }


def _fit_state_scaler(
    records: Iterable[SupervisedFeatureRecord],
) -> tuple[np.ndarray, np.ndarray] | dict[str, tuple[np.ndarray, np.ndarray]]:
    records = list(records)
    if any(record.modalities is not None for record in records):
        branches = list(records[0].modalities or {})
        if not branches:
            raise ValueError("No modality branches are available to fit feature scaler.")
        scalers = {
            branch: _fit_array_scaler(
                [
                    state
                    for record in records
                    for state in (record.modalities or {})[branch]
                ]
            )
            for branch in branches
        }
        if any(record.eeg_summary is not None for record in records):
            scalers["__eeg_summary__"] = _fit_vector_scaler(_eeg_summary_arrays(records))
        if any(record.qeeg_features is not None for record in records):
            scalers["__qeeg__"] = _fit_vector_scaler(_qeeg_feature_arrays(records))
        return scalers

    states = []
    for record in records:
        states.extend([record.eo, record.ec])
    return _fit_array_scaler(states)


def _fit_array_scaler(states: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    states = list(states)
    if not states:
        raise ValueError("No records available to fit feature scaler.")
    stacked = np.stack(states).astype(np.float32)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean.astype(np.float32), std


def _fit_vector_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Vector scaler requires a non-empty 2D array.")
    mean = values.astype(np.float32).mean(axis=0)
    std = values.astype(np.float32).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean.astype(np.float32), std


def _branch_state_arrays(records: list[SupervisedFeatureRecord], branch: str) -> tuple[np.ndarray, np.ndarray]:
    eo = np.stack([(record.modalities or {})[branch][0] for record in records]).astype(np.float32)
    ec = np.stack([(record.modalities or {})[branch][1] for record in records]).astype(np.float32)
    return eo, ec


def _eeg_summary_arrays(records: list[SupervisedFeatureRecord]) -> np.ndarray:
    missing = [record.subject_id for record in records if record.eeg_summary is None]
    if missing:
        raise ValueError(f"Missing EEG summary features for subject(s): {', '.join(missing)}")
    return np.stack([np.asarray(record.eeg_summary, dtype=np.float32) for record in records]).astype(np.float32)


def _eeg_summary_feature_names_from_records(records: list[SupervisedFeatureRecord]) -> tuple[str, ...]:
    for record in records:
        if record.eeg_summary_feature_names:
            return tuple(record.eeg_summary_feature_names)
    raise ValueError("eeg_summary_features_enabled requires records with eeg_summary_feature_names.")


def _qeeg_feature_arrays(records: list[SupervisedFeatureRecord]) -> np.ndarray:
    missing = [record.subject_id for record in records if record.qeeg_features is None]
    if missing:
        raise ValueError(f"Missing qEEG features for subject(s): {', '.join(missing)}")
    return np.stack([np.asarray(record.qeeg_features, dtype=np.float32) for record in records]).astype(np.float32)


def _qeeg_feature_names_from_records(records: list[SupervisedFeatureRecord]) -> tuple[str, ...]:
    for record in records:
        if record.qeeg_feature_names:
            return tuple(record.qeeg_feature_names)
    raise ValueError("qeeg_features_enabled requires records with qeeg_feature_names.")


def _select_named_feature_values(
    values: np.ndarray,
    feature_names: tuple[str, ...],
    requested_names: tuple[str, ...],
    *,
    feature_family: str,
) -> np.ndarray:
    if not requested_names:
        raise ValueError(f"{feature_family} feature selection requires at least one feature name.")
    lookup = {name: index for index, name in enumerate(feature_names)}
    missing = [name for name in requested_names if name not in lookup]
    if missing:
        raise ValueError(f"Missing requested {feature_family} feature(s): {', '.join(missing)}")
    selected = np.asarray([values[lookup[name]] for name in requested_names], dtype=np.float32)
    if selected.ndim != 1 or not np.isfinite(selected).all():
        raise ValueError(f"Selected {feature_family} features must be a finite 1D vector.")
    return selected


def _train_validation_subjects(train_subjects: list[str], fold_index: int) -> tuple[list[str], list[str]]:
    if len(train_subjects) < 3:
        return train_subjects, train_subjects
    validation_index = fold_index % len(train_subjects)
    val_subjects = [train_subjects[validation_index]]
    fit_subjects = [
        subject_id
        for index, subject_id in enumerate(train_subjects)
        if index != validation_index
    ]
    return fit_subjects, val_subjects


def _load_branch_array(output_root: Path, subject_id: str, state: str, branch: str) -> np.ndarray:
    if branch == "psd":
        return _load_feature_array(
            output_root / "data" / "features" / "psd" / f"{subject_id}_{state}_psd.npz",
            "psd",
        )
    if branch in {"wpli", "icoh"}:
        key = "wpli" if branch == "wpli" else "imaginary_coherence"
        return _load_feature_array(
            output_root / "data" / "features" / "fc" / f"{subject_id}_{state}_fc.npz",
            key,
        )
    raise ValueError(f"Unsupported feature branch: {branch}")


def _load_feature_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required feature file is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload:
            raise KeyError(f"Feature file {path} is missing required key {key!r}.")
        return np.asarray(payload[key], dtype=np.float32)


def _validate_feature_shape(array: np.ndarray, branch: str, subject_id: str, state: str) -> None:
    expected = (62, 90) if branch == "psd" else (1891, 6)
    if array.shape != expected:
        raise ValueError(
            f"Expected {branch} shape {expected} for {subject_id} {state}, got {array.shape}."
        )


def _normalize_feature_kind(feature_kind: str) -> str:
    if feature_kind in {"psd", "fc-wpli", "fc-icoh", "fc-both", "psd-fc-wpli", "psd-fc-icoh", "psd-fc-both"}:
        return feature_kind
    if feature_kind == "fc":
        return "fc-wpli"
    raise ValueError(
        "feature_kind must be 'psd', 'fc-wpli', 'fc-icoh', 'fc-both', "
        "'psd-fc-wpli', 'psd-fc-icoh', or 'psd-fc-both'."
    )


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
