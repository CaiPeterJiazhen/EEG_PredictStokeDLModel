from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.models.segment_mil_model import WPLISegmentAttentionMILModel
from eeg_recovery.training.loso import LOSOFold, make_loso_folds
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.schedulers import EarlyStopping
from eeg_recovery.training.segment_ssl_dataset import BASELINE_STAGE_ALIASES, SegmentSSLRecord
from eeg_recovery.training.ssl_checkpointing import (
    extract_wpli_encoder_from_dual_segment_barlow_checkpoint,
    load_ssl_encoder_checkpoint,
    make_ssl_checkpoint_name,
)
from eeg_recovery.training.train_supervised import _plot_loss_history, _train_validation_subjects, resolve_device

FOLDSTRICT_WPLI_CHECKPOINT_TAG = "segssl_wpli_barlow_foldstrict_v1"


@dataclass(frozen=True)
class WPLISegmentBagRecord:
    subject_id: str
    label: int
    eo_segments: np.ndarray
    ec_segments: np.ndarray
    segment_metadata: Mapping[str, list[dict[str, Any]]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", normalize_subject_id(self.subject_id))
        if self.label not in {0, 1}:
            raise ValueError("label must be 0 or 1.")
        _validate_segments(self.eo_segments, "eo_segments")
        _validate_segments(self.ec_segments, "ec_segments")
        if self.eo_segments.shape[0] < 1 or self.ec_segments.shape[0] < 1:
            raise ValueError("Both EO and EC bags must contain at least one segment.")


@dataclass(frozen=True)
class SegmentMILScaler:
    mean: np.ndarray
    std: np.ndarray
    fitted_subject_ids: tuple[str, ...]
    n_fit_segments: int


@dataclass(frozen=True)
class SegmentMILTrainingConfig:
    device: str = "auto"
    seed: int = 0
    embedding_dim: int = 32
    attention_hidden_dim: int = 16
    dropout: float = 0.0
    encoder_kind: str = "cnn"
    max_segments_per_state: int | None = 64
    stage1_epochs: int = 20
    stage2_epochs: int = 80
    head_lr: float = 0.002
    encoder_lr: float = 1e-4
    stage2_head_lr: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 100
    freeze_bn: bool = True
    reset_bn_running_stats: bool = False
    eval_use_all_segments: bool = False


@dataclass(frozen=True)
class WPLIEncoderCheckpointResolution:
    encoder_state_dict: dict[str, torch.Tensor]
    source: str
    checkpoint_path: Path | None
    source_checkpoint_path: Path | None = None


def load_wpli_segment_bag_records(
    records: Iterable[SegmentSSLRecord],
    label_table: pd.DataFrame,
    *,
    baseline_stage: str = "baseline",
) -> list[WPLISegmentBagRecord]:
    """Group baseline EO/EC wPLI segment records into one bag per supervised patient."""

    if "subject_id" not in label_table.columns or "label" not in label_table.columns:
        raise ValueError("label_table must include subject_id and label columns.")
    labels = {
        normalize_subject_id(row.subject_id): int(row.label)
        for row in label_table.loc[:, ["subject_id", "label"]].itertuples(index=False)
    }
    baseline_tokens = {baseline_stage, baseline_stage.strip().lower(), *BASELINE_STAGE_ALIASES}
    grouped: dict[str, dict[str, list[SegmentSSLRecord]]] = {
        subject_id: {"EO": [], "EC": []}
        for subject_id in labels
    }
    for record in records:
        subject_id = normalize_subject_id(record.subject_id)
        if record.group != "patient" or subject_id not in labels:
            continue
        stage = record.stage.strip().lower()
        if record.stage not in baseline_tokens and stage not in baseline_tokens:
            continue
        if record.state not in {"EO", "EC"}:
            continue
        if "wpli" not in record.features:
            raise ValueError(f"Segment record for {subject_id} {record.state} is missing wpli.")
        grouped[subject_id][record.state].append(record)

    bags: list[WPLISegmentBagRecord] = []
    for subject_id in sorted(labels):
        eo_records = sorted(grouped[subject_id]["EO"], key=lambda item: item.segment_index)
        ec_records = sorted(grouped[subject_id]["EC"], key=lambda item: item.segment_index)
        if not eo_records or not ec_records:
            raise ValueError(
                f"Missing baseline EO/EC wPLI segments for {subject_id}: "
                f"EO={len(eo_records)}, EC={len(ec_records)}."
            )
        eo_segments = np.stack([record.features["wpli"] for record in eo_records]).astype(np.float32)
        ec_segments = np.stack([record.features["wpli"] for record in ec_records]).astype(np.float32)
        bags.append(
            WPLISegmentBagRecord(
                subject_id=subject_id,
                label=labels[subject_id],
                eo_segments=eo_segments,
                ec_segments=ec_segments,
                segment_metadata={
                    "EO": [_segment_metadata(record) for record in eo_records],
                    "EC": [_segment_metadata(record) for record in ec_records],
                },
            )
        )
    return bags


def fit_segment_scaler_on_train_bags(
    bags: Iterable[WPLISegmentBagRecord],
    *,
    fit_subject_ids: Iterable[str],
) -> SegmentMILScaler:
    fit_ids = tuple(normalize_subject_id(subject_id) for subject_id in fit_subject_ids)
    fit_set = set(fit_ids)
    selected = [bag for bag in bags if bag.subject_id in fit_set]
    if not selected:
        raise ValueError("No fit_subject_ids were found in the segment bags.")
    arrays = [
        state
        for bag in selected
        for state in (bag.eo_segments, bag.ec_segments)
    ]
    stacked = np.concatenate(arrays, axis=0).astype(np.float32)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return SegmentMILScaler(
        mean=mean,
        std=std,
        fitted_subject_ids=tuple(sorted(fit_set)),
        n_fit_segments=int(stacked.shape[0]),
    )


def transform_bag(bag: WPLISegmentBagRecord, scaler: SegmentMILScaler) -> WPLISegmentBagRecord:
    return WPLISegmentBagRecord(
        subject_id=bag.subject_id,
        label=bag.label,
        eo_segments=((bag.eo_segments.astype(np.float32) - scaler.mean) / scaler.std).astype(np.float32),
        ec_segments=((bag.ec_segments.astype(np.float32) - scaler.mean) / scaler.std).astype(np.float32),
        segment_metadata=bag.segment_metadata,
    )


def load_wpli_segment_barlow_encoder_for_fold(
    *,
    checkpoint_dir: str | Path,
    fold: LOSOFold,
    seed: int,
    embedding_dim: int = 32,
    pretrain_epochs: int = 20,
    ssl_data_scope: str = "all-patient",
    source_feature_manifest_hash: str,
    dropout: float = 0.0,
    pretrain_lr: float = 1e-3,
    checkpoint_tag: str | None = None,
    reuse_only: bool = True,
    train_if_missing: Callable[[Path, dict[str, object]], None] | None = None,
) -> dict[str, torch.Tensor]:
    """Load one fold-specific WPLI Segment Barlow encoder checkpoint."""

    checkpoint_path = wpli_segment_barlow_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        fold=fold,
        seed=seed,
        embedding_dim=embedding_dim,
        pretrain_epochs=pretrain_epochs,
        ssl_data_scope=ssl_data_scope,
        checkpoint_tag=checkpoint_tag,
    )
    expected_metadata = expected_wpli_segment_barlow_metadata(
        fold=fold,
        seed=seed,
        embedding_dim=embedding_dim,
        pretrain_epochs=pretrain_epochs,
        ssl_data_scope=ssl_data_scope,
        source_feature_manifest_hash=source_feature_manifest_hash,
        dropout=dropout,
        pretrain_lr=pretrain_lr,
    )
    if not checkpoint_path.exists():
        if reuse_only or train_if_missing is None:
            raise FileNotFoundError(f"Missing WPLI Segment Barlow checkpoint: {checkpoint_path}")
        train_if_missing(checkpoint_path, expected_metadata)
    checkpoint = load_ssl_encoder_checkpoint(checkpoint_path, expected_metadata)
    encoder_state = checkpoint.get("encoder_state_dict")
    if not isinstance(encoder_state, dict):
        raise ValueError(f"SSL checkpoint {checkpoint_path} is missing encoder_state_dict.")
    return {
        str(key): value.detach().cpu().clone()
        for key, value in encoder_state.items()
        if isinstance(value, torch.Tensor)
    }


def resolve_existing_wpli_segment_barlow_encoder_for_fold(
    *,
    checkpoint_dir: str | Path,
    fold: LOSOFold,
    seed: int,
    embedding_dim: int = 32,
    pretrain_epochs: int = 20,
    ssl_data_scope: str = "all-patient",
    source_feature_manifest_hash: str,
    dual_source_feature_manifest_hash: str | None = None,
    dropout: float = 0.0,
    pretrain_lr: float = 1e-3,
    dual_checkpoint_dir: str | Path | None = None,
) -> WPLIEncoderCheckpointResolution:
    """Resolve an existing strict LOSO WPLI SSL encoder without training.

    Resolution order:
    1. matching branch-specific WPLI Segment Barlow checkpoint;
    2. matching foldstrict WPLI checkpoint previously extracted from dual;
    3. matching dual Segment Barlow checkpoint, from which the WPLI encoder is
       extracted and saved as a foldstrict branch-specific checkpoint.
    """

    checkpoint_root = Path(checkpoint_dir)
    expected_branch_metadata = expected_wpli_segment_barlow_metadata(
        fold=fold,
        seed=seed,
        embedding_dim=embedding_dim,
        pretrain_epochs=pretrain_epochs,
        ssl_data_scope=ssl_data_scope,
        source_feature_manifest_hash=source_feature_manifest_hash,
        dropout=dropout,
        pretrain_lr=pretrain_lr,
    )
    rejection_messages: list[str] = []
    for tag in (None, FOLDSTRICT_WPLI_CHECKPOINT_TAG):
        checkpoint_path = wpli_segment_barlow_checkpoint_path(
            checkpoint_dir=checkpoint_root,
            fold=fold,
            seed=seed,
            embedding_dim=embedding_dim,
            pretrain_epochs=pretrain_epochs,
            ssl_data_scope=ssl_data_scope,
            checkpoint_tag=tag,
        )
        if not checkpoint_path.exists():
            continue
        try:
            state = load_wpli_segment_barlow_encoder_for_fold(
                checkpoint_dir=checkpoint_root,
                fold=fold,
                seed=seed,
                embedding_dim=embedding_dim,
                pretrain_epochs=pretrain_epochs,
                ssl_data_scope=ssl_data_scope,
                source_feature_manifest_hash=source_feature_manifest_hash,
                dropout=dropout,
                pretrain_lr=pretrain_lr,
                checkpoint_tag=tag,
                reuse_only=True,
            )
        except ValueError as error:
            rejection_messages.append(f"{checkpoint_path.name}: {error}")
            continue
        return WPLIEncoderCheckpointResolution(
            encoder_state_dict=state,
            source="branch_specific",
            checkpoint_path=checkpoint_path,
        )

    if dual_source_feature_manifest_hash is not None:
        expected_dual_metadata = expected_dual_segment_barlow_metadata(
            fold=fold,
            seed=seed,
            embedding_dim=embedding_dim,
            pretrain_epochs=pretrain_epochs,
            ssl_data_scope=ssl_data_scope,
            source_feature_manifest_hash=dual_source_feature_manifest_hash,
            dropout=dropout,
            pretrain_lr=pretrain_lr,
        )
        extracted_path = wpli_segment_barlow_checkpoint_path(
            checkpoint_dir=checkpoint_root,
            fold=fold,
            seed=seed,
            embedding_dim=embedding_dim,
            pretrain_epochs=pretrain_epochs,
            ssl_data_scope=ssl_data_scope,
            checkpoint_tag=FOLDSTRICT_WPLI_CHECKPOINT_TAG,
        )
        for dual_path in _dual_segment_barlow_checkpoint_candidates(
            checkpoint_dir=dual_checkpoint_dir or checkpoint_root,
            fold=fold,
            seed=seed,
            embedding_dim=embedding_dim,
            pretrain_epochs=pretrain_epochs,
            ssl_data_scope=ssl_data_scope,
        ):
            try:
                state = extract_wpli_encoder_from_dual_segment_barlow_checkpoint(
                    dual_checkpoint_path=dual_path,
                    output_path=extracted_path,
                    expected_dual_metadata=expected_dual_metadata,
                    branch_metadata=expected_branch_metadata,
                )
            except ValueError as error:
                rejection_messages.append(f"{dual_path.name}: {error}")
                continue
            return WPLIEncoderCheckpointResolution(
                encoder_state_dict=state,
                source="dual_extracted",
                checkpoint_path=extracted_path,
                source_checkpoint_path=dual_path,
            )

    detail = "; ".join(rejection_messages)
    message = f"No reusable strict LOSO WPLI Segment Barlow checkpoint found for {fold.test_subject_id}."
    if detail:
        message = f"{message} Rejections: {detail}"
    raise FileNotFoundError(message)


def expected_wpli_segment_barlow_metadata(
    *,
    fold: LOSOFold,
    seed: int,
    embedding_dim: int,
    pretrain_epochs: int,
    ssl_data_scope: str,
    source_feature_manifest_hash: str,
    dropout: float,
    pretrain_lr: float,
) -> dict[str, object]:
    return {
        "checkpoint_type": "segment_ssl_encoder",
        "branch": "wpli",
        "ssl_objective": "barlow",
        "segment_ssl_method": "segment_barlow",
        "base_seed": seed,
        "fold_index": fold.fold_index,
        "test_subject_id": fold.test_subject_id,
        "excluded_subject_id": fold.test_subject_id,
        "ssl_data_scope": ssl_data_scope,
        "feature_kind": "fc-wpli",
        "encoder_kind": "cnn",
        "embedding_dim": embedding_dim,
        "dropout": dropout,
        "pretrain_epochs": pretrain_epochs,
        "pretrain_lr": pretrain_lr,
        "source_feature_manifest_hash": source_feature_manifest_hash,
    }


def expected_dual_segment_barlow_metadata(
    *,
    fold: LOSOFold,
    seed: int,
    embedding_dim: int,
    pretrain_epochs: int,
    ssl_data_scope: str,
    source_feature_manifest_hash: str,
    dropout: float,
    pretrain_lr: float,
) -> dict[str, object]:
    return {
        "checkpoint_type": "dual_segment_ssl_encoder",
        "branch": "dual",
        "ssl_objective": "barlow",
        "segment_ssl_method": "dual_segment_barlow",
        "base_seed": seed,
        "fold_index": fold.fold_index,
        "test_subject_id": fold.test_subject_id,
        "excluded_subject_id": fold.test_subject_id,
        "ssl_data_scope": ssl_data_scope,
        "feature_kind": "psd-fc-wpli",
        "encoder_kind": "cnn",
        "embedding_dim": embedding_dim,
        "dropout": dropout,
        "pretrain_epochs": pretrain_epochs,
        "pretrain_lr": pretrain_lr,
        "source_feature_manifest_hash": source_feature_manifest_hash,
    }


def wpli_segment_barlow_checkpoint_path(
    *,
    checkpoint_dir: str | Path,
    fold: LOSOFold,
    seed: int,
    embedding_dim: int,
    pretrain_epochs: int,
    ssl_data_scope: str,
    checkpoint_tag: str | None,
) -> Path:
    if checkpoint_tag == FOLDSTRICT_WPLI_CHECKPOINT_TAG:
        filename = (
            f"{FOLDSTRICT_WPLI_CHECKPOINT_TAG}_seed{seed}_fold{fold.fold_index:02d}_"
            f"test_{fold.test_subject_id}_emb{embedding_dim}_pre{pretrain_epochs}_{ssl_data_scope}.pt"
        )
        return Path(checkpoint_dir) / _safe_checkpoint_filename(filename)
    return Path(checkpoint_dir) / make_ssl_checkpoint_name(
        method="segssl",
        branch="wpli",
        ssl_objective="barlow",
        seed=seed,
        fold_index=fold.fold_index,
        test_subject_id=fold.test_subject_id,
        embedding_dim=embedding_dim,
        pretrain_epochs=pretrain_epochs,
        ssl_data_scope=ssl_data_scope,
        tag=checkpoint_tag,
    )


def _safe_checkpoint_filename(filename: str) -> str:
    stem, suffix = Path(filename).stem, Path(filename).suffix
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_-")
    if not safe_stem:
        raise ValueError("Checkpoint filename cannot be empty.")
    return f"{safe_stem}{suffix}"


def _dual_segment_barlow_checkpoint_candidates(
    *,
    checkpoint_dir: str | Path,
    fold: LOSOFold,
    seed: int,
    embedding_dim: int,
    pretrain_epochs: int,
    ssl_data_scope: str,
) -> list[Path]:
    root = Path(checkpoint_dir)
    exact = root / make_ssl_checkpoint_name(
        method="segssl",
        branch="dual",
        ssl_objective="barlow",
        seed=seed,
        fold_index=fold.fold_index,
        test_subject_id=fold.test_subject_id,
        embedding_dim=embedding_dim,
        pretrain_epochs=pretrain_epochs,
        ssl_data_scope=ssl_data_scope,
    )
    tokens = [
        "dual",
        f"seed{seed}",
        f"fold{fold.fold_index:02d}",
        f"test_{fold.test_subject_id}",
        f"emb{embedding_dim}",
        f"pre{pretrain_epochs}",
        ssl_data_scope,
    ]
    candidates = [exact] if exact.exists() else []
    if root.exists():
        for path in sorted(root.glob("*.pt")):
            name = path.name
            if all(token in name for token in tokens) and path not in candidates:
                candidates.append(path)
    return candidates


def run_loso_wpli_segment_mil_with_history(
    records: Iterable[WPLISegmentBagRecord],
    training_config: SegmentMILTrainingConfig | None = None,
    *,
    encoder_state_by_test_subject: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run patient-level LOSO fine-tuning for WPLI segment attention MIL."""

    config = training_config or SegmentMILTrainingConfig()
    _validate_config(config)
    _seed_everything(config.seed)
    device = resolve_device(config.device)
    bags = list(records)
    if len(bags) < 2:
        raise ValueError("Segment MIL LOSO requires at least two patient bags.")
    bag_by_subject = {bag.subject_id: bag for bag in bags}
    if len(bag_by_subject) != len(bags):
        raise ValueError("Segment MIL records must contain one bag per subject.")
    folds = make_loso_folds(bag_by_subject)
    predictions: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    model_name = "wpli_segbarlow_attention_mil"

    normalized_encoder_states = {
        normalize_subject_id(subject_id): state
        for subject_id, state in (encoder_state_by_test_subject or {}).items()
    }

    for fold in folds:
        fit_subjects, val_subjects = _train_validation_subjects(list(fold.train_subject_ids), fold.fold_index)
        scaler = fit_segment_scaler_on_train_bags(bags, fit_subject_ids=fit_subjects)
        transformed = {
            subject_id: transform_bag(bag_by_subject[subject_id], scaler)
            for subject_id in [*fit_subjects, *val_subjects, fold.test_subject_id]
        }
        model = WPLISegmentAttentionMILModel(
            embedding_dim=config.embedding_dim,
            attention_hidden_dim=config.attention_hidden_dim,
            dropout=config.dropout,
            encoder_kind=config.encoder_kind,
            encoder_state_dict=normalized_encoder_states.get(fold.test_subject_id),
        ).to(device)
        if config.reset_bn_running_stats:
            _reset_batch_norm_running_stats(model.encoder)
        early_stopping = EarlyStopping(patience=config.patience, mode="min")
        rng = np.random.default_rng(config.seed + fold.fold_index * 10_003)

        epoch_counter = 0
        if config.stage1_epochs > 0:
            _set_encoder_trainable(model, False)
            optimizer = torch.optim.Adam(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=config.head_lr,
                weight_decay=config.weight_decay,
            )
            epoch_counter = _run_stage(
                stage="stage1_freeze_encoder",
                start_epoch=epoch_counter,
                n_epochs=config.stage1_epochs,
                model=model,
                optimizer=optimizer,
                fit_bags=[transformed[subject_id] for subject_id in fit_subjects],
                val_bags=[transformed[subject_id] for subject_id in val_subjects],
                config=config,
                device=device,
                rng=rng,
                fold=fold,
                fit_subjects=fit_subjects,
                val_subjects=val_subjects,
                history_rows=history_rows,
                early_stopping=early_stopping,
            )

        if config.stage2_epochs > 0:
            _set_encoder_trainable(model, True)
            optimizer = torch.optim.Adam(
                [
                    {"params": list(model.encoder.parameters()), "lr": config.encoder_lr},
                    {
                        "params": [
                            parameter
                            for name, parameter in model.named_parameters()
                            if not name.startswith("encoder.")
                        ],
                        "lr": config.stage2_head_lr,
                    },
                ],
                weight_decay=config.weight_decay,
            )
            _run_stage(
                stage="stage2_finetune",
                start_epoch=epoch_counter,
                n_epochs=config.stage2_epochs,
                model=model,
                optimizer=optimizer,
                fit_bags=[transformed[subject_id] for subject_id in fit_subjects],
                val_bags=[transformed[subject_id] for subject_id in val_subjects],
                config=config,
                device=device,
                rng=rng,
                fold=fold,
                fit_subjects=fit_subjects,
                val_subjects=val_subjects,
                history_rows=history_rows,
                early_stopping=early_stopping,
            )
        early_stopping.restore_best_weights(model)

        test_bag = transformed[fold.test_subject_id]
        y_score, aux, sampled = _predict_bag(model, test_bag, config, device=device, deterministic=True)
        y_pred = int(y_score >= 0.5)
        predictions.append(
            {
                "model": model_name,
                "fold_index": fold.fold_index,
                "subject_id": fold.test_subject_id,
                "y_true": int(test_bag.label),
                "y_score": y_score,
                "y_pred": y_pred,
                "architecture": "segment_attention_mil",
                "feature_kind": "fc-wpli",
                "encoder_kind": config.encoder_kind,
                "pretrained": fold.test_subject_id in normalized_encoder_states,
                "pretrained_transfer_mode": "stage1_freeze_stage2_finetune"
                if fold.test_subject_id in normalized_encoder_states
                else "none",
                "embedding_dim": config.embedding_dim,
                "dropout": config.dropout,
                "seed": config.seed,
            }
        )
        attention_rows.extend(
            _attention_summary_rows(
                fold=fold,
                bag=test_bag,
                sampled_indices=sampled,
                aux=aux,
                y_score=y_score,
                y_pred=y_pred,
            )
        )

    predictions_df = pd.DataFrame(predictions).sort_values("subject_id").reset_index(drop=True)
    metric_values = binary_classification_metrics(
        predictions_df["y_true"].to_numpy(dtype=int),
        predictions_df["y_score"].to_numpy(dtype=float),
    )
    metrics_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "architecture": "segment_attention_mil",
                "feature_kind": "fc-wpli",
                "encoder_kind": config.encoder_kind,
                "embedding_dim": config.embedding_dim,
                "dropout": config.dropout,
                "seed": config.seed,
                **metric_values,
            }
        ]
    )
    return predictions_df, metrics_df, pd.DataFrame(history_rows), pd.DataFrame(attention_rows)


def write_segment_mil_outputs(
    *,
    output_root: str | Path,
    run_name: str,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    loss_history: pd.DataFrame,
    attention_summary: pd.DataFrame,
) -> dict[str, Path]:
    root = Path(output_root)
    attention_suffix = _attention_summary_suffix(run_name)
    paths = {
        "predictions": root / "results" / "predictions" / f"dl_loso_predictions_{run_name}.csv",
        "metrics": root / "results" / "metrics" / f"dl_model_comparison_{run_name}.csv",
        "loss_history": root / "results" / "training_logs" / f"dl_loss_history_{run_name}.csv",
        "loss_curve": root / "results" / "figures" / f"dl_loss_curve_{run_name}.png",
        "attention_summary": root / "results" / "metrics" / f"wpli_segbarlow_attention_mil_attention_summary_{attention_suffix}.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    loss_history.to_csv(paths["loss_history"], index=False)
    attention_summary.to_csv(paths["attention_summary"], index=False)
    if not loss_history.empty:
        _plot_loss_history(loss_history, paths["loss_curve"])
    return paths


def _attention_summary_suffix(run_name: str) -> str:
    match = re.fullmatch(r"wpli_segbarlow_attention_mil_(seed[^_]+)(?:_(.+))?", run_name)
    if match is None:
        return run_name
    suffix = match.group(1)
    if match.group(2):
        suffix = f"{suffix}_{match.group(2)}"
    return suffix


def _run_stage(
    *,
    stage: str,
    start_epoch: int,
    n_epochs: int,
    model: WPLISegmentAttentionMILModel,
    optimizer: torch.optim.Optimizer,
    fit_bags: list[WPLISegmentBagRecord],
    val_bags: list[WPLISegmentBagRecord],
    config: SegmentMILTrainingConfig,
    device: torch.device,
    rng: np.random.Generator,
    fold: LOSOFold,
    fit_subjects: list[str],
    val_subjects: list[str],
    history_rows: list[dict[str, Any]],
    early_stopping: EarlyStopping,
) -> int:
    epoch_counter = start_epoch
    for local_epoch in range(1, n_epochs + 1):
        epoch_counter += 1
        order = rng.permutation(len(fit_bags))
        train_losses = []
        model.train()
        if config.freeze_bn:
            _set_batch_norm_eval(model.encoder)
        for bag_index in order:
            batch = _make_bag_batch(
                fit_bags[int(bag_index)],
                config,
                device=device,
                rng=rng,
                deterministic=False,
            )
            target = torch.as_tensor([[fit_bags[int(bag_index)].label]], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            probability, _ = model(batch)
            loss = F.binary_cross_entropy(probability, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        val_loss = _evaluate_loss(model, val_bags, config, device=device)
        is_best_epoch = early_stopping.best_score is None or val_loss < early_stopping.best_score - early_stopping.min_delta
        stopped_early = early_stopping.step(val_loss, model)
        history_rows.append(
            {
                "model": "wpli_segbarlow_attention_mil",
                "fold_index": fold.fold_index,
                "test_subject_id": fold.test_subject_id,
                "fit_subject_ids": ";".join(fit_subjects),
                "val_subject_ids": ";".join(val_subjects),
                "epoch": epoch_counter,
                "stage_epoch": local_epoch,
                "stage": stage,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "weight_decay": config.weight_decay,
                "is_best_epoch": bool(is_best_epoch),
                "stopped_early": bool(stopped_early),
                "architecture": "segment_attention_mil",
                "feature_kind": "fc-wpli",
                "encoder_kind": config.encoder_kind,
            }
        )
        if stopped_early:
            break
    return epoch_counter


def _evaluate_loss(
    model: WPLISegmentAttentionMILModel,
    bags: list[WPLISegmentBagRecord],
    config: SegmentMILTrainingConfig,
    *,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for bag in bags:
            batch = _make_bag_batch(bag, config, device=device, rng=None, deterministic=True)
            target = torch.as_tensor([[bag.label]], dtype=torch.float32, device=device)
            probability, _ = model(batch)
            losses.append(float(F.binary_cross_entropy(probability, target).detach().cpu().item()))
    return float(np.mean(losses)) if losses else np.nan


def _predict_bag(
    model: WPLISegmentAttentionMILModel,
    bag: WPLISegmentBagRecord,
    config: SegmentMILTrainingConfig,
    *,
    device: torch.device,
    deterministic: bool,
) -> tuple[float, dict[str, torch.Tensor | int], dict[str, np.ndarray]]:
    model.eval()
    sampled = _sample_indices_for_bag(bag, config, rng=None, deterministic=deterministic)
    batch = _make_bag_batch_from_indices(bag, sampled, device=device)
    with torch.no_grad():
        probability, aux = model(batch)
    return float(probability.detach().cpu().numpy()[0, 0]), aux, sampled


def _make_bag_batch(
    bag: WPLISegmentBagRecord,
    config: SegmentMILTrainingConfig,
    *,
    device: torch.device,
    rng: np.random.Generator | None,
    deterministic: bool,
) -> dict[str, torch.Tensor | object]:
    sampled = _sample_indices_for_bag(bag, config, rng=rng, deterministic=deterministic)
    return _make_bag_batch_from_indices(bag, sampled, device=device)


def _make_bag_batch_from_indices(
    bag: WPLISegmentBagRecord,
    sampled: Mapping[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor | object]:
    return {
        "wpli_eo_segments": torch.as_tensor(bag.eo_segments[sampled["EO"]], dtype=torch.float32, device=device),
        "wpli_ec_segments": torch.as_tensor(bag.ec_segments[sampled["EC"]], dtype=torch.float32, device=device),
        "y": torch.as_tensor([bag.label], dtype=torch.float32, device=device),
        "subject_id": bag.subject_id,
    }


def _sample_indices_for_bag(
    bag: WPLISegmentBagRecord,
    config: SegmentMILTrainingConfig,
    *,
    rng: np.random.Generator | None,
    deterministic: bool,
) -> dict[str, np.ndarray]:
    return {
        "EO": _sample_indices(
            bag.eo_segments.shape[0],
            config.max_segments_per_state,
            rng=rng,
            deterministic=deterministic,
            use_all=config.eval_use_all_segments and deterministic,
        ),
        "EC": _sample_indices(
            bag.ec_segments.shape[0],
            config.max_segments_per_state,
            rng=rng,
            deterministic=deterministic,
            use_all=config.eval_use_all_segments and deterministic,
        ),
    }


def _sample_indices(
    n_segments: int,
    max_segments: int | None,
    *,
    rng: np.random.Generator | None,
    deterministic: bool,
    use_all: bool,
) -> np.ndarray:
    if n_segments < 1:
        raise ValueError("Cannot sample from an empty segment bag.")
    if use_all or max_segments is None or max_segments >= n_segments:
        return np.arange(n_segments, dtype=int)
    if max_segments < 1:
        raise ValueError("max_segments_per_state must be at least 1 when provided.")
    if deterministic:
        return np.unique(np.linspace(0, n_segments - 1, max_segments, dtype=int))
    if rng is None:
        raise ValueError("rng is required for random segment subsampling.")
    return np.sort(rng.choice(n_segments, size=max_segments, replace=False)).astype(int)


def _attention_summary_rows(
    *,
    fold: LOSOFold,
    bag: WPLISegmentBagRecord,
    sampled_indices: Mapping[str, np.ndarray],
    aux: Mapping[str, torch.Tensor | int],
    y_score: float,
    y_pred: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state, key in (("EO", "eo_attention_weights"), ("EC", "ec_attention_weights")):
        weights = aux[key]
        if not isinstance(weights, torch.Tensor):
            raise TypeError(f"aux[{key!r}] must be a tensor.")
        for local_index, original_index in enumerate(sampled_indices[state]):
            metadata = list(bag.segment_metadata.get(state, []))
            meta = metadata[int(original_index)] if int(original_index) < len(metadata) else {}
            rows.append(
                {
                    "subject_id": bag.subject_id,
                    "fold_index": fold.fold_index,
                    "state": state,
                    "segment_index": int(meta.get("segment_index", int(original_index))),
                    "attention_weight": float(weights.detach().cpu().numpy()[local_index]),
                    "segment_start": meta.get("start_sample", pd.NA),
                    "segment_end": meta.get("end_sample", pd.NA),
                    "y_true": int(bag.label),
                    "y_score": y_score,
                    "y_pred": y_pred,
                }
            )
    return rows


def _segment_metadata(record: SegmentSSLRecord) -> dict[str, Any]:
    metadata = {
        "segment_index": int(record.segment_index),
        "source_path": str(record.source_path),
    }
    if record.source_path.exists():
        try:
            with np.load(record.source_path, allow_pickle=False) as payload:
                for key in ("start_sample", "end_sample", "source_set_path", "source_fdt_path"):
                    if key in payload:
                        value = payload[key]
                        metadata[key] = value.item() if value.shape == () else value.tolist()
        except Exception:
            pass
    return metadata


def _set_encoder_trainable(model: WPLISegmentAttentionMILModel, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable
    for name, parameter in model.named_parameters():
        if not name.startswith("encoder."):
            parameter.requires_grad = True


def _set_batch_norm_eval(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


def _reset_batch_norm_running_stats(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.reset_running_stats()


def _validate_segments(array: np.ndarray, name: str) -> None:
    observed = tuple(np.asarray(array).shape)
    if len(observed) != 3 or observed[1:] != (1891, 6):
        raise ValueError(f"{name} must be shaped (n_segments, 1891, 6), got {observed}.")


def _validate_config(config: SegmentMILTrainingConfig) -> None:
    if config.embedding_dim < 1:
        raise ValueError("embedding_dim must be positive.")
    if config.attention_hidden_dim < 1:
        raise ValueError("attention_hidden_dim must be positive.")
    if config.encoder_kind not in {"cnn", "linear"}:
        raise ValueError("encoder_kind must be 'cnn' or 'linear'.")
    if config.dropout < 0:
        raise ValueError("dropout must be non-negative.")
    if config.stage1_epochs < 0 or config.stage2_epochs < 0:
        raise ValueError("stage epochs must be non-negative.")
    if config.stage1_epochs + config.stage2_epochs < 1:
        raise ValueError("At least one training epoch is required.")
    for name, value in (
        ("head_lr", config.head_lr),
        ("encoder_lr", config.encoder_lr),
        ("stage2_head_lr", config.stage2_head_lr),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if config.patience < 1:
        raise ValueError("patience must be positive.")
    if config.max_segments_per_state is not None and config.max_segments_per_state < 1:
        raise ValueError("max_segments_per_state must be at least 1 when provided.")


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
