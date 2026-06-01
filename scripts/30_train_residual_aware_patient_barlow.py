import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.models.multimodal_model import MultimodalEEGModel
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.optimizers import build_swa_model, update_swa_batch_norm
from eeg_recovery.training.residual_aware_losses import residual_aware_multitask_loss
from eeg_recovery.training.residual_targets import (
    compute_signed_distance_from_label_table,
    fold_local_standardize_signed_distance,
)
from eeg_recovery.training.schedulers import EarlyStopping, build_reduce_on_plateau
from eeg_recovery.training.train_supervised import (
    _fit_state_scaler,
    _make_batch,
    _train_validation_subjects,
    load_supervised_feature_records,
    resolve_device,
    write_dl_outputs,
    write_loss_history_outputs,
)


SEEDS_10 = (0, 1, 2, 3, 4, 5, 7, 13, 21, 42)
WATCHED_SUBJECTS = ("sub09", "sub14", "sub05", "sub13")


@dataclass(frozen=True)
class ResidualAwareVariant:
    name: str
    lambda_reg: float
    lambda_rank: float
    lambda_soft: float


VARIANTS = {
    "reg": ResidualAwareVariant("reg", lambda_reg=0.3, lambda_rank=0.0, lambda_soft=0.0),
    "reg_soft": ResidualAwareVariant("reg_soft", lambda_reg=0.3, lambda_rank=0.0, lambda_soft=0.2),
    "reg_soft_rank": ResidualAwareVariant("reg_soft_rank", lambda_reg=0.3, lambda_rank=0.1, lambda_soft=0.2),
    "lowreg_soft_rank": ResidualAwareVariant("lowreg_soft_rank", lambda_reg=0.1, lambda_rank=0.1, lambda_soft=0.2),
    "highrank": ResidualAwareVariant("highrank", lambda_reg=0.3, lambda_rank=0.3, lambda_soft=0.1),
}


class ResidualAwarePatientBarlowModel(nn.Module):
    def __init__(
        self,
        *,
        feature_kind: str = "psd-fc-wpli",
        fusion: str = "gated",
        encoder_kind: str = "cnn",
        embedding_dim: int = 32,
        dropout: float = 0.0,
        residual_alpha: float = 0.5,
        residual_probability_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = MultimodalEEGModel(
            feature_kind=feature_kind,
            fusion=fusion,
            encoder_kind=encoder_kind,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        fused_dim = embedding_dim * len(self.backbone.branches)
        hidden_dim = max(4, min(32, fused_dim))
        self.classification_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_alpha = float(residual_alpha)
        self.residual_probability_scale = float(residual_probability_scale)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding = self.backbone.extract_embedding(batch)
        logits = self.classification_head(embedding)
        residual_score = self.residual_head(embedding)
        p_cls = torch.sigmoid(logits)
        p_residual = torch.sigmoid(self.residual_probability_scale * residual_score)
        p_final = self.residual_alpha * p_cls + (1.0 - self.residual_alpha) * p_residual
        return {
            "embedding": embedding,
            "classification_logits": logits,
            "classification_probability": p_cls,
            "residual_score": residual_score,
            "residual_probability": p_residual,
            "final_probability": p_final,
        }

    def load_patient_barlow_state(self, state: Mapping[str, torch.Tensor]) -> None:
        mapped = {
            f"backbone.{key}": value
            for key, value in state.items()
            if key.startswith("branch_models.")
        }
        incompatible = self.load_state_dict(mapped, strict=False)
        if incompatible.unexpected_keys:
            joined = ", ".join(incompatible.unexpected_keys)
            raise ValueError(f"Unexpected Patient-level Barlow state key(s): {joined}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Residual-aware multi-task fine-tuning for Patient-level Barlow SSL-CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["reg", "reg_soft", "reg_soft_rank"])
    parser.add_argument("--pretraining-mode", choices=("patient_barlow", "no_ssl"), default="patient_barlow")
    parser.add_argument("--reuse-ssl-encoders", action="store_true")
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--data-scope", default="all-patient")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--residual-alpha", type=float, default=0.5)
    parser.add_argument("--alpha-candidates", type=float, nargs="+", default=None)
    parser.add_argument("--residual-probability-scale", type=float, default=1.0)
    parser.add_argument("--use-swa", action="store_true")
    parser.add_argument("--swa-start-epoch", type=int, default=50)
    parser.add_argument("--swa-lr", type=float, default=5e-4)
    parser.add_argument("--save-supervised-fold-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument("--output-prefix", default="patient_barlow_residualaware")
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    label_table = load_supervised_label_table(path_config)
    records = load_supervised_feature_records(path_config, label_table, feature_kind="psd-fc-wpli")
    targets = compute_signed_distance_from_label_table(label_table, threshold=1.5)
    source_hash = _source_feature_manifest_hash_from_labels(label_table)
    device = resolve_device(args.device)

    for seed in args.seeds:
        for variant_name in args.variants:
            variant = VARIANTS[variant_name]
            run_name = _run_name(args.output_prefix, variant=variant.name, seed=seed, n_variants=len(args.variants), n_seeds=len(args.seeds))
            predictions, metrics, loss_history = _run_residual_aware_loso(
                records=records,
                targets=targets,
                output_root=path_config.output_root,
                source_feature_manifest_hash=source_hash,
                seed=seed,
                variant=variant,
                pretraining_mode=args.pretraining_mode,
                device=device,
                data_scope=args.data_scope,
                embedding_dim=args.embedding_dim,
                projection_dim=args.projection_dim,
                dropout=args.dropout,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                rank_margin=args.rank_margin,
                residual_alpha=args.residual_alpha,
                alpha_candidates=tuple(args.alpha_candidates or [args.residual_alpha]),
                residual_probability_scale=args.residual_probability_scale,
                use_swa=args.use_swa,
                swa_start_epoch=args.swa_start_epoch,
                swa_lr=args.swa_lr,
                save_supervised_fold_checkpoints=args.save_supervised_fold_checkpoints,
                checkpoint_tag=args.checkpoint_tag or args.output_prefix,
                reuse_ssl_encoders=args.reuse_ssl_encoders,
                reuse_only=args.reuse_only,
            )
            prediction_path, metric_path = write_dl_outputs(
                predictions,
                metrics,
                path_config.output_root,
                run_name=run_name,
            )
            write_loss_history_outputs(loss_history, path_config.output_root, run_name=run_name)
            _write_seed0_comparison_if_available(path_config.output_root)
            _write_watched_subject_scores(path_config.output_root)
            print(f"Wrote predictions: {prediction_path}")
            print(f"Wrote metrics: {metric_path}")


def _run_residual_aware_loso(
    *,
    records: list[Any],
    targets: pd.DataFrame,
    output_root: str | Path,
    source_feature_manifest_hash: str,
    seed: int,
    variant: ResidualAwareVariant,
    pretraining_mode: str,
    device: torch.device,
    data_scope: str,
    embedding_dim: int,
    projection_dim: int,
    dropout: float,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    rank_margin: float,
    residual_alpha: float,
    alpha_candidates: tuple[float, ...],
    residual_probability_scale: float,
    use_swa: bool,
    swa_start_epoch: int,
    swa_lr: float,
    save_supervised_fold_checkpoints: bool,
    checkpoint_tag: str,
    reuse_ssl_encoders: bool,
    reuse_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _seed_everything(seed)
    subject_ids = [record.subject_id for record in records]
    record_by_subject = {record.subject_id: record for record in records}
    folds = make_loso_folds(subject_ids)
    prediction_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    pretraining_metadata = _pretraining_mode_metadata(pretraining_mode, variant)
    model_name = str(pretraining_metadata["model_name"])

    for fold in folds:
        state: dict[str, torch.Tensor] | None = None
        if pretraining_metadata["requires_checkpoint"]:
            metadata = _patient_barlow_checkpoint_metadata(
                data_scope=data_scope,
                fold_index=fold.fold_index,
                test_subject_id=fold.test_subject_id,
                seed=seed,
                embedding_dim=embedding_dim,
                projection_dim=projection_dim,
                source_feature_manifest_hash=source_feature_manifest_hash,
            )
            checkpoint_path = _patient_barlow_checkpoint_path(output_root, metadata)
            if not reuse_ssl_encoders:
                raise ValueError("Residual-aware Patient-level Barlow requires --reuse-ssl-encoders.")
            state = _load_patient_barlow_checkpoint(checkpoint_path, expected_metadata=metadata)

        train_subjects = list(fold.train_subject_ids)
        fit_subjects, val_subjects = _train_validation_subjects(train_subjects, fold.fold_index)
        scaler = _fit_state_scaler(record_by_subject[subject_id] for subject_id in fit_subjects)
        target_scaler = fold_local_standardize_signed_distance(
            targets,
            fit_subject_ids=fit_subjects,
            transform_subject_ids=[*fit_subjects, *val_subjects, fold.test_subject_id],
        )
        train_batch = _make_residual_batch(
            (record_by_subject[subject_id] for subject_id in fit_subjects),
            fit_subjects,
            scaler,
            target_scaler.frame,
            device,
        )
        val_batch = _make_residual_batch(
            (record_by_subject[subject_id] for subject_id in val_subjects),
            val_subjects,
            scaler,
            target_scaler.frame,
            device,
        )
        test_batch = _make_batch(
            [record_by_subject[fold.test_subject_id]],
            scaler,
            device,
            "multimodal",
        )

        model = ResidualAwarePatientBarlowModel(
            embedding_dim=embedding_dim,
            dropout=dropout,
            residual_alpha=residual_alpha,
            residual_probability_scale=residual_probability_scale,
        ).to(device)
        if state is not None:
            model.load_patient_barlow_state(state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = build_reduce_on_plateau(optimizer, mode="min", patience=max(1, patience // 2))
        swa_model = build_swa_model(model) if use_swa else None
        swa_scheduler = (
            torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)
            if use_swa
            else None
        )
        early_stopping = EarlyStopping(patience=patience, mode="min")

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            outputs = model(train_batch)
            loss, components = residual_aware_multitask_loss(
                outputs,
                train_batch,
                lambda_reg=variant.lambda_reg,
                lambda_rank=variant.lambda_rank,
                lambda_soft=variant.lambda_soft,
                rank_margin=rank_margin,
            )
            loss.backward()
            optimizer.step()
            train_loss = float(loss.detach().cpu().item())
            val_loss, val_components = _evaluate_residual_loss(model, val_batch, variant, rank_margin)
            if _should_update_swa(epoch=epoch, use_swa=use_swa, swa_start_epoch=swa_start_epoch):
                if swa_model is None or swa_scheduler is None:
                    raise RuntimeError("SWA is enabled but SWA state was not initialized.")
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step(val_loss)
            stopped = early_stopping.step(val_loss, model)
            loss_rows.append(
                {
                    "model": model_name,
                    "fold_index": fold.fold_index,
                    "test_subject_id": fold.test_subject_id,
                    "fit_subject_ids": ";".join(fit_subjects),
                    "val_subject_ids": ";".join(val_subjects),
                    "epoch": epoch,
                    "training_phase": "residual_aware_supervised",
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "lambda_reg": variant.lambda_reg,
                    "lambda_rank": variant.lambda_rank,
                    "lambda_soft": variant.lambda_soft,
                    "rank_margin": rank_margin,
                    "use_swa": bool(use_swa),
                    "swa_start_epoch": int(swa_start_epoch),
                    "swa_lr": float(swa_lr),
                    "swa_n_averaged": (
                        int(swa_model.n_averaged.detach().cpu().item())
                        if swa_model is not None
                        else 0
                    ),
                    "target_mean": target_scaler.mean,
                    "target_std": target_scaler.std,
                    "target_tau": target_scaler.tau,
                    **{f"train_{key}": float(value.cpu().item()) for key, value in components.items()},
                    **{f"val_{key}": value for key, value in val_components.items()},
                    "stopped_early": bool(stopped),
                }
            )
            if stopped:
                break

        eval_model: nn.Module = model
        if _has_swa_weights(swa_model):
            update_swa_batch_norm([train_batch], swa_model)
            eval_model = swa_model
        else:
            early_stopping.restore_best_weights(model)
        eval_model.eval()
        with torch.no_grad():
            val_outputs = eval_model(val_batch)
            selected_alpha = _select_alpha_on_validation(
                classification_scores=val_outputs["classification_probability"].detach().cpu().reshape(-1).numpy(),
                residual_scores=val_outputs["residual_probability"].detach().cpu().reshape(-1).numpy(),
                y_true=val_batch["y"].detach().cpu().reshape(-1).numpy(),
                candidates=alpha_candidates,
            )
            outputs = eval_model(test_batch)
            cls_probability = float(outputs["classification_probability"].detach().cpu().numpy()[0, 0])
            residual_probability = float(outputs["residual_probability"].detach().cpu().numpy()[0, 0])
            probability = selected_alpha * cls_probability + (1.0 - selected_alpha) * residual_probability
            residual_score = float(outputs["residual_score"].detach().cpu().numpy()[0, 0])
        if save_supervised_fold_checkpoints:
            _save_supervised_fold_checkpoint(
                output_root=output_root,
                checkpoint_tag=checkpoint_tag,
                model=eval_model,
                scaler=scaler,
                seed=seed,
                fold_index=fold.fold_index,
                test_subject_id=fold.test_subject_id,
                fit_subject_ids=fit_subjects,
                val_subject_ids=val_subjects,
                variant=variant,
                pretraining_metadata=pretraining_metadata,
                embedding_dim=embedding_dim,
                dropout=dropout,
                rank_margin=rank_margin,
                residual_alpha=residual_alpha,
                selected_alpha=selected_alpha,
                alpha_candidates=alpha_candidates,
                residual_probability_scale=residual_probability_scale,
                use_swa=use_swa,
                swa_start_epoch=swa_start_epoch,
                swa_lr=swa_lr,
                target_scaler={
                    "mean": target_scaler.mean,
                    "std": target_scaler.std,
                    "tau": target_scaler.tau,
                },
            )
        test_record = record_by_subject[fold.test_subject_id]
        prediction_rows.append(
            {
                "model": model_name,
                "fold_index": fold.fold_index,
                "subject_id": fold.test_subject_id,
                "y_true": int(test_record.label),
                "y_score": probability,
                "y_pred": int(probability >= 0.5),
                "classification_y_score": cls_probability,
                "residual_y_score": residual_probability,
                "residual_score_z": residual_score,
                "signed_distance": float(targets.loc[fold.test_subject_id, "signed_distance"]),
                "pretrained": bool(pretraining_metadata["pretrained"]),
                "pretrained_transfer_mode": str(pretraining_metadata["pretrained_transfer_mode"]),
                "architecture": "multimodal",
                "feature_kind": "psd-fc-wpli",
                "fusion": "gated",
                "encoder_kind": "cnn",
                "embedding_dim": embedding_dim,
                "dropout": dropout,
                "seed": seed,
                "lambda_reg": variant.lambda_reg,
                "lambda_rank": variant.lambda_rank,
                "lambda_soft": variant.lambda_soft,
                "rank_margin": rank_margin,
                "residual_alpha": residual_alpha,
                "selected_alpha": selected_alpha,
                "alpha_candidates": ";".join(str(candidate) for candidate in alpha_candidates),
                "residual_probability_scale": residual_probability_scale,
                "use_swa": bool(use_swa),
                "swa_start_epoch": int(swa_start_epoch),
                "swa_lr": float(swa_lr),
                "swa_n_averaged": (
                    int(swa_model.n_averaged.detach().cpu().item())
                    if swa_model is not None
                    else 0
                ),
            }
        )

    predictions = pd.DataFrame(prediction_rows).sort_values("subject_id").reset_index(drop=True)
    metrics_values = binary_classification_metrics(
        predictions["y_true"].to_numpy(dtype=int),
        predictions["y_score"].to_numpy(dtype=float),
    )
    metrics = pd.DataFrame(
        [
            {
                "model": model_name,
                "architecture": "multimodal",
                "feature_kind": "psd-fc-wpli",
                "fusion": "gated",
                "encoder_kind": "cnn",
                "pretrained": bool(pretraining_metadata["pretrained"]),
                "pretrained_transfer_mode": str(pretraining_metadata["pretrained_transfer_mode"]),
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "embedding_dim": embedding_dim,
                "dropout": dropout,
                "seed": seed,
                "lambda_reg": variant.lambda_reg,
                "lambda_rank": variant.lambda_rank,
                "lambda_soft": variant.lambda_soft,
                "rank_margin": rank_margin,
                "residual_alpha": residual_alpha,
                "alpha_candidates": ";".join(str(candidate) for candidate in alpha_candidates),
                "residual_probability_scale": residual_probability_scale,
                "use_swa": bool(use_swa),
                "swa_start_epoch": int(swa_start_epoch),
                "swa_lr": float(swa_lr),
                **metrics_values,
            }
        ]
    )
    return predictions, metrics, pd.DataFrame(loss_rows)


def _select_alpha_on_validation(
    *,
    classification_scores: Iterable[float],
    residual_scores: Iterable[float],
    y_true: Iterable[int | float],
    candidates: Iterable[float],
) -> float:
    cls = np.asarray(list(classification_scores), dtype=float)
    residual = np.asarray(list(residual_scores), dtype=float)
    y = np.asarray(list(y_true), dtype=int)
    candidate_values = [float(candidate) for candidate in candidates]
    if cls.shape != residual.shape or cls.shape[0] != y.shape[0]:
        raise ValueError("classification_scores, residual_scores, and y_true must have matching lengths.")
    if not candidate_values:
        raise ValueError("alpha candidates must not be empty.")
    best_alpha = candidate_values[0]
    best_key: tuple[float, float, float] | None = None
    for alpha in candidate_values:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha candidates must be in [0, 1].")
        score = alpha * cls + (1.0 - alpha) * residual
        metrics = binary_classification_metrics(y, score)
        key = (
            float(metrics["balanced_accuracy"]),
            -float(metrics["brier_score"]),
            float(metrics["roc_auc"]) if np.isfinite(metrics["roc_auc"]) else -1.0,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_alpha = alpha
    return best_alpha


def _should_update_swa(*, epoch: int, use_swa: bool, swa_start_epoch: int) -> bool:
    if swa_start_epoch <= 0:
        raise ValueError("swa_start_epoch must be positive.")
    return bool(use_swa and epoch >= swa_start_epoch)


def _has_swa_weights(swa_model: nn.Module | None) -> bool:
    if swa_model is None or not hasattr(swa_model, "n_averaged"):
        return False
    n_averaged = getattr(swa_model, "n_averaged")
    if isinstance(n_averaged, torch.Tensor):
        return int(n_averaged.detach().cpu().item()) > 0
    return int(n_averaged) > 0


def _pretraining_mode_metadata(pretraining_mode: str, variant: ResidualAwareVariant) -> dict[str, Any]:
    if pretraining_mode == "patient_barlow":
        return {
            "model_name": f"patient_barlow_residualaware_{variant.name}",
            "pretrained": True,
            "pretrained_transfer_mode": "residual_aware_finetune",
            "requires_checkpoint": True,
        }
    if pretraining_mode == "no_ssl":
        return {
            "model_name": f"no_ssl_residualaware_{variant.name}",
            "pretrained": False,
            "pretrained_transfer_mode": "residual_aware_multitask_from_scratch",
            "requires_checkpoint": False,
        }
    raise ValueError("pretraining_mode must be patient_barlow or no_ssl.")


def _supervised_checkpoint_path(
    output_root: str | Path,
    *,
    checkpoint_tag: str,
    seed: int,
    fold_index: int,
    test_subject_id: str,
) -> Path:
    safe_tag = _safe_filename_token(checkpoint_tag)
    safe_subject = _safe_filename_token(normalize_subject_id(test_subject_id))
    return (
        Path(output_root)
        / "results"
        / "checkpoints"
        / "supervised"
        / safe_tag
        / f"{safe_tag}_seed{int(seed)}_fold{int(fold_index):02d}_test_{safe_subject}.pt"
    )


def _save_supervised_fold_checkpoint(
    *,
    output_root: str | Path,
    checkpoint_tag: str,
    model: nn.Module,
    scaler: Any,
    seed: int,
    fold_index: int,
    test_subject_id: str,
    fit_subject_ids: list[str],
    val_subject_ids: list[str],
    variant: ResidualAwareVariant,
    pretraining_metadata: Mapping[str, Any],
    embedding_dim: int,
    dropout: float,
    rank_margin: float,
    residual_alpha: float,
    selected_alpha: float,
    alpha_candidates: tuple[float, ...],
    residual_probability_scale: float,
    use_swa: bool,
    swa_start_epoch: int,
    swa_lr: float,
    target_scaler: Mapping[str, float],
) -> Path:
    path = _supervised_checkpoint_path(
        output_root,
        checkpoint_tag=checkpoint_tag,
        seed=seed,
        fold_index=fold_index,
        test_subject_id=test_subject_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_type": "residual_aware_supervised_fold",
        "state_dict": _model_state_dict_for_checkpoint(model),
        "state_scaler": _scaler_to_checkpoint_payload(scaler),
        "metadata": {
            "model_group": checkpoint_tag,
            "model_name": pretraining_metadata["model_name"],
            "pretrained": bool(pretraining_metadata["pretrained"]),
            "pretrained_transfer_mode": str(pretraining_metadata["pretrained_transfer_mode"]),
            "variant": variant.name,
            "lambda_reg": float(variant.lambda_reg),
            "lambda_rank": float(variant.lambda_rank),
            "lambda_soft": float(variant.lambda_soft),
            "rank_margin": float(rank_margin),
            "seed": int(seed),
            "fold_index": int(fold_index),
            "test_subject_id": normalize_subject_id(test_subject_id),
            "fit_subject_ids": [normalize_subject_id(subject_id) for subject_id in fit_subject_ids],
            "val_subject_ids": [normalize_subject_id(subject_id) for subject_id in val_subject_ids],
            "architecture": "multimodal",
            "feature_kind": "psd-fc-wpli",
            "fusion": "gated",
            "encoder_kind": "cnn",
            "embedding_dim": int(embedding_dim),
            "dropout": float(dropout),
            "residual_alpha": float(residual_alpha),
            "selected_alpha": float(selected_alpha),
            "alpha_candidates": [float(candidate) for candidate in alpha_candidates],
            "residual_probability_scale": float(residual_probability_scale),
            "use_swa": bool(use_swa),
            "swa_start_epoch": int(swa_start_epoch),
            "swa_lr": float(swa_lr),
            "target_scaler": dict(target_scaler),
        },
    }
    torch.save(payload, path)
    return path


def _model_state_dict_for_checkpoint(model: nn.Module) -> dict[str, torch.Tensor]:
    if hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module):
        state_dict = getattr(model, "module").state_dict()
    else:
        state_dict = model.state_dict()
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def _scaler_to_checkpoint_payload(scaler: Any) -> Any:
    if isinstance(scaler, dict):
        return {
            key: {
                "mean": torch.as_tensor(mean).detach().cpu(),
                "std": torch.as_tensor(std).detach().cpu(),
            }
            for key, (mean, std) in scaler.items()
        }
    mean, std = scaler
    return {
        "mean": torch.as_tensor(mean).detach().cpu(),
        "std": torch.as_tensor(std).detach().cpu(),
    }


def _make_residual_batch(
    records: Iterable[Any],
    subject_ids: list[str],
    scaler: Any,
    target_frame: pd.DataFrame,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = _make_batch(records, scaler, device, "multimodal")
    ordered = [normalize_subject_id(subject_id) for subject_id in subject_ids]
    batch["signed_distance_z"] = torch.as_tensor(
        target_frame.loc[ordered, "signed_distance_z"].to_numpy(dtype=np.float32).reshape(-1, 1),
        device=device,
    )
    batch["soft_y"] = torch.as_tensor(
        target_frame.loc[ordered, "soft_y"].to_numpy(dtype=np.float32).reshape(-1, 1),
        device=device,
    )
    return batch


def _evaluate_residual_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    variant: ResidualAwareVariant,
    rank_margin: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        loss, components = residual_aware_multitask_loss(
            model(batch),
            batch,
            lambda_reg=variant.lambda_reg,
            lambda_rank=variant.lambda_rank,
            lambda_soft=variant.lambda_soft,
            rank_margin=rank_margin,
        )
    return float(loss.detach().cpu().item()), {
        key: float(value.detach().cpu().item())
        for key, value in components.items()
    }


def _residual_aware_output_paths(output_root: str | Path, *, output_tag: str) -> tuple[Path, Path]:
    root = Path(output_root)
    safe = _safe_filename_token(output_tag)
    return (
        root / "results" / "predictions" / f"dl_loso_predictions_{safe}.csv",
        root / "results" / "metrics" / f"dl_model_comparison_{safe}.csv",
    )


def _run_name(prefix: str, *, variant: str, seed: int, n_variants: int, n_seeds: int) -> str:
    if n_variants == 1 and n_seeds == 1 and prefix.endswith(f"seed{seed}"):
        return prefix
    if n_variants == 1:
        return f"{prefix}_seed{seed}"
    return f"{prefix}_{variant}_seed{seed}"


def _patient_barlow_checkpoint_metadata(
    *,
    data_scope: str,
    fold_index: int,
    test_subject_id: str,
    seed: int,
    embedding_dim: int,
    projection_dim: int,
    source_feature_manifest_hash: str,
) -> dict[str, Any]:
    normalized_test_subject = normalize_subject_id(test_subject_id)
    return {
        "checkpoint_type": "patient_barlow_ssl_encoder",
        "ssl_objective": "barlow",
        "ssl_level": "patient_level",
        "data_scope": data_scope,
        "fold_index": int(fold_index),
        "test_subject_id": normalized_test_subject,
        "excluded_subject_id": normalized_test_subject,
        "seed": int(seed),
        "encoder_kind": "cnn",
        "embedding_dim": int(embedding_dim),
        "projection_dim": int(projection_dim),
        "feature_kind": "psd-fc-wpli",
        "fusion": "gated",
        "source_feature_manifest_hash": source_feature_manifest_hash,
    }


def _patient_barlow_checkpoint_path(output_root: str | Path, metadata: Mapping[str, Any]) -> Path:
    filename = (
        "patient_barlow_ssl_encoder_"
        f"{_safe_filename_token(str(metadata['data_scope']))}_"
        f"fold{int(metadata['fold_index']):02d}_{_safe_filename_token(str(metadata['test_subject_id']))}_"
        f"seed{int(metadata['seed'])}_emb{int(metadata['embedding_dim'])}_"
        f"proj{int(metadata['projection_dim'])}.pt"
    )
    return Path(output_root) / "results" / "checkpoints" / "ssl_encoders" / filename


def _load_patient_barlow_checkpoint(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing reusable Patient-level Barlow checkpoint: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if payload.get("checkpoint_type") != "patient_barlow_ssl_encoder":
        raise ValueError(f"Unexpected checkpoint_type in {path}: {payload.get('checkpoint_type')!r}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Patient-level Barlow checkpoint missing metadata: {path}")
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        detail = "; ".join(f"{key}: got {got!r}, expected {expected!r}" for key, (got, expected) in mismatches.items())
        raise ValueError(f"Patient-level Barlow checkpoint metadata mismatch for {path}: {detail}")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Patient-level Barlow checkpoint missing state_dict: {path}")
    return state


def _source_feature_manifest_hash_from_labels(label_table: pd.DataFrame) -> str:
    # Must match scripts/29_train_patient_barlow_stabilized.py for the locally cached checkpoints.
    import hashlib
    from eeg_recovery.config import load_path_config
    from eeg_recovery.io.index import build_eeg_file_index
    from eeg_recovery.training.train_feature_ssl import (
        build_feature_ssl_pair_records_from_feature_records,
        compute_feature_records_for_eeg_records,
    )
    from eeg_recovery.training.train_ssl import select_ssl_records

    config = load_path_config(PROJECT_ROOT / "configs" / "paths.example.yaml")
    supervised_ids = label_table["subject_id"].tolist()
    eeg_records = build_eeg_file_index(
        config.patient_eeg_root,
        config.health_eeg_root,
        supervised_subject_ids=supervised_ids,
        validate_supervised_baseline=True,
    )
    eeg_records = select_ssl_records(eeg_records, data_scope="all-patient")
    feature_records = compute_feature_records_for_eeg_records(config, eeg_records, feature_kind="psd-fc-wpli")
    pairs = build_feature_ssl_pair_records_from_feature_records(eeg_records, feature_records)
    hasher = hashlib.sha256()
    for pair in sorted(pairs, key=lambda item: (item.group, item.stage, item.subject_key)):
        hasher.update(f"{pair.group}|{pair.subject_id}|{pair.subject_key}|{pair.stage}|".encode("utf-8"))
        for branch in sorted(pair.modalities):
            eo, ec = pair.modalities[branch]
            hasher.update(f"{branch}|{eo.shape}|{ec.shape}|{eo.dtype}|{ec.dtype}|".encode("utf-8"))
    return hasher.hexdigest()


def _write_seed0_comparison_if_available(output_root: str | Path) -> None:
    metric_dir = Path(output_root) / "results" / "metrics"
    rows = []
    for path in sorted(metric_dir.glob("dl_model_comparison_patient_barlow_residualaware*_seed0.csv")):
        row = pd.read_csv(path).iloc[0].to_dict()
        row["model_group"] = path.stem.replace("dl_model_comparison_", "")
        row["source_file"] = path.name
        rows.append(row)
    prior = metric_dir / "dl_model_comparison_feature_ssl_all-patient_psd-fc-wpli_gated_cnn_barlow_finetune_seed0_pre50_temp0_2_noise0_02_mask0_01_bs8_sup100_proj32.csv"
    if prior.exists():
        row = pd.read_csv(prior).iloc[0].to_dict()
        row["model_group"] = "prior_patient_barlow_seed0"
        row["source_file"] = prior.name
        rows.append(row)
    no_ssl = metric_dir / "dl_model_comparison_no_ssl_psdfcwpli_gated_cnn_rerun_20260531_seed0.csv"
    if no_ssl.exists():
        row = pd.read_csv(no_ssl).iloc[0].to_dict()
        row["model_group"] = "no_ssl_rerun_seed0"
        row["source_file"] = no_ssl.name
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(metric_dir / "patient_barlow_residualaware_seed0_comparison.csv", index=False)


def _write_watched_subject_scores(output_root: str | Path) -> None:
    pred_dir = Path(output_root) / "results" / "predictions"
    frames = []
    for path in sorted(pred_dir.glob("dl_loso_predictions_patient_barlow_residualaware*_seed*.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame[frame["subject_id"].isin(WATCHED_SUBJECTS)])
    if frames:
        watched = pd.concat(frames, ignore_index=True)
        watched["error"] = (watched["y_true"].astype(int) != watched["y_pred"].astype(int)).astype(int)
        watched.to_csv(
            Path(output_root) / "results" / "metrics" / "residual_aware_patient_barlow_watched_subject_scores.csv",
            index=False,
        )


def _safe_filename_token(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value)
    safe = safe.strip("_-")
    if not safe:
        raise ValueError("filename token must contain at least one safe character.")
    return safe


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
