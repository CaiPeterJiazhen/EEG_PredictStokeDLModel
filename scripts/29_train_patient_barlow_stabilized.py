from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.io.index import build_eeg_file_index
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.training.loso import make_loso_folds
from eeg_recovery.training.metrics import binary_classification_metrics
from eeg_recovery.training.train_feature_ssl import (
    FeatureSSLPairRecord,
    FeatureSSLTrainingConfig,
    build_feature_ssl_pair_records_from_feature_records,
    compute_feature_records_for_eeg_records,
    feature_ssl_pairs_for_scope,
    run_feature_ssl_pretraining,
)
from eeg_recovery.training.train_ssl import select_ssl_records
from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    load_supervised_feature_records,
    run_loso_supervised_with_history,
    write_dl_outputs,
    write_loss_history_outputs,
)


DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5, 7, 13, 21, 42)
REFERENCE_PATIENT_BARLOW_GLOB = (
    "dl_model_comparison_feature_ssl_all-patient_psd-fc-wpli_gated_cnn_"
    "barlow_finetune_seed0_pre50_temp0_2_noise0_02_mask0_01_bs8_sup100_proj32*.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Patient-level Barlow SSL-CNN stabilized supervised fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--reuse-ssl-encoders", action="store_true")
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--data-scope", default="all-patient")
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--pretrain-batch-size", type=int, default=8)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--feature-mask-prob", type=float, default=0.01)
    parser.add_argument("--finetune-schedule", choices=("swa_only", "sam_only", "staged_sam_swa"), default="staged_sam_swa")
    parser.add_argument("--optimizer", dest="optimizer_name", choices=("adam", "adamw", "sam_adamw"), default="sam_adamw")
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--use-swa", action="store_true")
    parser.add_argument("--swa-start-epoch", type=int, default=50)
    parser.add_argument("--swa-lr", type=float, default=5e-4)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=20)
    parser.add_argument("--freeze-conv-backbone", action="store_true")
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    labels = load_supervised_label_table(path_config)
    supervised_ids = labels["subject_id"].tolist()
    supervised_records = load_supervised_feature_records(
        path_config,
        labels,
        feature_kind="psd-fc-wpli",
    )
    eeg_records = build_eeg_file_index(
        path_config.patient_eeg_root,
        path_config.health_eeg_root,
        supervised_subject_ids=supervised_ids,
        validate_supervised_baseline=True,
    )
    eeg_records = select_ssl_records(eeg_records, data_scope=args.data_scope)
    feature_records = compute_feature_records_for_eeg_records(
        path_config,
        eeg_records,
        feature_kind="psd-fc-wpli",
    )
    all_ssl_pairs = build_feature_ssl_pair_records_from_feature_records(eeg_records, feature_records)
    source_manifest_hash = _source_feature_manifest_hash(all_ssl_pairs)
    folds = make_loso_folds([record.subject_id for record in supervised_records])

    for seed in args.seeds:
        pretrained_state_by_test_subject: dict[str, Mapping[str, torch.Tensor]] = {}
        missing_checkpoints: list[tuple[Path, dict[str, Any], list[FeatureSSLPairRecord]]] = []
        ssl_history_frames: list[pd.DataFrame] = []
        for fold in folds:
            ssl_pairs = feature_ssl_pairs_for_scope(
                all_ssl_pairs,
                data_scope=args.data_scope,
                strict_loso_test_subject_id=fold.test_subject_id,
            )
            metadata = _patient_barlow_checkpoint_metadata(
                data_scope=args.data_scope,
                fold_index=fold.fold_index,
                test_subject_id=fold.test_subject_id,
                seed=seed,
                embedding_dim=args.embedding_dim,
                projection_dim=args.projection_dim,
                source_feature_manifest_hash=source_manifest_hash,
            )
            checkpoint_path = _patient_barlow_checkpoint_path(path_config.output_root, metadata)
            if args.reuse_ssl_encoders and checkpoint_path.exists():
                pretrained_state_by_test_subject[fold.test_subject_id] = _load_patient_barlow_checkpoint(
                    checkpoint_path,
                    expected_metadata=metadata,
                    map_location="cpu",
                )
            else:
                missing_checkpoints.append((checkpoint_path, metadata, ssl_pairs))

        if missing_checkpoints:
            if args.reuse_only:
                missing = "\n".join(str(item[0]) for item in missing_checkpoints[:5])
                raise FileNotFoundError(
                    "Missing reusable Patient-level Barlow checkpoint(s). "
                    "Run without --reuse-only to build the SSL cache only.\n"
                    f"{missing}"
                )
            for checkpoint_path, metadata, ssl_pairs in missing_checkpoints:
                ssl_config = FeatureSSLTrainingConfig(
                    data_scope=args.data_scope,
                    feature_kind="psd-fc-wpli",
                    fusion="gated",
                    encoder_kind="cnn",
                    epochs=args.pretrain_epochs,
                    batch_size=args.pretrain_batch_size,
                    embedding_dim=args.embedding_dim,
                    projection_dim=args.projection_dim,
                    dropout=args.dropout,
                    lr=args.pretrain_lr,
                    ssl_objective="barlow",
                    temperature=args.temperature,
                    noise_std=args.noise_std,
                    feature_mask_prob=args.feature_mask_prob,
                    device=args.device,
                    seed=seed,
                )
                state, history = run_feature_ssl_pretraining(ssl_pairs, ssl_config)
                _save_patient_barlow_checkpoint(checkpoint_path, state, metadata)
                history = history.copy()
                history["checkpoint_path"] = str(checkpoint_path)
                history["ssl_level"] = "patient_level"
                ssl_history_frames.append(history)
            _write_ssl_cache_history(path_config.output_root, seed, ssl_history_frames)
            print("Patient-level Barlow SSL cache was incomplete; generated missing checkpoints only.")
            print("Re-run with --reuse-ssl-encoders --reuse-only to start supervised fine-tuning.")
            continue

        if not pretrained_state_by_test_subject:
            raise RuntimeError("No Patient-level Barlow checkpoints were loaded.")

        run_name = _run_name_for_seed(args.output_tag, seed=seed, n_seeds=len(args.seeds), schedule=args.finetune_schedule)
        training_config = SupervisedTrainingConfig(
            architecture="multimodal",
            feature_kind="psd-fc-wpli",
            fusion="gated",
            encoder_kind="cnn",
            device=args.device,
            epochs=args.epochs,
            patience=args.patience,
            lr=0.002,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer_name,
            use_swa=args.use_swa,
            swa_start_epoch=args.swa_start_epoch,
            swa_lr=args.swa_lr,
            finetune_schedule=args.finetune_schedule,
            encoder_lr=args.encoder_lr,
            head_lr=args.lr_head,
            freeze_encoder_epochs=args.freeze_encoder_epochs,
            freeze_pretrained_encoder_epochs=args.freeze_encoder_epochs,
            freeze_conv_backbone=args.freeze_conv_backbone,
            sam_rho=args.sam_rho,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            seed=seed,
            pretrained_transfer_mode="finetune",
        )
        predictions, metrics, loss_history = run_loso_supervised_with_history(
            supervised_records,
            training_config,
            pretrained_state_by_test_subject=pretrained_state_by_test_subject,
        )
        prediction_path, metric_path = write_dl_outputs(
            predictions,
            metrics,
            output_root=path_config.output_root,
            run_name=run_name,
        )
        write_loss_history_outputs(
            loss_history,
            output_root=path_config.output_root,
            run_name=run_name,
        )
        _write_seed0_comparison_if_available(path_config.output_root)
        print(f"Wrote predictions: {prediction_path}")
        print(f"Wrote metrics: {metric_path}")


def _patient_barlow_stabilized_output_paths(output_root: str | Path, *, output_tag: str) -> tuple[Path, Path]:
    root = Path(output_root)
    safe = _safe_filename_token(output_tag)
    return (
        root / "results" / "predictions" / f"dl_loso_predictions_{safe}.csv",
        root / "results" / "metrics" / f"dl_model_comparison_{safe}.csv",
    )


def _run_name_for_seed(output_tag: str | None, *, seed: int, n_seeds: int, schedule: str) -> str:
    if output_tag is None:
        return f"patient_barlow_{schedule}_seed{seed}"
    if "{seed}" in output_tag:
        return output_tag.format(seed=seed)
    if n_seeds > 1 and f"seed{seed}" not in output_tag:
        return f"{output_tag}_seed{seed}"
    return output_tag


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
    test_subject = _safe_filename_token(str(metadata["test_subject_id"]))
    filename = (
        "patient_barlow_ssl_encoder_"
        f"{_safe_filename_token(str(metadata['data_scope']))}_"
        f"fold{int(metadata['fold_index']):02d}_{test_subject}_"
        f"seed{int(metadata['seed'])}_emb{int(metadata['embedding_dim'])}_"
        f"proj{int(metadata['projection_dim'])}.pt"
    )
    return Path(output_root) / "results" / "checkpoints" / "ssl_encoders" / filename


def _save_patient_barlow_checkpoint(
    path: str | Path,
    state_dict: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_type": "patient_barlow_ssl_encoder",
        "metadata": {
            **dict(metadata),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "state_dict": {key: value.detach().cpu() for key, value in state_dict.items()},
    }
    torch.save(payload, path)


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
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Patient-level Barlow checkpoint payload: {path}")
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


def _source_feature_manifest_hash(pairs: list[FeatureSSLPairRecord]) -> str:
    hasher = hashlib.sha256()
    for pair in sorted(pairs, key=lambda item: (item.group, item.stage, item.subject_key)):
        hasher.update(f"{pair.group}|{pair.subject_id}|{pair.subject_key}|{pair.stage}|".encode("utf-8"))
        for branch in sorted(pair.modalities):
            eo, ec = pair.modalities[branch]
            hasher.update(f"{branch}|{eo.shape}|{ec.shape}|{eo.dtype}|{ec.dtype}|".encode("utf-8"))
    return hasher.hexdigest()


def _write_ssl_cache_history(output_root: str | Path, seed: int, history_frames: list[pd.DataFrame]) -> None:
    if not history_frames:
        return
    path = (
        Path(output_root)
        / "results"
        / "ssl"
        / f"patient_barlow_ssl_cache_history_seed{int(seed)}.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(history_frames, ignore_index=True).to_csv(path, index=False)


def _write_seed0_comparison_if_available(output_root: str | Path) -> None:
    root = Path(output_root)
    metric_dir = root / "results" / "metrics"
    rows = []
    candidates = {
        "patient_barlow_swa_seed0": metric_dir / "dl_model_comparison_patient_barlow_swa_seed0.csv",
        "patient_barlow_sam_seed0": metric_dir / "dl_model_comparison_patient_barlow_sam_seed0.csv",
        "patient_barlow_staged_sam_swa_seed0": metric_dir / "dl_model_comparison_patient_barlow_staged_sam_swa_seed0.csv",
    }
    prior = sorted(metric_dir.glob(REFERENCE_PATIENT_BARLOW_GLOB))
    if prior:
        candidates["prior_patient_barlow_seed0"] = prior[0]
    for model_group, path in candidates.items():
        if not path.exists():
            continue
        metric = pd.read_csv(path).iloc[0].to_dict()
        metric["model_group"] = model_group
        metric["source_file"] = path.name
        rows.append(metric)
    if not rows:
        return
    comparison = pd.DataFrame(rows)
    comparison_path = metric_dir / "patient_barlow_stabilized_seed0_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    _write_seed0_doc(root, comparison)


def _write_seed0_doc(output_root: Path, comparison: pd.DataFrame) -> None:
    doc_path = output_root / "docs" / "patient_barlow_stabilized_seed0_results.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    metric_columns = [
        column
        for column in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc", "brier_score")
        if column in comparison.columns
    ]
    table = comparison[["model_group", *metric_columns, "source_file"]].to_markdown(index=False)
    doc_path.write_text(
        "# Patient-level Barlow Stabilized Seed0 Results\n\n"
        "This pilot reuses fold-specific Patient-level Barlow encoder checkpoints and changes only supervised fine-tuning.\n"
        "No qEEG branch, MIL, Dual encoder, or new input feature is used.\n\n"
        "sub09/sub14 were monitored as post-hoc repeated-error subjects, not used for optimization.\n\n"
        f"{table}\n",
        encoding="utf-8",
    )


def _safe_filename_token(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value)
    safe = safe.strip("_-")
    if not safe:
        raise ValueError("filename token must contain at least one safe character.")
    return safe


if __name__ == "__main__":
    main()
