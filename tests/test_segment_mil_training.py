from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest
import torch

from eeg_recovery.training.loso import LOSOFold
from eeg_recovery.training.segment_ssl_dataset import SegmentSSLRecord
from eeg_recovery.training.train_segment_mil import (
    FOLDSTRICT_WPLI_CHECKPOINT_TAG,
    SegmentMILTrainingConfig,
    WPLISegmentBagRecord,
    fit_segment_scaler_on_train_bags,
    load_wpli_segment_barlow_encoder_for_fold,
    resolve_existing_wpli_segment_barlow_encoder_for_fold,
    run_loso_wpli_segment_mil_with_history,
    transform_bag,
)
from eeg_recovery.training.ssl_checkpointing import make_ssl_checkpoint_name, save_ssl_encoder_checkpoint


def _bag(subject_id: str, label: int, value: float) -> WPLISegmentBagRecord:
    eo = np.full((2, 1891, 6), value, dtype=np.float32)
    ec = np.full((2, 1891, 6), value + 2.0, dtype=np.float32)
    return WPLISegmentBagRecord(
        subject_id=subject_id,
        label=label,
        eo_segments=eo,
        ec_segments=ec,
        segment_metadata={
            "EO": [{"segment_index": 0, "start_sample": 0, "end_sample": 2000}, {"segment_index": 1, "start_sample": 1000, "end_sample": 3000}],
            "EC": [{"segment_index": 0, "start_sample": 0, "end_sample": 2000}, {"segment_index": 1, "start_sample": 1000, "end_sample": 3000}],
        },
    )


def test_fold_local_segment_scaler_uses_only_fit_subjects() -> None:
    fit_bag = _bag("sub01", 0, 1.0)
    val_bag = _bag("sub02", 1, 50.0)
    test_bag = _bag("sub03", 1, 100.0)

    scaler = fit_segment_scaler_on_train_bags([fit_bag, val_bag, test_bag], fit_subject_ids=["sub01"])

    np.testing.assert_allclose(scaler.mean, np.full((1891, 6), 2.0, dtype=np.float32))
    assert scaler.fitted_subject_ids == ("sub01",)
    transformed = transform_bag(fit_bag, scaler)
    np.testing.assert_allclose(transformed.eo_segments, np.full((2, 1891, 6), -1.0, dtype=np.float32))
    assert not np.allclose(scaler.mean, np.full((1891, 6), 51.5, dtype=np.float32))


def test_run_loso_segment_mil_emits_one_prediction_per_patient() -> None:
    records = [_bag("sub01", 0, 0.0), _bag("sub02", 1, 1.0), _bag("sub03", 1, 2.0)]
    config = SegmentMILTrainingConfig(
        device="cpu",
        seed=3,
        embedding_dim=2,
        dropout=0.0,
        max_segments_per_state=2,
        stage1_epochs=1,
        stage2_epochs=0,
        freeze_bn=True,
    )

    predictions, metrics, history, attention = run_loso_wpli_segment_mil_with_history(records, config)

    assert predictions.shape[0] == 3
    assert predictions["subject_id"].is_unique
    assert predictions.groupby("subject_id").size().eq(1).all()
    assert set(["accuracy", "balanced_accuracy", "roc_auc", "pr_auc"]).issubset(metrics.columns)
    assert set(["fold_index", "epoch", "stage", "train_loss", "val_loss"]).issubset(history.columns)
    assert set(["subject_id", "state", "segment_index", "attention_weight", "y_score"]).issubset(attention.columns)


def test_wpli_segment_barlow_checkpoint_metadata_mismatch_raises(tmp_path: Path) -> None:
    fold = LOSOFold(fold_index=4, test_subject_id="sub09", train_subject_ids=("sub01", "sub05"))
    checkpoint_dir = tmp_path / "checkpoints"
    path = checkpoint_dir / make_ssl_checkpoint_name(
        method="segssl",
        branch="wpli",
        ssl_objective="barlow",
        seed=0,
        fold_index=fold.fold_index,
        test_subject_id=fold.test_subject_id,
        embedding_dim=32,
        pretrain_epochs=20,
        ssl_data_scope="all-patient",
    )
    metadata = {
        "checkpoint_type": "segment_ssl_encoder",
        "branch": "psd",
        "ssl_objective": "barlow",
        "segment_ssl_method": "segment_barlow",
        "base_seed": 0,
        "fold_index": fold.fold_index,
        "test_subject_id": fold.test_subject_id,
        "excluded_subject_id": fold.test_subject_id,
        "ssl_data_scope": "all-patient",
        "segment_feature_kind": "fc-wpli",
        "supervised_feature_kind": "psd-fc-wpli",
        "feature_kind": "fc-wpli",
        "encoder_kind": "cnn",
        "embedding_dim": 32,
        "dropout": 0.0,
        "projection_dim": 32,
        "pretrain_epochs": 20,
        "pretrain_lr": 1e-3,
        "feature_mask_prob": 0.03,
        "noise_std": 0.02,
        "source_feature_manifest_hash": "manifest-a",
    }
    save_ssl_encoder_checkpoint(path, metadata, {"encoder.network.0.weight": torch.ones(1)})

    with pytest.raises(ValueError, match="branch"):
        load_wpli_segment_barlow_encoder_for_fold(
            checkpoint_dir=checkpoint_dir,
            fold=fold,
            seed=0,
            embedding_dim=32,
            pretrain_epochs=20,
            ssl_data_scope="all-patient",
            source_feature_manifest_hash="manifest-a",
            dropout=0.0,
            pretrain_lr=1e-3,
        )


def test_wpli_segment_barlow_checkpoint_loader_supports_tagged_pure_wpli_checkpoints(tmp_path: Path) -> None:
    fold = LOSOFold(fold_index=4, test_subject_id="sub09", train_subject_ids=("sub01", "sub05"))
    checkpoint_dir = tmp_path / "checkpoints"
    path = checkpoint_dir / make_ssl_checkpoint_name(
        method="segssl",
        branch="wpli",
        ssl_objective="barlow",
        seed=0,
        fold_index=fold.fold_index,
        test_subject_id=fold.test_subject_id,
        embedding_dim=32,
        pretrain_epochs=20,
        ssl_data_scope="all-patient",
        tag="mil",
    )
    metadata = {
        "checkpoint_type": "segment_ssl_encoder",
        "branch": "wpli",
        "ssl_objective": "barlow",
        "segment_ssl_method": "segment_barlow",
        "base_seed": 0,
        "fold_index": fold.fold_index,
        "test_subject_id": fold.test_subject_id,
        "excluded_subject_id": fold.test_subject_id,
        "ssl_data_scope": "all-patient",
        "segment_feature_kind": "fc-wpli",
        "supervised_feature_kind": "psd-fc-wpli",
        "feature_kind": "fc-wpli",
        "encoder_kind": "cnn",
        "embedding_dim": 32,
        "dropout": 0.0,
        "projection_dim": 32,
        "pretrain_epochs": 20,
        "pretrain_lr": 1e-3,
        "feature_mask_prob": 0.03,
        "noise_std": 0.02,
        "source_feature_manifest_hash": "manifest-a",
    }
    save_ssl_encoder_checkpoint(path, metadata, {"encoder.network.0.weight": torch.ones(1)})

    state = load_wpli_segment_barlow_encoder_for_fold(
        checkpoint_dir=checkpoint_dir,
        fold=fold,
        seed=0,
        embedding_dim=32,
        pretrain_epochs=20,
        ssl_data_scope="all-patient",
        source_feature_manifest_hash="manifest-a",
        dropout=0.0,
        pretrain_lr=1e-3,
        checkpoint_tag="mil",
    )

    assert set(state) == {"encoder.network.0.weight"}


def test_wpli_segment_barlow_resolver_extracts_dual_checkpoint_when_branch_specific_missing(tmp_path: Path) -> None:
    fold = LOSOFold(fold_index=4, test_subject_id="sub09", train_subject_ids=("sub01", "sub05"))
    checkpoint_dir = Path("results") / "checkpoints" / "ssl_encoders" / f"test_dual_extract_{uuid.uuid4().hex[:8]}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dual_path = checkpoint_dir / "dual_segbarlow_seed0_fold04_test_sub09_emb32_pre20_all-patient.pt"
    torch.save(
        {
            "checkpoint_type": "dual_segment_ssl_encoder",
            "metadata": {
                "checkpoint_type": "dual_segment_ssl_encoder",
                "branch": "dual",
                "ssl_objective": "barlow",
                "segment_ssl_method": "dual_segment_barlow",
                "base_seed": 0,
                "fold_index": fold.fold_index,
                "test_subject_id": fold.test_subject_id,
                "excluded_subject_id": fold.test_subject_id,
                "ssl_data_scope": "all-patient",
                "feature_kind": "psd-fc-wpli",
                "encoder_kind": "cnn",
                "embedding_dim": 32,
                "dropout": 0.0,
                "pretrain_epochs": 20,
                "pretrain_lr": 1e-3,
                "source_feature_manifest_hash": "dual-manifest",
            },
            "state_dict": {
                "branch_models.wpli.encoder.encoder.network.0.weight": torch.full((1,), 5.0),
            },
        },
        dual_path,
    )

    resolution = resolve_existing_wpli_segment_barlow_encoder_for_fold(
        checkpoint_dir=checkpoint_dir,
        fold=fold,
        seed=0,
        embedding_dim=32,
        pretrain_epochs=20,
        ssl_data_scope="all-patient",
        source_feature_manifest_hash="wpli-manifest",
        dual_source_feature_manifest_hash="dual-manifest",
        dropout=0.0,
        pretrain_lr=1e-3,
    )

    assert resolution.source == "dual_extracted"
    assert resolution.checkpoint_path is not None
    assert FOLDSTRICT_WPLI_CHECKPOINT_TAG in resolution.checkpoint_path.name
    assert set(resolution.encoder_state_dict) == {"encoder.network.0.weight"}
    torch.testing.assert_close(resolution.encoder_state_dict["encoder.network.0.weight"], torch.full((1,), 5.0))
    saved = torch.load(resolution.checkpoint_path, map_location="cpu")
    assert saved["metadata"]["branch"] == "wpli"
    assert saved["metadata"]["source_feature_manifest_hash"] == "wpli-manifest"


def test_segment_mil_can_load_baseline_wpli_records_from_segment_cache(tmp_path: Path) -> None:
    from eeg_recovery.training.train_segment_mil import load_wpli_segment_bag_records

    segment_dir = tmp_path / "data" / "features" / "segment_level" / "fc"
    segment_dir.mkdir(parents=True)
    for state in ("EO", "EC"):
        for index in range(2):
            np.savez_compressed(
                segment_dir / f"patient_sub01_baseline_{state}_seg{index:04d}_fc.npz",
                wpli=np.full((1891, 6), float(index + (state == "EC")), dtype=np.float32),
                group=np.array("patient"),
                subject_id=np.array("sub01"),
                subject_key=np.array("patient:sub01"),
                stage=np.array("baseline"),
                state=np.array(state),
                segment_index=np.array(index),
                start_sample=np.array(index * 1000),
                end_sample=np.array(index * 1000 + 2000),
                source_set_path=np.array("source.set"),
            )
    records = [
        SegmentSSLRecord(
            group="patient",
            subject_id="sub01",
            subject_key="patient:sub01",
            stage="baseline",
            state=state,
            segment_index=index,
            features={"wpli": np.full((1891, 6), 1.0, dtype=np.float32)},
            source_path=Path("unused"),
        )
        for state in ("EO", "EC")
        for index in range(2)
    ]
    labels = pd.DataFrame({"subject_id": ["sub01"], "label": [1]})

    bags = load_wpli_segment_bag_records(records, labels)

    assert len(bags) == 1
    assert bags[0].subject_id == "sub01"
    assert bags[0].eo_segments.shape == (2, 1891, 6)
    assert bags[0].ec_segments.shape == (2, 1891, 6)
