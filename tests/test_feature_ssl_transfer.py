from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from eeg_recovery.io.index import EEGFileRecord
from eeg_recovery.models.multimodal_model import MultimodalEEGModel
from eeg_recovery.training.train_feature_ssl import (
    FeatureSSLPairRecord,
    FeatureSSLTrainingConfig,
    aggregate_seed_ensemble_predictions,
    build_feature_ssl_pair_records_from_feature_records,
    feature_ssl_transfer_run_name,
    feature_ssl_pairs_for_scope,
    run_feature_ssl_pretraining,
    widest_feature_ssl_scope,
)
from eeg_recovery.training.train_supervised import (
    SupervisedFeatureRecord,
    SupervisedTrainingConfig,
    _apply_pretrained_transfer_mode,
    run_loso_supervised_with_history,
)


def _feature_record(subject_id: str, label: int, value: float) -> SupervisedFeatureRecord:
    psd = np.full((62, 90), value, dtype=np.float32)
    wpli = np.full((1891, 6), value + 1.0, dtype=np.float32)
    return SupervisedFeatureRecord(
        subject_id=subject_id,
        label=label,
        eo=psd,
        ec=psd + 0.5,
        modalities={
            "psd": (psd, psd + 0.5),
            "wpli": (wpli, wpli + 0.5),
        },
    )


def _eeg_record(
    *,
    group: str,
    subject_id: str,
    stage: str,
    state: str,
    supervised: bool,
) -> EEGFileRecord:
    path = Path(f"{group}_{subject_id}_{stage}_{state}.set")
    return EEGFileRecord(
        group=group,
        subject_id=subject_id,
        subject_key=f"{group}:{subject_id}",
        stage=stage,
        state=state,
        set_path=path,
        fdt_path=path.with_suffix(".fdt"),
        is_supervised_subject=supervised,
    )


def test_feature_ssl_pairs_preserve_requested_scope_and_loso_exclusion():
    pairs = [
        FeatureSSLPairRecord(
            group="patient",
            subject_id="sub01",
            subject_key="patient:sub01",
            stage="基线",
            is_supervised_subject=True,
            modalities={},
        ),
        FeatureSSLPairRecord(
            group="patient",
            subject_id="sub02",
            subject_key="patient:sub02",
            stage="基线",
            is_supervised_subject=True,
            modalities={},
        ),
        FeatureSSLPairRecord(
            group="patient",
            subject_id="sub03",
            subject_key="patient:sub03",
            stage="最终",
            is_supervised_subject=False,
            modalities={},
        ),
        FeatureSSLPairRecord(
            group="health",
            subject_id="health01",
            subject_key="health:health01",
            stage="health",
            is_supervised_subject=False,
            modalities={},
        ),
    ]

    supervised = feature_ssl_pairs_for_scope(
        pairs,
        data_scope="supervised-baseline",
        strict_loso_test_subject_id="sub01",
    )
    all_patient = feature_ssl_pairs_for_scope(
        pairs,
        data_scope="all-patient",
        strict_loso_test_subject_id="sub01",
    )
    all_patient_health = feature_ssl_pairs_for_scope(
        pairs,
        data_scope="all-patient-health",
        strict_loso_test_subject_id="sub01",
    )

    assert [pair.subject_key for pair in supervised] == ["patient:sub02"]
    assert {pair.subject_key for pair in all_patient} == {"patient:sub02", "patient:sub03"}
    assert {pair.subject_key for pair in all_patient_health} == {
        "patient:sub02",
        "patient:sub03",
        "health:health01",
    }


def test_widest_feature_ssl_scope_limits_unnecessary_feature_computation():
    assert widest_feature_ssl_scope(["supervised-baseline"]) == "supervised-baseline"
    assert widest_feature_ssl_scope(["supervised-baseline", "all-patient-baseline"]) == "all-patient-baseline"
    assert widest_feature_ssl_scope(["all-patient-baseline", "all-patient"]) == "all-patient"
    assert widest_feature_ssl_scope(["supervised-baseline", "all-patient-health"]) == "all-patient-health"
    assert widest_feature_ssl_scope(["supervised-baseline", "health-only"]) == "all-patient-health"


def test_build_feature_ssl_pair_records_requires_complete_eo_ec_pairs():
    feature_records = {
        ("patient:sub01", "基线", "EO"): {
            "psd": np.zeros((62, 90), dtype=np.float32),
            "wpli": np.zeros((1891, 6), dtype=np.float32),
        }
    }
    eeg_records = [
        _eeg_record(group="patient", subject_id="sub01", stage="基线", state="EO", supervised=True),
    ]

    with pytest.raises(ValueError, match="complete EO/EC"):
        build_feature_ssl_pair_records_from_feature_records(eeg_records, feature_records)


def test_feature_ssl_pretraining_returns_branch_state_for_supervised_cnn():
    pairs = [
        FeatureSSLPairRecord(
            group="patient",
            subject_id=f"sub0{index + 1}",
            subject_key=f"patient:sub0{index + 1}",
            stage="基线",
            is_supervised_subject=True,
            modalities={
                "psd": (
                    np.random.default_rng(index).normal(size=(62, 90)).astype(np.float32),
                    np.random.default_rng(index + 10).normal(size=(62, 90)).astype(np.float32),
                ),
                "wpli": (
                    np.random.default_rng(index + 20).normal(size=(1891, 6)).astype(np.float32),
                    np.random.default_rng(index + 30).normal(size=(1891, 6)).astype(np.float32),
                ),
            },
        )
        for index in range(3)
    ]
    config = FeatureSSLTrainingConfig(
        data_scope="supervised-baseline",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="cnn",
        epochs=1,
        batch_size=2,
        embedding_dim=4,
        projection_dim=4,
        dropout=0.0,
        device="cpu",
        seed=5,
    )

    pretrained_state, history = run_feature_ssl_pretraining(pairs, config)
    model = MultimodalEEGModel("psd-fc-wpli", fusion="gated", embedding_dim=4, dropout=0.0, encoder_kind="cnn")
    incompatible = model.load_state_dict(pretrained_state, strict=False)

    assert history["epoch"].tolist() == [1]
    assert history["n_pairs"].tolist() == [3]
    assert np.isfinite(history["loss"].to_numpy()).all()
    assert all(key.startswith("branch_models.") for key in pretrained_state)
    assert not incompatible.unexpected_keys


def test_feature_ssl_pretraining_can_use_vicreg_without_negative_samples():
    rng = np.random.default_rng(17)
    pairs = [
        FeatureSSLPairRecord(
            group="patient",
            subject_id=f"sub0{index + 1}",
            subject_key=f"patient:sub0{index + 1}",
            stage="基线",
            is_supervised_subject=True,
            modalities={
                "psd": (
                    rng.normal(size=(62, 90)).astype(np.float32),
                    rng.normal(size=(62, 90)).astype(np.float32),
                ),
                "wpli": (
                    rng.normal(size=(1891, 6)).astype(np.float32),
                    rng.normal(size=(1891, 6)).astype(np.float32),
                ),
            },
        )
        for index in range(4)
    ]
    config = FeatureSSLTrainingConfig(
        data_scope="supervised-baseline",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="cnn",
        epochs=1,
        batch_size=4,
        embedding_dim=4,
        projection_dim=4,
        dropout=0.0,
        ssl_objective="vicreg",
        device="cpu",
        seed=17,
    )

    pretrained_state, history = run_feature_ssl_pretraining(pairs, config)
    model = MultimodalEEGModel("psd-fc-wpli", fusion="gated", embedding_dim=4, dropout=0.0, encoder_kind="cnn")
    incompatible = model.load_state_dict(pretrained_state, strict=False)

    assert history["objective"].tolist() == ["feature_vicreg"]
    assert history["ssl_objective"].tolist() == ["vicreg"]
    assert np.isfinite(history["alignment_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_invariance_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_variance_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_covariance_loss"].to_numpy()).all()
    assert not incompatible.unexpected_keys


def test_loso_supervised_loads_pretrained_state_per_test_subject():
    records = [_feature_record("sub01", 0, 0.0), _feature_record("sub02", 1, 1.0)]
    template = MultimodalEEGModel("psd-fc-wpli", fusion="gated", embedding_dim=2, dropout=0.0, encoder_kind="linear")
    pretrained_state = {
        key: value.detach().clone()
        for key, value in template.state_dict().items()
        if key.startswith("branch_models.")
    }
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        device="cpu",
        epochs=1,
        patience=2,
        embedding_dim=2,
        dropout=0.0,
        seed=7,
    )

    predictions, metrics, loss_history = run_loso_supervised_with_history(
        records,
        config,
        pretrained_state_by_test_subject={"sub01": pretrained_state, "sub02": pretrained_state},
    )

    assert predictions["status"].tolist() == ["trained", "trained"]
    assert metrics["status"].tolist() == ["trained"]
    assert set(loss_history["pretrained"].unique()) == {True}


def test_pretrained_transfer_mode_freezes_only_branch_encoders():
    model = MultimodalEEGModel("psd-fc-wpli", fusion="gated", embedding_dim=2, dropout=0.0, encoder_kind="linear")

    _apply_pretrained_transfer_mode(model, "freeze-encoder")

    frozen_names = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert frozen_names
    assert all(".encoder." in name for name in frozen_names)
    assert any(".gate." in name for name in trainable_names)
    assert any(name.startswith("classifier.") for name in trainable_names)


def test_seed_ensemble_predictions_average_patient_probabilities_and_metrics():
    first = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02"],
            "fold_index": [0, 1],
            "y_true": [1, 0],
            "y_score": [0.6, 0.4],
        }
    )
    second = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02"],
            "fold_index": [0, 1],
            "y_true": [1, 0],
            "y_score": [0.8, 0.7],
        }
    )

    predictions, metrics = aggregate_seed_ensemble_predictions([first, second], threshold=0.5)

    assert predictions["subject_id"].tolist() == ["sub01", "sub02"]
    np.testing.assert_allclose(predictions["y_score"].to_numpy(), np.array([0.7, 0.55]))
    assert predictions["y_pred"].tolist() == [1, 1]
    assert metrics.loc[0, "accuracy"] == 0.5
    assert metrics.loc[0, "ensemble_size"] == 2


def test_feature_ssl_transfer_run_name_includes_pretrain_batch_size():
    run_name = feature_ssl_transfer_run_name(
        data_scope="all-patient",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="cnn",
        transfer_mode="finetune",
        seed="2",
        ssl_objective="vicreg",
        pretrain_epochs=50,
        temperature=0.2,
        noise_std=0.02,
        feature_mask_prob=0.05,
        pretrain_batch_size=8,
        supervised_epochs=100,
    )

    assert "bs8" in run_name
    assert "vicreg" in run_name
    assert "temp0_2" in run_name


def test_feature_ssl_transfer_run_name_records_nondefault_tuning_parameters():
    run_name = feature_ssl_transfer_run_name(
        data_scope="all-patient",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="cnn",
        transfer_mode="finetune",
        seed="3",
        ssl_objective="barlow",
        pretrain_epochs=50,
        temperature=0.2,
        noise_std=0.02,
        feature_mask_prob=0.01,
        pretrain_batch_size=8,
        supervised_epochs=100,
        projection_dim=64,
        dropout=0.1,
        supervised_lr=5e-4,
    )

    assert "proj64" in run_name
    assert "drop0_1" in run_name
    assert "slr0_0005" in run_name


def test_feature_ssl_transfer_script_help_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-B", "scripts/07_train_feature_ssl_transfer.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--data-scopes" in result.stdout
    assert "all-patient-health" in result.stdout
    assert "--pretrain-epochs" in result.stdout
    assert "--seeds" in result.stdout
    assert "--transfer-modes" in result.stdout
    assert "--temperature" in result.stdout
    assert "--ssl-objective" in result.stdout
