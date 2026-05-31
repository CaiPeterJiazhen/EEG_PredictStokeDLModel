from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import importlib.util

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
    _apply_latent_modality_mask,
    _safe_run_name,
)
from eeg_recovery.training.train_supervised import (
    SupervisedFeatureRecord,
    SupervisedTrainingConfig,
    _apply_pretrained_transfer_mode,
    _build_model,
    _build_transfer_optimizer,
    _compute_supervised_loss,
    _probability_to_logit,
    _supervised_binary_loss,
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


def _feature_record_with_eeg_summary(subject_id: str, label: int, value: float) -> SupervisedFeatureRecord:
    record = _feature_record(subject_id, label, value)
    return SupervisedFeatureRecord(
        subject_id=record.subject_id,
        label=record.label,
        eo=record.eo,
        ec=record.ec,
        modalities=record.modalities,
        eeg_summary=np.asarray([value, value + 0.5, value + 1.0], dtype=np.float32),
        eeg_summary_feature_names=("psd_beta_mean", "wpli_beta_mean", "beta_bsi"),
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


def test_eeg_summary_features_are_fold_scaled_and_exposed_in_loso_predictions():
    records = [
        _feature_record_with_eeg_summary(f"sub0{index + 1}", index % 2, float(index + 1))
        for index in range(4)
    ]
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        device="cpu",
        epochs=1,
        patience=1,
        lr=1e-3,
        embedding_dim=4,
        dropout=0.0,
        seed=3,
        eeg_summary_features_enabled=True,
        eeg_summary_feature_names=("psd_beta_mean", "wpli_beta_mean", "beta_bsi"),
        eeg_summary_embedding_dim=4,
    )

    predictions, metrics, _ = run_loso_supervised_with_history(records, config)

    assert metrics["eeg_summary_features_enabled"].tolist() == [True]
    assert set(predictions.columns) >= {
        "modality_weight_psd",
        "modality_weight_wpli",
        "modality_weight_eeg_summary",
        "eeg_summary_importance_psd_beta_mean",
        "eeg_summary_importance_wpli_beta_mean",
        "eeg_summary_importance_beta_bsi",
    }
    assert np.isfinite(predictions["modality_weight_eeg_summary"]).all()


def test_feature_ssl_pretraining_is_reproducible_for_same_seed():
    rng = np.random.default_rng(29)
    pairs = [
        FeatureSSLPairRecord(
            group="patient",
            subject_id=f"sub0{index + 1}",
            subject_key=f"patient:sub0{index + 1}",
            stage="鍩虹嚎",
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
        ssl_objective="barlow",
        device="cpu",
        seed=29,
    )

    first_state, first_history = run_feature_ssl_pretraining(pairs, config)
    second_state, second_history = run_feature_ssl_pretraining(pairs, config)

    assert first_history["loss"].tolist() == second_history["loss"].tolist()
    assert first_state.keys() == second_state.keys()
    for key in first_state:
        torch.testing.assert_close(first_state[key], second_state[key])


def test_feature_ssl_pretraining_can_use_branch_aware_barlow_objective():
    rng = np.random.default_rng(31)
    pairs = [
        FeatureSSLPairRecord(
            group="patient",
            subject_id=f"sub0{index + 1}",
            subject_key=f"patient:sub0{index + 1}",
            stage="baseline",
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
        encoder_kind="linear",
        epochs=1,
        batch_size=4,
        embedding_dim=4,
        projection_dim=4,
        dropout=0.0,
        ssl_objective="branch-barlow",
        branch_barlow_weight=0.5,
        device="cpu",
        seed=31,
    )

    pretrained_state, history = run_feature_ssl_pretraining(pairs, config)

    assert history["objective"].tolist() == ["feature_branch_barlow"]
    assert history["ssl_objective"].tolist() == ["branch-barlow"]
    assert np.isfinite(history["loss"].to_numpy()).all()
    assert np.isfinite(history["branch_barlow_loss"].to_numpy()).all()
    assert history["branch_barlow_weight"].tolist() == [0.5]
    assert all(key.startswith("branch_models.") for key in pretrained_state)


def test_latent_modality_mask_keeps_at_least_one_branch_per_sample():
    branch_embeddings = {
        "psd": torch.ones((8, 3)),
        "wpli": torch.full((8, 3), 2.0),
    }
    generator = torch.Generator().manual_seed(123)

    masked = _apply_latent_modality_mask(
        branch_embeddings,
        branches=("psd", "wpli"),
        mask_prob=1.0,
        generator=generator,
    )

    assert masked.shape == (8, 6)
    assert torch.all(masked.abs().sum(dim=1) > 0)
    assert torch.all((masked[:, :3].abs().sum(dim=1) > 0) | (masked[:, 3:].abs().sum(dim=1) > 0))


def test_feature_ssl_pretraining_can_use_masked_barlow_without_negative_samples():
    rng = np.random.default_rng(41)
    pairs = [
        FeatureSSLPairRecord(
            group="patient",
            subject_id=f"sub0{index + 1}",
            subject_key=f"patient:sub0{index + 1}",
            stage="baseline",
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
        encoder_kind="linear",
        epochs=1,
        batch_size=4,
        embedding_dim=4,
        projection_dim=4,
        dropout=0.0,
        ssl_objective="masked-barlow",
        latent_modality_mask_prob=0.75,
        device="cpu",
        seed=41,
    )

    pretrained_state, history = run_feature_ssl_pretraining(pairs, config)

    assert history["objective"].tolist() == ["feature_masked_barlow"]
    assert history["ssl_objective"].tolist() == ["masked-barlow"]
    assert history["latent_modality_mask_prob"].tolist() == [0.75]
    assert np.isfinite(history["loss"].to_numpy()).all()
    assert np.isfinite(history["global_barlow_loss"].to_numpy()).all()
    assert all(key.startswith("branch_models.") for key in pretrained_state)


def test_feature_ssl_pretraining_can_use_masked_vicreg_without_negative_samples():
    rng = np.random.default_rng(43)
    pairs = [
        FeatureSSLPairRecord(
            group="patient",
            subject_id=f"sub0{index + 1}",
            subject_key=f"patient:sub0{index + 1}",
            stage="baseline",
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
        encoder_kind="linear",
        epochs=1,
        batch_size=4,
        embedding_dim=4,
        projection_dim=4,
        dropout=0.0,
        ssl_objective="masked-vicreg",
        latent_modality_mask_prob=0.5,
        device="cpu",
        seed=43,
    )

    pretrained_state, history = run_feature_ssl_pretraining(pairs, config)

    assert history["objective"].tolist() == ["feature_masked_vicreg"]
    assert history["ssl_objective"].tolist() == ["masked-vicreg"]
    assert np.isfinite(history["loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_invariance_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_variance_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_covariance_loss"].to_numpy()).all()
    assert all(key.startswith("branch_models.") for key in pretrained_state)


def test_feature_ssl_pretraining_can_use_masked_reconstruction_without_negative_samples():
    rng = np.random.default_rng(47)
    pairs = [
        FeatureSSLPairRecord(
            group="patient",
            subject_id=f"sub0{index + 1}",
            subject_key=f"patient:sub0{index + 1}",
            stage="baseline",
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
        encoder_kind="linear",
        epochs=1,
        batch_size=4,
        embedding_dim=4,
        projection_dim=4,
        dropout=0.0,
        ssl_objective="masked-reconstruction",
        feature_mask_prob=0.25,
        device="cpu",
        seed=47,
    )

    pretrained_state, history = run_feature_ssl_pretraining(pairs, config)

    assert history["objective"].tolist() == ["feature_masked_reconstruction"]
    assert history["ssl_objective"].tolist() == ["masked-reconstruction"]
    assert np.isfinite(history["loss"].to_numpy()).all()
    assert np.isfinite(history["reconstruction_loss"].to_numpy()).all()
    assert all(key.startswith("branch_models.") for key in pretrained_state)


def test_feature_ssl_pretraining_can_use_masked_reconstruction_vicreg_without_negative_samples():
    rng = np.random.default_rng(49)
    pairs = [
        FeatureSSLPairRecord(
            group="patient",
            subject_id=f"sub0{index + 1}",
            subject_key=f"patient:sub0{index + 1}",
            stage="baseline",
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
        encoder_kind="linear",
        epochs=1,
        batch_size=4,
        embedding_dim=4,
        projection_dim=4,
        dropout=0.0,
        ssl_objective="masked-reconstruction-vicreg",
        feature_mask_prob=0.5,
        device="cpu",
        seed=49,
    )

    pretrained_state, history = run_feature_ssl_pretraining(pairs, config)

    assert history["objective"].tolist() == ["feature_masked_reconstruction_vicreg"]
    assert history["ssl_objective"].tolist() == ["masked-reconstruction-vicreg"]
    assert np.isfinite(history["loss"].to_numpy()).all()
    assert np.isfinite(history["reconstruction_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_invariance_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_variance_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_covariance_loss"].to_numpy()).all()
    assert all(key.startswith("branch_models.") for key in pretrained_state)


def test_feature_ssl_pretraining_can_use_simsiam_without_negative_samples():
    rng = np.random.default_rng(51)
    pairs = [
        FeatureSSLPairRecord(
            group="patient",
            subject_id=f"sub0{index + 1}",
            subject_key=f"patient:sub0{index + 1}",
            stage="baseline",
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
        encoder_kind="linear",
        epochs=1,
        batch_size=4,
        embedding_dim=4,
        projection_dim=4,
        dropout=0.0,
        ssl_objective="simsiam",
        byol_predictor_hidden_dim=8,
        device="cpu",
        seed=51,
    )

    pretrained_state, history = run_feature_ssl_pretraining(pairs, config)

    assert history["objective"].tolist() == ["feature_simsiam"]
    assert history["ssl_objective"].tolist() == ["simsiam"]
    assert np.isfinite(history["loss"].to_numpy()).all()
    assert np.isfinite(history["byol_loss"].to_numpy()).all()
    assert all(key.startswith("branch_models.") for key in pretrained_state)


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


def test_transfer_optimizer_uses_lower_lr_for_pretrained_encoders():
    model = MultimodalEEGModel("psd-fc-wpli", fusion="gated", embedding_dim=2, dropout=0.0, encoder_kind="linear")
    config = SupervisedTrainingConfig(
        lr=0.01,
        weight_decay=1e-4,
        encoder_lr_multiplier=0.1,
        transfer_head_lr_multiplier=0.5,
    )

    optimizer = _build_transfer_optimizer(model, config)

    group_lrs = sorted({round(float(group["lr"]), 8) for group in optimizer.param_groups})
    assert group_lrs == [0.001, 0.005, 0.01]
    assert all(group["params"] for group in optimizer.param_groups)
    assert {float(group["weight_decay"]) for group in optimizer.param_groups} == {1e-4}


def test_pretrained_finetune_can_warmup_with_encoder_frozen_then_unfreeze():
    records = [
        _feature_record("sub01", 0, 0.0),
        _feature_record("sub02", 1, 1.0),
        _feature_record("sub03", 0, 0.2),
    ]
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
        epochs=2,
        patience=3,
        embedding_dim=2,
        dropout=0.0,
        seed=7,
        pretrained_transfer_mode="finetune",
        freeze_pretrained_encoder_epochs=1,
    )

    _, _, loss_history = run_loso_supervised_with_history(
        records,
        config,
        pretrained_state_by_test_subject={record.subject_id: pretrained_state for record in records},
    )

    trainable_by_epoch = (
        loss_history.groupby("epoch")["encoder_trainable"]
        .apply(lambda values: sorted(set(bool(value) for value in values)))
        .to_dict()
    )
    assert trainable_by_epoch[1] == [False]
    assert trainable_by_epoch[2] == [True]


def test_supervised_config_builds_ssl_bridge_model_when_enabled():
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        ssl_bridge_enabled=True,
    )

    model = _build_model(config)

    assert model.__class__.__name__ == "SSLBridgeMultimodalEEGModel"
    assert all(not parameter.requires_grad for parameter in model.ssl_branch_models.parameters())


def test_supervised_config_builds_ssl_two_head_bridge_model_when_enabled():
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        ssl_two_head_enabled=True,
    )

    model = _build_model(config)

    assert model.__class__.__name__ == "SSLTwoHeadMultimodalEEGModel"
    assert all(not parameter.requires_grad for parameter in model.ssl_branch_models.parameters())
    assert hasattr(model, "trainable_classifier")
    assert hasattr(model, "ssl_classifier")


def test_supervised_config_builds_ssl_residual_bridge_model_when_enabled():
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        ssl_residual_enabled=True,
        ssl_residual_weight=0.14,
        bridge_residual_weight=0.04,
    )

    model = _build_model(config)

    assert model.__class__.__name__ == "SSLResidualMultimodalEEGModel"
    assert all(not parameter.requires_grad for parameter in model.ssl_branch_models.parameters())
    assert model.ssl_residual_weight == pytest.approx(0.14)
    assert model.bridge_residual_weight == pytest.approx(0.04)


def test_supervised_config_can_build_ssl_residual_with_random_supervised_main_path():
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        ssl_residual_enabled=True,
        ssl_residual_random_main=True,
    )

    model = _build_model(config)

    assert model.__class__.__name__ == "SSLResidualMultimodalEEGModel"
    assert model.preload_trainable_branch is False


def test_supervised_config_can_build_preserved_main_ssl_residual():
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        ssl_residual_enabled=True,
        ssl_residual_preserve_main=True,
        ssl_residual_weight=0.14,
        bridge_residual_weight=0.04,
    )

    model = _build_model(config)

    assert model.__class__.__name__ == "FrozenMainSSLResidualMultimodalEEGModel"
    assert all(not parameter.requires_grad for parameter in model.main_model.parameters())
    assert all(not parameter.requires_grad for parameter in model.ssl_branch_models.parameters())


def test_supervised_config_can_build_preserved_main_ssl_logit_delta():
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        ssl_residual_enabled=True,
        ssl_residual_preserve_main=True,
        ssl_residual_logit_delta_enabled=True,
        ssl_residual_logit_delta_scale=0.5,
    )

    model = _build_model(config)

    assert model.__class__.__name__ == "FrozenMainSSLLogitDeltaMultimodalEEGModel"
    assert model.logit_delta_scale == pytest.approx(0.5)
    assert all(not parameter.requires_grad for parameter in model.main_model.parameters())
    assert all(not parameter.requires_grad for parameter in model.ssl_branch_models.parameters())
    assert any(parameter.requires_grad for parameter in model.delta_head.parameters())


def test_ssl_residual_main_loss_only_trains_on_supervised_main_head_but_validates_mixture():
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        dropout=0.0,
        ssl_residual_enabled=True,
        ssl_residual_random_main=True,
        ssl_residual_main_loss_only=True,
        ssl_residual_weight=0.25,
        bridge_residual_weight=0.25,
        ssl_aux_head_weight=0.0,
    )
    model = _build_model(config)
    batch = {
        "psd_eo": torch.randn(3, 62, 90),
        "psd_ec": torch.randn(3, 62, 90),
        "wpli_eo": torch.randn(3, 1891, 6),
        "wpli_ec": torch.randn(3, 1891, 6),
        "y": torch.tensor([[0.0], [1.0], [1.0]]),
    }

    mixture_probabilities, aux = model(batch, return_aux=True)
    training_loss = _compute_supervised_loss(model, batch, config, training=True)
    validation_loss = _compute_supervised_loss(model, batch, config, training=False)

    expected_training_loss = _supervised_binary_loss(aux["trainable_probabilities"], batch["y"], config)
    expected_validation_loss = _supervised_binary_loss(mixture_probabilities, batch["y"], config)
    torch.testing.assert_close(training_loss, expected_training_loss)
    torch.testing.assert_close(validation_loss, expected_validation_loss)
    assert not torch.isclose(training_loss, expected_validation_loss)


def test_ssl_residual_logit_preservation_adds_main_path_distillation_penalty():
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        dropout=0.0,
        ssl_residual_enabled=True,
        ssl_residual_weight=0.25,
        bridge_residual_weight=0.15,
        ssl_aux_head_weight=0.0,
        ssl_residual_logit_preservation_weight=0.7,
    )
    model = _build_model(config)
    batch = {
        "psd_eo": torch.randn(3, 62, 90),
        "psd_ec": torch.randn(3, 62, 90),
        "wpli_eo": torch.randn(3, 1891, 6),
        "wpli_ec": torch.randn(3, 1891, 6),
        "y": torch.tensor([[0.0], [1.0], [1.0]]),
    }

    probabilities, aux = model(batch, return_aux=True)
    expected = _supervised_binary_loss(probabilities, batch["y"], config)
    expected = expected + 0.7 * torch.nn.functional.mse_loss(
        _probability_to_logit(probabilities),
        _probability_to_logit(aux["trainable_probabilities"]).detach(),
    )

    loss = _compute_supervised_loss(model, batch, config, training=True)
    validation_loss = _compute_supervised_loss(model, batch, config, training=False)

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(validation_loss, _supervised_binary_loss(probabilities, batch["y"], config))
    assert loss > validation_loss


def test_ssl_residual_logit_preservation_weight_must_be_non_negative():
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        ssl_residual_enabled=True,
        ssl_residual_logit_preservation_weight=-0.1,
    )

    with pytest.raises(ValueError, match="ssl_residual_logit_preservation_weight"):
        _build_model(config)


def test_loso_preserved_main_ssl_residual_trains_main_phase_before_residual_phase():
    records = [_feature_record("sub01", 0, 0.0), _feature_record("sub02", 1, 1.0), _feature_record("sub03", 1, 2.0)]
    template = MultimodalEEGModel(
        feature_kind="psd-fc-wpli",
        fusion="gated",
        embedding_dim=2,
        dropout=0.0,
        encoder_kind="linear",
    )
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
        ssl_residual_enabled=True,
        ssl_residual_preserve_main=True,
        ssl_residual_weight=0.14,
        bridge_residual_weight=0.04,
        ssl_aux_head_weight=0.0,
        seed=7,
    )

    predictions, metrics, loss_history = run_loso_supervised_with_history(
        records,
        config,
        pretrained_state_by_test_subject={record.subject_id: pretrained_state for record in records},
    )

    assert predictions["ssl_residual_preserve_main"].eq(True).all()
    assert {"trainable_y_score", "ssl_residual_y_score", "bridge_residual_y_score"}.issubset(predictions.columns)
    assert predictions[["trainable_y_score", "ssl_residual_y_score", "bridge_residual_y_score"]].notna().all().all()
    assert metrics["ssl_residual_preserve_main"].eq(True).all()
    assert set(loss_history["training_phase"]) == {"preserved_main", "ssl_residual"}
    assert loss_history.groupby(["fold_index", "training_phase"]).size().min() >= 1


def test_loso_preserved_main_ssl_residual_records_validation_selected_weights():
    records = [_feature_record("sub01", 0, 0.0), _feature_record("sub02", 1, 1.0), _feature_record("sub03", 1, 2.0)]
    template = MultimodalEEGModel(
        feature_kind="psd-fc-wpli",
        fusion="gated",
        embedding_dim=2,
        dropout=0.0,
        encoder_kind="linear",
    )
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
        ssl_residual_enabled=True,
        ssl_residual_preserve_main=True,
        ssl_residual_select_weights_on_val=True,
        ssl_residual_weight_candidates=(0.0, 0.5),
        bridge_residual_weight_candidates=(0.0, 0.25),
        ssl_aux_head_weight=0.0,
        seed=7,
    )

    predictions, metrics, loss_history = run_loso_supervised_with_history(
        records,
        config,
        pretrained_state_by_test_subject={record.subject_id: pretrained_state for record in records},
    )

    assert predictions["ssl_residual_select_weights_on_val"].eq(True).all()
    assert {"selected_ssl_residual_weight", "selected_bridge_residual_weight"}.issubset(predictions.columns)
    assert predictions["selected_ssl_residual_weight"].isin({0.0, 0.5}).all()
    assert predictions["selected_bridge_residual_weight"].isin({0.0, 0.25}).all()
    assert metrics["ssl_residual_select_weights_on_val"].eq(True).all()
    assert {"selected_ssl_residual_weight", "selected_bridge_residual_weight"}.issubset(metrics.columns)
    assert "selected_ssl_residual_weight" in loss_history.columns


def test_supervised_config_builds_multimodal_embedding_adapter_when_requested():
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        embedding_adapter_dim=3,
        embedding_adapter_scale=1.0,
    )

    model = _build_model(config)

    assert model.__class__.__name__ == "MultimodalEEGModel"
    assert model.embedding_adapter is not None


def test_loso_supervised_ssl_bridge_requires_pretrained_state():
    records = [_feature_record("sub01", 0, 0.0), _feature_record("sub02", 1, 1.0)]
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
        ssl_bridge_enabled=True,
    )

    with pytest.raises(ValueError, match="ssl_bridge_enabled requires pretrained"):
        run_loso_supervised_with_history(records, config)


def test_loso_supervised_ssl_two_head_requires_pretrained_state():
    records = [_feature_record("sub01", 0, 0.0), _feature_record("sub02", 1, 1.0)]
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        device="cpu",
        epochs=1,
        patience=1,
        embedding_dim=2,
        dropout=0.0,
        ssl_two_head_enabled=True,
    )

    with pytest.raises(ValueError, match="ssl_two_head_enabled requires pretrained"):
        run_loso_supervised_with_history(records, config)


def test_loso_supervised_ssl_residual_requires_pretrained_state():
    records = [_feature_record("sub01", 0, 0.0), _feature_record("sub02", 1, 1.0)]
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
        ssl_residual_enabled=True,
    )

    with pytest.raises(ValueError, match="ssl_residual_enabled requires pretrained"):
        run_loso_supervised_with_history(records, config)


def test_ssl_bridge_and_two_head_modes_are_mutually_exclusive():
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        ssl_bridge_enabled=True,
        ssl_two_head_enabled=True,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_model(config)


def test_ssl_two_head_forward_returns_fused_and_branch_probabilities():
    model = _build_model(
        SupervisedTrainingConfig(
            architecture="multimodal",
            feature_kind="psd-fc-wpli",
            fusion="gated",
            encoder_kind="linear",
            embedding_dim=2,
            ssl_two_head_enabled=True,
        )
    )
    batch = {
        "psd_eo": torch.randn(3, 62, 90),
        "psd_ec": torch.randn(3, 62, 90),
        "wpli_eo": torch.randn(3, 1891, 6),
        "wpli_ec": torch.randn(3, 1891, 6),
        "y": torch.tensor([[0.0], [1.0], [0.0]]),
    }

    probabilities, aux = model(batch, return_aux=True)

    assert probabilities.shape == (3, 1)
    assert aux["trainable_probabilities"].shape == (3, 1)
    assert aux["ssl_probabilities"].shape == (3, 1)
    assert torch.isfinite(probabilities).all()


def test_ssl_two_head_auxiliary_loss_backpropagates_to_heads_but_not_frozen_ssl_encoder():
    model = _build_model(
        SupervisedTrainingConfig(
            architecture="multimodal",
            feature_kind="psd-fc-wpli",
            fusion="gated",
            encoder_kind="linear",
            embedding_dim=2,
            ssl_two_head_enabled=True,
        )
    )
    batch = {
        "psd_eo": torch.randn(3, 62, 90),
        "psd_ec": torch.randn(3, 62, 90),
        "wpli_eo": torch.randn(3, 1891, 6),
        "wpli_ec": torch.randn(3, 1891, 6),
        "y": torch.tensor([[0.0], [1.0], [0.0]]),
    }
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        ssl_two_head_enabled=True,
        ssl_aux_head_weight=0.5,
    )

    loss = _compute_supervised_loss(model, batch, config, training=True)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.branch_models.parameters())
    assert any(parameter.grad is not None for parameter in model.trainable_classifier.parameters())
    assert any(parameter.grad is not None for parameter in model.ssl_classifier.parameters())
    assert all(parameter.grad is None for parameter in model.ssl_branch_models.parameters())


def test_ssl_consistency_loss_requires_bridge_model_and_backpropagates_to_trainable_encoder_only():
    model = _build_model(
        SupervisedTrainingConfig(
            architecture="multimodal",
            feature_kind="psd-fc-wpli",
            fusion="gated",
            encoder_kind="linear",
            embedding_dim=2,
            ssl_bridge_enabled=True,
            ssl_consistency_weight=0.25,
        )
    )
    batch = {
        "psd_eo": torch.randn(3, 62, 90),
        "psd_ec": torch.randn(3, 62, 90),
        "wpli_eo": torch.randn(3, 1891, 6),
        "wpli_ec": torch.randn(3, 1891, 6),
        "y": torch.tensor([[0.0], [1.0], [0.0]]),
    }
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        ssl_bridge_enabled=True,
        ssl_consistency_weight=0.25,
    )

    loss = _compute_supervised_loss(model, batch, config, training=True)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.branch_models.parameters())
    assert all(parameter.grad is None for parameter in model.ssl_branch_models.parameters())


def test_ssl_consistency_loss_rejects_non_bridge_model():
    model = MultimodalEEGModel("psd-fc-wpli", fusion="gated", embedding_dim=2, dropout=0.0, encoder_kind="linear")
    batch = {
        "psd_eo": torch.randn(3, 62, 90),
        "psd_ec": torch.randn(3, 62, 90),
        "wpli_eo": torch.randn(3, 1891, 6),
        "wpli_ec": torch.randn(3, 1891, 6),
        "y": torch.tensor([[0.0], [1.0], [0.0]]),
    }
    config = SupervisedTrainingConfig(
        architecture="multimodal",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=2,
        ssl_consistency_weight=0.25,
    )

    with pytest.raises(ValueError, match="ssl consistency requires"):
        _compute_supervised_loss(model, batch, config, training=True)


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


def test_feature_ssl_transfer_run_name_records_transfer_optimizer_tuning():
    run_name = feature_ssl_transfer_run_name(
        data_scope="all-patient",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="cnn",
        transfer_mode="finetune",
        seed="0",
        ssl_objective="barlow",
        pretrain_epochs=50,
        temperature=0.2,
        noise_std=0.02,
        feature_mask_prob=0.01,
        pretrain_batch_size=8,
        supervised_epochs=100,
        supervised_lr=0.002,
        encoder_lr_multiplier=0.1,
        transfer_head_lr_multiplier=2.0,
        freeze_pretrained_encoder_epochs=10,
    )

    assert "encLRx0_1" in run_name
    assert "headLRx2" in run_name
    assert "frzenc10" in run_name


def test_feature_ssl_transfer_run_name_records_ssl_bridge():
    run_name = feature_ssl_transfer_run_name(
        data_scope="all-patient",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="cnn",
        transfer_mode="finetune",
        seed="0",
        ssl_objective="barlow",
        pretrain_epochs=50,
        temperature=0.2,
        noise_std=0.02,
        feature_mask_prob=0.01,
        pretrain_batch_size=8,
        supervised_epochs=100,
        supervised_lr=0.002,
        ssl_bridge_enabled=True,
    )

    assert "sslbridge" in run_name


def test_feature_ssl_transfer_run_name_records_random_main_residual_loss_mode():
    run_name = feature_ssl_transfer_run_name(
        data_scope="all-patient",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="cnn",
        transfer_mode="finetune",
        seed="0",
        ssl_objective="barlow",
        pretrain_epochs=50,
        temperature=0.2,
        noise_std=0.02,
        feature_mask_prob=0.01,
        pretrain_batch_size=8,
        supervised_epochs=100,
        ssl_residual_enabled=True,
        ssl_residual_random_main=True,
        ssl_residual_main_loss_only=True,
    )

    assert "sslresidual" in run_name
    assert "randommain" in run_name
    assert "mainloss" in run_name


def test_feature_ssl_transfer_run_name_records_preserved_main_residual():
    run_name = feature_ssl_transfer_run_name(
        data_scope="all-patient",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="cnn",
        transfer_mode="finetune",
        seed="0",
        ssl_objective="barlow",
        pretrain_epochs=50,
        temperature=0.2,
        noise_std=0.02,
        feature_mask_prob=0.01,
        pretrain_batch_size=8,
        supervised_epochs=100,
        ssl_residual_enabled=True,
        ssl_residual_preserve_main=True,
    )

    assert "sslresidual" in run_name
    assert "preservemain" in run_name


def test_feature_ssl_transfer_run_name_records_logit_delta_residual():
    run_name = feature_ssl_transfer_run_name(
        data_scope="all-patient",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="cnn",
        transfer_mode="finetune",
        seed="0",
        ssl_objective="barlow",
        pretrain_epochs=50,
        temperature=0.2,
        noise_std=0.02,
        feature_mask_prob=0.01,
        pretrain_batch_size=8,
        supervised_epochs=100,
        ssl_residual_enabled=True,
        ssl_residual_preserve_main=True,
        ssl_residual_logit_delta_enabled=True,
        ssl_residual_logit_delta_scale=0.5,
    )

    assert "sslresidual" in run_name
    assert "preservemain" in run_name
    assert "logitdelta0_5" in run_name


def test_feature_ssl_transfer_run_name_records_val_selected_residual_weights():
    run_name = feature_ssl_transfer_run_name(
        data_scope="all-patient",
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="cnn",
        transfer_mode="finetune",
        seed="0",
        ssl_objective="barlow",
        pretrain_epochs=50,
        temperature=0.2,
        noise_std=0.02,
        feature_mask_prob=0.01,
        pretrain_batch_size=8,
        supervised_epochs=100,
        ssl_residual_enabled=True,
        ssl_residual_preserve_main=True,
        ssl_residual_select_weights_on_val=True,
    )

    assert "sslresidual" in run_name
    assert "preservemain" in run_name
    assert "valtune" in run_name


def test_safe_run_name_shortens_long_tokens_for_windows_paths():
    run_name = "feature_ssl_" + ("all-patient_psd-fc-wpli_gated_cnn_barlow_finetune_seed0_" * 6)

    safe = _safe_run_name(run_name)

    assert len(safe) <= 120
    assert safe.startswith("feature_ssl_")
    assert safe.rsplit("_", maxsplit=1)[-1].isalnum()


def test_feature_ssl_transfer_ensemble_paths_use_short_safe_run_name(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "07_train_feature_ssl_transfer.py"
    spec = importlib.util.spec_from_file_location("feature_ssl_transfer_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    long_run_name = "feature_ssl_" + ("all-patient_psd-fc-wpli_gated_cnn_barlow_freeze-encoder_seedensemble3_" * 5)

    prediction_path, metric_path = module._ensemble_output_paths(tmp_path, long_run_name)

    assert prediction_path.parent == tmp_path / "results" / "predictions"
    assert metric_path.parent == tmp_path / "results" / "metrics"
    assert len(prediction_path.name) <= len("dl_loso_predictions_") + 120 + len(".csv")
    assert len(metric_path.name) <= len("dl_model_comparison_") + 120 + len(".csv")


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
    assert "--encoder-lr-multiplier" in result.stdout
    assert "--transfer-head-lr-multiplier" in result.stdout
    assert "--freeze-pretrained-encoder-epochs" in result.stdout
    assert "--ssl-bridge-enabled" in result.stdout
    assert "--ssl-two-head-enabled" in result.stdout
    assert "gncnn" in result.stdout
    assert "rescnn" in result.stdout
    assert "branch-barlow" in result.stdout
    assert "masked-barlow" in result.stdout
    assert "masked-vicreg" in result.stdout
    assert "masked-reconstruction" in result.stdout
    assert "masked-reconstruction-vicreg" in result.stdout
    assert "simsiam" in result.stdout
    assert "--branch-barlow-weight" in result.stdout
    assert "--latent-modality-mask-prob" in result.stdout
    assert "--ssl-consistency-weight" in result.stdout
    assert "--ssl-aux-head-weight" in result.stdout
    assert "--ssl-fusion-weight" in result.stdout
    assert "--embedding-adapter-dim" in result.stdout
    assert "--embedding-adapter-scale" in result.stdout
    assert "--loss-name" in result.stdout
    assert "--negative-class-weight" in result.stdout
    assert "--positive-class-weight" in result.stdout
