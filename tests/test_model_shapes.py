from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from eeg_recovery.config import PathConfig
from eeg_recovery.models.dual_state_model import DualStateEEGModel
from eeg_recovery.models.encoders import FCConv1DEncoder, PSDConv2DEncoder, SharedFCEncoder, SharedPSDEncoder
from eeg_recovery.models.fusion_3d_model import Fusion3DEEGModel
from eeg_recovery.models.multimodal_model import MultimodalEEGModel
from eeg_recovery.training.schedulers import EarlyStopping
from eeg_recovery.training.train_supervised import (
    SupervisedFeatureRecord,
    SupervisedTrainingConfig,
    _build_optimizer,
    _fit_state_scaler,
    _train_validation_subjects,
    _weighted_binary_cross_entropy,
    aggregate_patient_probabilities,
    run_loso_supervised,
    run_loso_supervised_with_history,
    write_loss_history_outputs,
)


def test_cnn_psd_encoder_outputs_embedding_shape():
    model = SharedPSDEncoder(embedding_dim=8, dropout=0.0, encoder_kind="cnn")
    features = torch.randn(2, 62, 90)

    embeddings = model(features)

    assert embeddings.shape == (2, 8)


def test_cnn_fc_encoder_outputs_embedding_shape():
    model = SharedFCEncoder(embedding_dim=8, dropout=0.0, encoder_kind="cnn")
    features = torch.randn(2, 1891, 6)

    embeddings = model(features)

    assert embeddings.shape == (2, 8)


def test_cnn_encoders_expose_pre_pooling_feature_maps_for_local_ssl():
    psd_encoder = PSDConv2DEncoder(embedding_dim=8, dropout=0.0)
    fc_encoder = FCConv1DEncoder(embedding_dim=8, dropout=0.0)

    psd_features = psd_encoder.forward_features(torch.zeros(2, 62, 90))
    fc_features = fc_encoder.forward_features(torch.zeros(2, 1891, 6))

    assert psd_features.shape == (2, 16, 62, 90)
    assert fc_features.shape == (2, 16, 1891)
    assert psd_encoder(torch.zeros(2, 62, 90)).shape == (2, 8)
    assert fc_encoder(torch.zeros(2, 1891, 6)).shape == (2, 8)


@pytest.mark.parametrize(
    ("feature_kind", "input_shape"),
    [
        ("psd", (62, 90)),
        ("fc", (1891, 6)),
    ],
)
def test_dual_state_model_outputs_batch_probability_shape(feature_kind, input_shape):
    model = DualStateEEGModel(feature_kind=feature_kind, fusion="concat", embedding_dim=8, encoder_kind="cnn")
    eo = torch.randn(3, *input_shape)
    ec = torch.randn(3, *input_shape)

    probabilities = model(eo, ec)

    assert probabilities.shape == (3, 1)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))


@pytest.mark.parametrize(
    ("feature_kind", "input_shape"),
    [
        ("psd", (2, 62, 90)),
        ("fc", (2, 1891, 6)),
    ],
)
def test_fusion_3d_model_outputs_batch_probability_shape(feature_kind, input_shape):
    model = Fusion3DEEGModel(feature_kind=feature_kind, fusion="concat", embedding_dim=8, encoder_kind="cnn")
    features = torch.randn(4, *input_shape)

    probabilities = model(features)

    assert probabilities.shape == (4, 1)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_gated_dual_state_model_returns_interpretable_state_weights():
    model = DualStateEEGModel(feature_kind="psd", fusion="gated", embedding_dim=8, encoder_kind="cnn")
    eo = torch.randn(5, 62, 90)
    ec = torch.randn(5, 62, 90)

    probabilities, aux = model(eo, ec, return_aux=True)

    assert probabilities.shape == (5, 1)
    assert aux["state_weights"].shape == (5, 2)
    torch.testing.assert_close(
        aux["state_weights"].sum(dim=1),
        torch.ones(5),
        atol=1e-6,
        rtol=1e-6,
    )


def test_multimodal_model_outputs_batch_probability_for_psd_fc_both():
    model = MultimodalEEGModel(feature_kind="psd-fc-both", fusion="concat", embedding_dim=8, encoder_kind="cnn")
    batch = {
        "psd_eo": torch.randn(2, 62, 90),
        "psd_ec": torch.randn(2, 62, 90),
        "wpli_eo": torch.randn(2, 1891, 6),
        "wpli_ec": torch.randn(2, 1891, 6),
        "icoh_eo": torch.randn(2, 1891, 6),
        "icoh_ec": torch.randn(2, 1891, 6),
    }

    probabilities = model(batch)

    assert probabilities.shape == (2, 1)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_multimodal_model_exposes_reusable_embedding_for_ssl_transfer():
    model = MultimodalEEGModel(feature_kind="psd-fc-wpli", fusion="gated", embedding_dim=8, encoder_kind="cnn")
    batch = {
        "psd_eo": torch.randn(2, 62, 90),
        "psd_ec": torch.randn(2, 62, 90),
        "wpli_eo": torch.randn(2, 1891, 6),
        "wpli_ec": torch.randn(2, 1891, 6),
    }

    embedding, aux = model.extract_embedding(batch, return_aux=True)
    probabilities = model(batch)

    assert embedding.shape == (2, 16)
    assert set(aux) == {"psd_state_weights", "wpli_state_weights"}
    assert probabilities.shape == (2, 1)


def test_gated_multimodal_model_returns_branch_state_weights():
    model = MultimodalEEGModel(feature_kind="psd-fc-both", fusion="gated", embedding_dim=8, encoder_kind="cnn")
    batch = {
        "psd_eo": torch.randn(3, 62, 90),
        "psd_ec": torch.randn(3, 62, 90),
        "wpli_eo": torch.randn(3, 1891, 6),
        "wpli_ec": torch.randn(3, 1891, 6),
        "icoh_eo": torch.randn(3, 1891, 6),
        "icoh_ec": torch.randn(3, 1891, 6),
    }

    probabilities, aux = model(batch, return_aux=True)

    assert probabilities.shape == (3, 1)
    assert set(aux) == {"psd_state_weights", "wpli_state_weights", "icoh_state_weights"}
    for weights in aux.values():
        assert weights.shape == (3, 2)
        torch.testing.assert_close(
            weights.sum(dim=1),
            torch.ones(3),
            atol=1e-6,
            rtol=1e-6,
        )


def test_early_stopping_restores_best_model_weights():
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    early_stopping = EarlyStopping(patience=2, mode="min")

    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.5)
    assert early_stopping.step(0.4, model) is False

    with torch.no_grad():
        model.weight.fill_(9.0)
        model.bias.fill_(9.0)
    assert early_stopping.step(0.6, model) is False
    assert early_stopping.step(0.7, model) is True

    early_stopping.restore_best_weights(model)

    torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
    torch.testing.assert_close(model.bias, torch.full_like(model.bias, 0.5))


def test_early_stopping_restores_buffer_only_state_without_parameters():
    class BufferOnlyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_value", torch.tensor([1.0, 2.0]))

    model = BufferOnlyModule()
    early_stopping = EarlyStopping(patience=1, mode="min")

    assert early_stopping.step(0.1, model) is False
    model.running_value.fill_(9.0)

    early_stopping.restore_best_weights(model)

    torch.testing.assert_close(model.running_value, torch.tensor([1.0, 2.0]))


def test_train_validation_subjects_rotate_by_fold_index():
    train_subjects = ["sub01", "sub02", "sub03", "sub04"]

    first_train, first_val = _train_validation_subjects(train_subjects, fold_index=0)
    second_train, second_val = _train_validation_subjects(train_subjects, fold_index=1)

    assert first_val == ["sub01"]
    assert second_val == ["sub02"]
    assert first_val != second_val
    assert first_val[0] not in first_train
    assert second_val[0] not in second_train


def test_fold_local_state_scaler_excludes_loso_test_subject():
    train_record = SupervisedFeatureRecord(
        subject_id="sub01",
        label=0,
        eo=np.full((2, 2), 1.0, dtype=np.float32),
        ec=np.full((2, 2), 3.0, dtype=np.float32),
    )
    test_record = SupervisedFeatureRecord(
        subject_id="sub02",
        label=1,
        eo=np.full((2, 2), 100.0, dtype=np.float32),
        ec=np.full((2, 2), 200.0, dtype=np.float32),
    )

    mean, _ = _fit_state_scaler([train_record])

    np.testing.assert_allclose(mean, np.full((2, 2), 2.0, dtype=np.float32))
    assert not np.allclose(mean, np.mean([train_record.eo, train_record.ec, test_record.eo, test_record.ec], axis=0))


def test_load_supervised_feature_records_loads_fc_both_as_separate_branches(tmp_path):
    fc_dir = tmp_path / "data" / "features" / "fc"
    fc_dir.mkdir(parents=True)
    wpli = np.arange(1891 * 6, dtype=np.float32).reshape(1891, 6)
    icoh = -wpli
    for state in ("EO", "EC"):
        np.savez_compressed(
            fc_dir / f"sub01_{state}_fc.npz",
            wpli=wpli + (1.0 if state == "EC" else 0.0),
            imaginary_coherence=icoh + (2.0 if state == "EC" else 0.0),
        )
    config = PathConfig(
        patient_info_integrity_xlsx=tmp_path / "integrity.xlsx",
        patient_info_clinical_xlsx=tmp_path / "clinical.xlsx",
        patient_eeg_root=tmp_path / "patient",
        health_eeg_root=tmp_path / "health",
        standard_1005_ced=tmp_path / "standard.ced",
        output_root=tmp_path,
    )
    label_table = pd.DataFrame({"subject_id": ["sub01"], "label": [1]})

    from eeg_recovery.training.train_supervised import load_supervised_feature_records

    records = load_supervised_feature_records(config, label_table, feature_kind="fc-both")

    assert len(records) == 1
    assert set(records[0].modalities) == {"wpli", "icoh"}
    np.testing.assert_allclose(records[0].modalities["wpli"][0], wpli)
    np.testing.assert_allclose(records[0].modalities["wpli"][1], wpli + 1.0)
    np.testing.assert_allclose(records[0].modalities["icoh"][0], icoh)
    np.testing.assert_allclose(records[0].modalities["icoh"][1], icoh + 2.0)


def test_multimodal_batch_uses_independent_branch_scalers():
    record = SupervisedFeatureRecord(
        subject_id="sub01",
        label=1,
        eo=np.zeros((2, 2), dtype=np.float32),
        ec=np.zeros((2, 2), dtype=np.float32),
        modalities={
            "psd": (
                np.full((2, 2), 1.0, dtype=np.float32),
                np.full((2, 2), 3.0, dtype=np.float32),
            ),
            "wpli": (
                np.full((3, 1), 10.0, dtype=np.float32),
                np.full((3, 1), 14.0, dtype=np.float32),
            ),
        },
    )

    from eeg_recovery.training.train_supervised import _make_batch

    scaler = _fit_state_scaler([record])
    batch = _make_batch([record], scaler, torch.device("cpu"), "multimodal")

    assert set(scaler) == {"psd", "wpli"}
    np.testing.assert_allclose(scaler["psd"][0], np.full((2, 2), 2.0, dtype=np.float32))
    np.testing.assert_allclose(scaler["wpli"][0], np.full((3, 1), 12.0, dtype=np.float32))
    assert batch["psd_eo"].shape == (1, 2, 2)
    assert batch["wpli_eo"].shape == (1, 3, 1)
    torch.testing.assert_close(batch["psd_eo"], torch.full((1, 2, 2), -1.0))
    torch.testing.assert_close(batch["wpli_eo"], torch.full((1, 3, 1), -1.0))


def test_run_loso_supervised_rejects_multibranch_features_without_multimodal_architecture():
    records = []
    for subject_id, label in (("sub01", 0), ("sub02", 1)):
        wpli = np.full((1891, 6), float(label + 1), dtype=np.float32)
        icoh = np.full((1891, 6), float(label + 3), dtype=np.float32)
        records.append(
            SupervisedFeatureRecord(
                subject_id=subject_id,
                label=label,
                eo=wpli,
                ec=wpli,
                modalities={
                    "wpli": (wpli, wpli),
                    "icoh": (icoh, icoh),
                },
            )
        )
    config = SupervisedTrainingConfig(
        architecture="dual_state",
        feature_kind="fc-both",
        device="cpu",
        epochs=1,
        patience=1,
    )

    with pytest.raises(ValueError, match=r"fc-both.*architecture=multimodal"):
        run_loso_supervised(records, config)


def test_run_loso_supervised_records_per_fold_epoch_loss_history():
    records = []
    for subject_id, label, value in (("sub01", 0, 0.0), ("sub02", 1, 1.0)):
        eo = np.full((62, 90), value, dtype=np.float32)
        ec = np.full((62, 90), value + 0.5, dtype=np.float32)
        records.append(SupervisedFeatureRecord(subject_id=subject_id, label=label, eo=eo, ec=ec))
    config = SupervisedTrainingConfig(
        architecture="dual_state",
        feature_kind="psd",
        encoder_kind="linear",
        device="cpu",
        epochs=2,
        patience=5,
        embedding_dim=2,
        dropout=0.0,
        seed=1,
    )

    _, _, loss_history = run_loso_supervised_with_history(records, config)

    expected_columns = {
        "model",
        "fold_index",
        "test_subject_id",
        "epoch",
        "train_loss",
        "val_loss",
        "learning_rate",
        "is_best_epoch",
        "stopped_early",
        "architecture",
        "feature_kind",
        "fusion",
        "encoder_kind",
    }
    assert expected_columns.issubset(loss_history.columns)
    assert len(loss_history) == 4
    assert loss_history.groupby("fold_index")["epoch"].apply(list).to_dict() == {0: [1, 2], 1: [1, 2]}
    assert np.isfinite(loss_history["train_loss"].to_numpy()).all()
    assert np.isfinite(loss_history["val_loss"].to_numpy()).all()
    assert (loss_history["learning_rate"] > 0.0).all()


def test_supervised_training_config_passes_weight_decay_to_adam_optimizer():
    model = torch.nn.Linear(2, 1)
    config = SupervisedTrainingConfig(lr=3e-4, weight_decay=1e-4)

    optimizer = _build_optimizer(model.parameters(), config)

    assert optimizer.param_groups[0]["lr"] == 3e-4
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_weighted_binary_cross_entropy_applies_class_specific_weights():
    probabilities = torch.tensor([[0.8], [0.2]], dtype=torch.float32)
    targets = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    config = SupervisedTrainingConfig(positive_class_weight=2.0, negative_class_weight=3.0)

    loss = _weighted_binary_cross_entropy(probabilities, targets, config)
    expected = torch.nn.functional.binary_cross_entropy(
        probabilities,
        targets,
        weight=torch.tensor([[2.0], [3.0]], dtype=torch.float32),
    )

    assert torch.isclose(loss, expected)


def test_write_loss_history_outputs_saves_csv_and_curve(tmp_path):
    loss_history = pd.DataFrame(
        {
            "model": ["demo_model"] * 4,
            "fold_index": [0, 0, 1, 1],
            "test_subject_id": ["sub01", "sub01", "sub02", "sub02"],
            "epoch": [1, 2, 1, 2],
            "train_loss": [0.8, 0.7, 0.9, 0.6],
            "val_loss": [0.85, 0.75, 0.95, 0.65],
            "learning_rate": [0.001] * 4,
            "is_best_epoch": [True, True, True, True],
            "stopped_early": [False, False, False, False],
            "architecture": ["dual_state"] * 4,
            "feature_kind": ["psd"] * 4,
            "fusion": ["concat"] * 4,
            "encoder_kind": ["cnn"] * 4,
        }
    )

    csv_path, figure_path = write_loss_history_outputs(loss_history, tmp_path)

    assert csv_path == tmp_path / "results" / "training_logs" / "dl_loss_history_demo_model.csv"
    assert figure_path == tmp_path / "results" / "figures" / "dl_loss_curve_demo_model.png"
    assert csv_path.exists()
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    saved = pd.read_csv(csv_path)
    assert saved["epoch"].tolist() == [1, 2, 1, 2]


def test_patient_level_probabilities_are_mean_aggregated():
    sample_predictions = pd.DataFrame(
        {
            "subject_id": ["sub02", "sub01", "sub01"],
            "y_true": [0, 1, 1],
            "y_score": [0.4, 0.2, 0.8],
            "fold_index": [1, 0, 0],
        }
    )

    aggregated = aggregate_patient_probabilities(sample_predictions)

    assert aggregated["subject_id"].tolist() == ["sub01", "sub02"]
    np.testing.assert_allclose(aggregated["y_score"].to_numpy(), np.array([0.5, 0.4]))
    assert aggregated["y_true"].tolist() == [1, 0]
    assert aggregated["y_pred"].tolist() == [1, 0]
    assert aggregated["fold_index"].tolist() == [0, 1]


def test_train_supervised_loso_script_help_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-B", "scripts/05_train_supervised_loso.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--architecture" in result.stdout
    assert "--encoder" in result.stdout
    assert "--weight-decay" in result.stdout
    assert "cnn" in result.stdout
    assert "linear" in result.stdout
    assert "multimodal" in result.stdout
    assert "psd-fc-both" in result.stdout
    assert "configs/paths.example.yaml" in result.stdout
