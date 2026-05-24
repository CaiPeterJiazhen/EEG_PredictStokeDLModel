from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eeg_recovery.models.multimodal_model import MultimodalEEGModel
from eeg_recovery.training.train_feature_ssl import FeatureSSLPairRecord
from eeg_recovery.training.train_masked_ssl import (
    MaskedSSLConfig,
    _FCMaskedAutoencoder,
    _PSDMaskedAutoencoder,
    _barlow_twins_loss,
    _byol_loss,
    _embedding_consistency_loss,
    _vicreg_loss,
    build_masked_fc_view,
    build_masked_psd_view,
    run_masked_ssl_pretraining,
)
from eeg_recovery.training.train_structured_ssl import build_fc_edge_index


def test_psd_masked_view_masks_channels_and_frequencies():
    psd = np.ones((62, 90), dtype=np.float32)

    masked, mask = build_masked_psd_view(
        psd,
        seed=2,
        channel_mask_prob=1.0,
        frequency_mask_prob=0.0,
        element_mask_prob=0.0,
    )

    assert masked.shape == psd.shape
    assert mask.shape == psd.shape
    assert mask.dtype == np.bool_
    assert mask.all()
    assert np.count_nonzero(masked) == 0


def test_fc_masked_view_masks_incident_edges_for_dropped_nodes():
    fc = np.ones((6, 2), dtype=np.float32)
    edge_index = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
        ],
        dtype=np.int64,
    )

    masked, mask = build_masked_fc_view(
        fc,
        edge_index=edge_index,
        seed=3,
        node_mask_prob=1.0,
        edge_mask_prob=0.0,
        band_mask_prob=0.0,
    )

    assert masked.shape == fc.shape
    assert mask.shape == fc.shape
    assert mask.all()
    assert np.count_nonzero(masked) == 0


def test_masked_ssl_pretraining_returns_loadable_psd_and_wpli_encoder_states():
    rng = np.random.default_rng(4)
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
        for index in range(3)
    ]
    config = MaskedSSLConfig(
        psd_epochs=1,
        fc_epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        device="cpu",
        seed=9,
    )

    pretrained_state, history = run_masked_ssl_pretraining(pairs, config)
    model = MultimodalEEGModel("psd-fc-wpli", fusion="gated", embedding_dim=4, dropout=0.0, encoder_kind="cnn")
    incompatible = model.load_state_dict(pretrained_state, strict=False)

    assert set(history["branch"]) == {"psd_masked", "wpli_masked"}
    assert set(history["objective"]) == {"masked_local_reconstruction"}
    assert np.isfinite(history["loss"].to_numpy()).all()
    assert any(key.startswith("branch_models.psd.encoder.encoder.") for key in pretrained_state)
    assert any(key.startswith("branch_models.wpli.encoder.encoder.") for key in pretrained_state)
    assert not incompatible.unexpected_keys


def test_masked_ssl_can_add_eo_ec_consistency_loss_to_history():
    rng = np.random.default_rng(11)
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
        for index in range(3)
    ]
    config = MaskedSSLConfig(
        psd_epochs=1,
        fc_epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        eo_ec_consistency_weight=0.1,
        device="cpu",
        seed=12,
    )

    _, history = run_masked_ssl_pretraining(pairs, config)

    assert set(history["objective"]) == {"masked_local_reconstruction_with_eo_ec_consistency"}
    assert (history["consistency_loss"] >= 0).all()
    assert np.isfinite(history["reconstruction_loss"].to_numpy()).all()


def test_masked_ssl_can_add_feature_contrastive_loss_to_local_reconstruction():
    rng = np.random.default_rng(13)
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
        for index in range(3)
    ]
    config = MaskedSSLConfig(
        psd_epochs=1,
        fc_epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        contrastive_weight=0.25,
        projection_dim=4,
        contrastive_temperature=0.2,
        contrastive_noise_std=0.01,
        contrastive_feature_mask_prob=0.05,
        device="cpu",
        seed=14,
    )

    pretrained_state, history = run_masked_ssl_pretraining(pairs, config)
    model = MultimodalEEGModel("psd-fc-wpli", fusion="gated", embedding_dim=4, dropout=0.0, encoder_kind="cnn")
    incompatible = model.load_state_dict(pretrained_state, strict=False)

    assert history["branch"].tolist() == ["psd_wpli_multitask"]
    assert history["objective"].tolist() == ["masked_local_reconstruction_with_feature_contrastive"]
    assert np.isfinite(history["contrastive_loss"].to_numpy()).all()
    assert np.isfinite(history["psd_reconstruction_loss"].to_numpy()).all()
    assert np.isfinite(history["fc_reconstruction_loss"].to_numpy()).all()
    assert any(key.startswith("branch_models.psd.gate.") for key in pretrained_state)
    assert any(key.startswith("branch_models.wpli.gate.") for key in pretrained_state)
    assert not incompatible.unexpected_keys


def test_masked_ssl_can_use_vicreg_alignment_without_negative_samples():
    rng = np.random.default_rng(15)
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
        for index in range(3)
    ]
    config = MaskedSSLConfig(
        psd_epochs=1,
        fc_epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        contrastive_weight=0.01,
        alignment_method="vicreg",
        projection_dim=4,
        contrastive_noise_std=0.01,
        contrastive_feature_mask_prob=0.05,
        device="cpu",
        seed=16,
    )

    pretrained_state, history = run_masked_ssl_pretraining(pairs, config)
    model = MultimodalEEGModel("psd-fc-wpli", fusion="gated", embedding_dim=4, dropout=0.0, encoder_kind="cnn")
    incompatible = model.load_state_dict(pretrained_state, strict=False)

    assert history["alignment_method"].tolist() == ["vicreg"]
    assert history["objective"].tolist() == ["masked_local_reconstruction_with_vicreg_alignment"]
    assert np.isfinite(history["alignment_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_invariance_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_variance_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_covariance_loss"].to_numpy()).all()
    assert not incompatible.unexpected_keys


def test_vicreg_loss_reports_invariance_variance_and_covariance_components():
    projection_a = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.0],
        ]
    )
    projection_b = projection_a + 0.01

    components = _vicreg_loss(projection_a, projection_b)

    assert set(components) == {"loss", "invariance", "variance", "covariance"}
    assert float(components["loss"].detach().cpu()) > 0
    assert float(components["invariance"].detach().cpu()) < 0.001
    assert all(torch.isfinite(value) for value in components.values())


def test_masked_ssl_can_use_barlow_alignment_without_negative_samples():
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
        for index in range(3)
    ]
    config = MaskedSSLConfig(
        psd_epochs=1,
        fc_epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        contrastive_weight=0.01,
        alignment_method="barlow",
        projection_dim=4,
        contrastive_noise_std=0.01,
        contrastive_feature_mask_prob=0.05,
        device="cpu",
        seed=18,
    )

    _, history = run_masked_ssl_pretraining(pairs, config)

    assert history["alignment_method"].tolist() == ["barlow"]
    assert history["objective"].tolist() == ["masked_local_reconstruction_with_barlow_alignment"]
    assert np.isfinite(history["alignment_loss"].to_numpy()).all()
    assert np.isfinite(history["barlow_on_diag_loss"].to_numpy()).all()
    assert np.isfinite(history["barlow_off_diag_loss"].to_numpy()).all()


def test_barlow_twins_loss_reports_on_and_off_diagonal_components():
    projection_a = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.0],
        ]
    )
    projection_b = projection_a + 0.01

    components = _barlow_twins_loss(projection_a, projection_b)

    assert set(components) == {"loss", "on_diag", "off_diag"}
    assert float(components["loss"].detach().cpu()) > 0
    assert all(torch.isfinite(value) for value in components.values())


def test_masked_ssl_can_use_byol_alignment_without_negative_samples():
    rng = np.random.default_rng(19)
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
        for index in range(3)
    ]
    config = MaskedSSLConfig(
        psd_epochs=1,
        fc_epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        contrastive_weight=0.01,
        alignment_method="byol",
        projection_dim=4,
        contrastive_noise_std=0.01,
        contrastive_feature_mask_prob=0.05,
        device="cpu",
        seed=20,
    )

    _, history = run_masked_ssl_pretraining(pairs, config)

    assert history["alignment_method"].tolist() == ["byol"]
    assert history["objective"].tolist() == ["masked_local_reconstruction_with_byol_alignment"]
    assert np.isfinite(history["alignment_loss"].to_numpy()).all()
    assert np.isfinite(history["byol_loss"].to_numpy()).all()


def test_byol_loss_reports_symmetric_cosine_prediction_loss():
    prediction_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_b = prediction_a.clone()
    prediction_b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    target_a = prediction_b.clone()

    components = _byol_loss(prediction_a, target_b, prediction_b, target_a)

    assert set(components) == {"loss"}
    assert float(components["loss"].detach().cpu()) == 0.0


def test_embedding_consistency_loss_is_zero_for_identical_embeddings():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss = _embedding_consistency_loss(embeddings, embeddings.clone())

    assert float(loss.detach().cpu()) == 0.0


def test_masked_ssl_decoders_reconstruct_from_pre_pooling_feature_maps():
    psd_model = _PSDMaskedAutoencoder(embedding_dim=4, hidden_dim=8, dropout=0.0)
    fc_model = _FCMaskedAutoencoder(embedding_dim=4, hidden_dim=8, dropout=0.0)

    psd_output = psd_model(torch.zeros(2, 62, 90))
    fc_output = fc_model(torch.zeros(2, 1891, 6))

    assert psd_output.shape == (2, 62, 90)
    assert fc_output.shape == (2, 1891, 6)
    assert any(isinstance(module, nn.Conv2d) for module in psd_model.decoder.modules())
    assert any(isinstance(module, nn.Conv1d) for module in fc_model.decoder.modules())


def test_masked_ssl_uses_full_project_fc_edge_count():
    edge_index = build_fc_edge_index()
    assert edge_index.shape == (1891, 2)


def test_masked_ssl_transfer_script_help_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-B", "scripts/09_train_masked_ssl_transfer.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--psd-channel-mask-prob" in result.stdout
    assert "--fc-edge-mask-prob" in result.stdout
    assert "--contrastive-weight" in result.stdout
    assert "--contrastive-temperature" in result.stdout
    assert "--alignment-method" in result.stdout
    assert "--vicreg-invariance-weight" in result.stdout
    assert "--supervised-weight-decay" in result.stdout
    assert "--negative-class-weight" in result.stdout
    assert "barlow" in result.stdout
    assert "byol" in result.stdout


def test_vicreg_transfer_run_name_stays_short_enough_for_windows_paths():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "09_train_masked_ssl_transfer.py"
    spec = importlib.util.spec_from_file_location("train_masked_ssl_transfer_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        contrastive_weight=0.001,
        alignment_method="vicreg",
        data_scope="all-patient",
        transfer_mode="finetune",
        seed=2,
        psd_ssl_epochs=20,
        fc_ssl_epochs=20,
        fc_node_mask_prob=0.02,
        fc_edge_mask_prob=0.05,
        fc_band_mask_prob=0.02,
        vicreg_invariance_weight=25.0,
        vicreg_variance_weight=10.0,
        vicreg_covariance_weight=0.2,
        ssl_batch_size=8,
        supervised_epochs=100,
    )

    run_name = module._build_run_name(args)

    assert run_name.startswith("mtvicreg_all-patient_psdfcwpli_gated_finetune")
    assert "_vi25_0_vv10_0_vc0_2_" in run_name
    assert len(run_name) < 140


def test_masked_ssl_transfer_run_name_records_nondefault_supervised_regularization():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "09_train_masked_ssl_transfer.py"
    spec = importlib.util.spec_from_file_location("train_masked_ssl_transfer_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        contrastive_weight=0.001,
        alignment_method="vicreg",
        data_scope="all-patient",
        transfer_mode="finetune",
        seed=2,
        psd_ssl_epochs=20,
        fc_ssl_epochs=20,
        fc_node_mask_prob=0.02,
        fc_edge_mask_prob=0.05,
        fc_band_mask_prob=0.02,
        vicreg_invariance_weight=25.0,
        vicreg_variance_weight=25.0,
        vicreg_covariance_weight=1.0,
        supervised_lr=0.002,
        supervised_weight_decay=1e-5,
        ssl_batch_size=8,
        supervised_epochs=100,
    )

    run_name = module._build_run_name(args)

    assert "_slr0_002_swd1e-05_" in run_name
    assert len(run_name) < 150


def test_masked_ssl_transfer_run_name_records_nondefault_capacity_parameters():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "09_train_masked_ssl_transfer.py"
    spec = importlib.util.spec_from_file_location("train_masked_ssl_transfer_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        contrastive_weight=0.001,
        alignment_method="vicreg",
        data_scope="all-patient",
        transfer_mode="finetune",
        seed=2,
        psd_ssl_epochs=20,
        fc_ssl_epochs=20,
        fc_node_mask_prob=0.02,
        fc_edge_mask_prob=0.05,
        fc_band_mask_prob=0.02,
        vicreg_invariance_weight=25.0,
        vicreg_variance_weight=25.0,
        vicreg_covariance_weight=1.0,
        embedding_dim=64,
        projection_dim=32,
        ssl_batch_size=8,
        supervised_epochs=100,
    )

    run_name = module._build_run_name(args)

    assert "_ed64_pd32_" in run_name
    assert len(run_name) < 150


def test_masked_ssl_transfer_run_name_records_nondefault_class_weights():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "09_train_masked_ssl_transfer.py"
    spec = importlib.util.spec_from_file_location("train_masked_ssl_transfer_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        contrastive_weight=0.001,
        alignment_method="vicreg",
        data_scope="all-patient",
        transfer_mode="finetune",
        seed=13,
        psd_ssl_epochs=20,
        fc_ssl_epochs=20,
        fc_node_mask_prob=0.02,
        fc_edge_mask_prob=0.05,
        fc_band_mask_prob=0.02,
        vicreg_invariance_weight=25.0,
        vicreg_variance_weight=25.0,
        vicreg_covariance_weight=1.0,
        positive_class_weight=1.0,
        negative_class_weight=1.5,
        ssl_batch_size=8,
        supervised_epochs=100,
    )

    run_name = module._build_run_name(args)

    assert "_ncw1_5_" in run_name
    assert len(run_name) < 150


def test_ntxent_transfer_run_name_stays_short_enough_for_windows_paths():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "09_train_masked_ssl_transfer.py"
    spec = importlib.util.spec_from_file_location("train_masked_ssl_transfer_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        contrastive_weight=0.01,
        alignment_method="ntxent",
        data_scope="all-patient",
        transfer_mode="finetune",
        seed=13,
        psd_ssl_epochs=20,
        fc_ssl_epochs=20,
        fc_node_mask_prob=0.02,
        fc_edge_mask_prob=0.05,
        fc_band_mask_prob=0.02,
        eo_ec_consistency_weight=0.0,
        contrastive_temperature=0.2,
        contrastive_noise_std=0.02,
        contrastive_feature_mask_prob=0.05,
        ssl_batch_size=8,
        supervised_epochs=100,
    )

    run_name = module._build_run_name(args)

    assert run_name.startswith("mtntxent_all-patient_psdfcwpli_gated_finetune")
    assert "_s13_p20_f20_" in run_name
    assert "_w0_01_t0_2_" in run_name
    assert len(run_name) < 140
