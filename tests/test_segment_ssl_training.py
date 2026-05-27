from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from eeg_recovery.models.multimodal_model import MultimodalEEGModel
from eeg_recovery.training.segment_ssl_dataset import SegmentSSLRecord
from eeg_recovery.training.train_segment_ssl import (
    SegmentSSLTrainingConfig,
    augment_segment_batch,
    barlow_twins_loss,
    extract_branch_encoder_state,
    masked_latent_prediction_loss,
    run_segment_ssl_pretraining,
    segment_records_manifest_hash,
    segment_ssl_transfer_run_name,
    vicreg_loss,
)


def _records(n_records: int = 4) -> list[SegmentSSLRecord]:
    rng = np.random.default_rng(23)
    return [
        SegmentSSLRecord(
            group="patient",
            subject_id=f"sub{index + 1:02d}",
            subject_key=f"patient:sub{index + 1:02d}",
            stage="baseline",
            state="EO" if index % 2 == 0 else "EC",
            segment_index=index,
            features={
                "psd": rng.normal(size=(62, 90)).astype(np.float32),
                "wpli": rng.normal(size=(1891, 6)).astype(np.float32),
            },
            source_path=Path(f"sub{index + 1:02d}.npz"),
        )
        for index in range(n_records)
    ]


def test_segment_augmentations_preserve_shapes_without_channel_permutation() -> None:
    batch = {
        "psd": torch.ones(2, 62, 90),
        "wpli": torch.ones(2, 1891, 6),
    }

    augmented = augment_segment_batch(
        batch,
        seed=1,
        psd_channel_mask_prob=1.0,
        psd_frequency_mask_prob=0.0,
        feature_mask_prob=0.0,
        fc_node_mask_prob=0.0,
        fc_edge_mask_prob=1.0,
        fc_band_mask_prob=0.0,
        noise_std=0.0,
        amplitude_scale_range=(1.0, 1.0),
    )

    assert augmented["psd"].shape == (2, 62, 90)
    assert augmented["wpli"].shape == (2, 1891, 6)
    assert torch.count_nonzero(augmented["psd"]) == 0
    assert torch.count_nonzero(augmented["wpli"]) == 0


def test_segment_barlow_and_vicreg_losses_are_finite_without_negatives() -> None:
    projection_a = torch.randn(4, 8)
    projection_b = projection_a + 0.01 * torch.randn(4, 8)

    barlow = barlow_twins_loss(projection_a, projection_b)
    vicreg = vicreg_loss(projection_a, projection_b)

    assert set(barlow) == {"loss", "on_diag", "off_diag"}
    assert set(vicreg) == {"loss", "invariance", "variance", "covariance"}
    assert all(torch.isfinite(value) for value in barlow.values())
    assert all(torch.isfinite(value) for value in vicreg.values())


def test_masked_latent_prediction_loss_uses_stop_gradient_target() -> None:
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(62 * 90, 4))
    predictor = torch.nn.Linear(4, 4)
    clean = torch.randn(3, 62, 90)
    masked = clean.clone()
    masked[:, :5, :] = 0.0

    loss = masked_latent_prediction_loss(
        encoder=encoder,
        predictor=predictor,
        clean=clean,
        masked=masked,
        loss_type="cosine",
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert predictor.weight.grad is not None
    assert encoder[1].weight.grad is not None


def test_segment_ssl_pretraining_returns_loadable_encoder_state_with_masked_latent() -> None:
    config = SegmentSSLTrainingConfig(
        objective="barlow",
        feature_kind="psd-fc-wpli",
        epochs=1,
        batch_size=2,
        embedding_dim=4,
        projection_dim=4,
        lambda_latent=1.0,
        lambda_local=0.0,
        device="cpu",
        seed=24,
    )

    pretrained_state, history = run_segment_ssl_pretraining(_records(), config)
    model = MultimodalEEGModel("psd-fc-wpli", fusion="gated", embedding_dim=4, dropout=0.0, encoder_kind="cnn")
    incompatible = model.load_state_dict(pretrained_state, strict=False)

    assert history["objective"].tolist() == ["barlow"]
    assert history["masked_latent_loss"].iloc[0] >= 0
    assert np.isfinite(history["loss"].to_numpy()).all()
    assert any(key.startswith("branch_models.psd.encoder.encoder.") for key in pretrained_state)
    assert any(key.startswith("branch_models.wpli.encoder.encoder.") for key in pretrained_state)
    assert not incompatible.unexpected_keys


def test_extract_branch_encoder_state_removes_full_model_prefix() -> None:
    config = SegmentSSLTrainingConfig(
        objective="barlow",
        feature_kind="psd-fc-wpli",
        epochs=1,
        batch_size=2,
        embedding_dim=4,
        projection_dim=4,
        lambda_latent=0.0,
        lambda_local=0.0,
        device="cpu",
        seed=26,
    )

    pretrained_state, _ = run_segment_ssl_pretraining(_records(), config)
    psd_encoder_state = extract_branch_encoder_state(pretrained_state, "psd")

    assert psd_encoder_state
    assert all(not key.startswith("branch_models.") for key in psd_encoder_state)
    assert any(key.startswith("encoder.network.") for key in psd_encoder_state)


def test_segment_records_manifest_hash_tracks_source_pool() -> None:
    records = _records()

    first_hash = segment_records_manifest_hash(records)
    second_hash = segment_records_manifest_hash(list(reversed(records)))
    changed_hash = segment_records_manifest_hash(records[:-1])

    assert first_hash == second_hash
    assert first_hash != changed_hash


def test_segment_ssl_pretraining_can_use_vicreg() -> None:
    config = SegmentSSLTrainingConfig(
        objective="vicreg",
        feature_kind="psd",
        epochs=1,
        batch_size=2,
        embedding_dim=4,
        projection_dim=4,
        lambda_latent=0.0,
        lambda_local=0.0,
        device="cpu",
        seed=25,
    )

    _, history = run_segment_ssl_pretraining(_records(), config)

    assert history["objective"].tolist() == ["vicreg"]
    assert np.isfinite(history["vicreg_invariance_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_variance_loss"].to_numpy()).all()
    assert np.isfinite(history["vicreg_covariance_loss"].to_numpy()).all()


def test_segment_ssl_transfer_run_name_uses_segssl_prefix_and_seed() -> None:
    run_name = segment_ssl_transfer_run_name(
        objective="barlow",
        feature_kind="psd-fc-wpli",
        data_scope="all-patient",
        transfer_mode="finetune",
        seed=13,
        pretrain_epochs=20,
        projection_dim=32,
        feature_mask_prob=0.03,
        noise_std=0.02,
        lambda_latent=1.0,
        supervised_epochs=100,
    )

    assert run_name.startswith("segssl_barlow_all-patient_psd-fc-wpli")
    assert "_seed13_" in run_name
    assert "proj32" in run_name
    assert "mlp1_0" in run_name


def test_segment_ssl_transfer_script_help_runs_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-B", "scripts/17_train_segment_ssl_transfer.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--objective" in result.stdout
    assert "barlow" in result.stdout
    assert "vicreg" in result.stdout
    assert "--lambda-latent" in result.stdout
    assert "--seeds" in result.stdout
    assert "--save-ssl-encoders" in result.stdout
    assert "--reuse-ssl-encoders" in result.stdout
    assert "--ssl-checkpoint-dir" in result.stdout
    assert "--force-retrain-ssl" in result.stdout
    assert "--reuse-only" in result.stdout
    assert "--checkpoint-tag" in result.stdout
    assert "fc-wpli" in result.stdout
    assert "psd-fc-wpli" in result.stdout


def test_dual_segment_barlow_transfer_script_help_runs_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-B", "scripts/18_train_dual_segment_barlow_transfer.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--experiment-set" in result.stdout
    assert "dual-finetune" in result.stdout
    assert "dual-freeze" in result.stdout
    assert "dual-low-lr" in result.stdout
    assert "--reuse-ssl-encoders" in result.stdout
    assert "--allow-mismatched-ssl-seeds" in result.stdout
