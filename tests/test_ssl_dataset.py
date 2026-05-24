from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from eeg_recovery.io.index import EEGFileRecord
from eeg_recovery.models.ssl_model import (
    MaskedReconstructionLoss,
    NTXentLoss,
    SSLTimeSeriesModel,
)
from eeg_recovery.training.train_ssl import (
    SSLAugmentationConfig,
    SSLTrainingConfig,
    make_ssl_views,
    run_ssl_pretraining_on_arrays,
    select_ssl_records,
    write_ssl_outputs,
)


def _record(
    *,
    group: str,
    subject_id: str,
    stage: str,
    state: str,
    supervised: bool = False,
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


def test_ssl_scope_supervised_baseline_excludes_loso_test_subject():
    records = [
        _record(group="patient", subject_id="sub01", stage="基线", state="EO", supervised=True),
        _record(group="patient", subject_id="sub01", stage="基线", state="EC", supervised=True),
        _record(group="patient", subject_id="sub02", stage="基线", state="EO", supervised=True),
        _record(group="patient", subject_id="sub02", stage="基线", state="EC", supervised=True),
        _record(group="patient", subject_id="sub03", stage="基线", state="EO", supervised=False),
        _record(group="patient", subject_id="sub02", stage="最终", state="EO", supervised=True),
        _record(group="health", subject_id="health01", stage="health", state="EO"),
    ]

    selected = select_ssl_records(
        records,
        data_scope="supervised-baseline",
        strict_loso_test_subject_id="sub01",
    )

    assert [(record.subject_key, record.stage, record.state) for record in selected] == [
        ("patient:sub02", "基线", "EC"),
        ("patient:sub02", "基线", "EO"),
    ]


@pytest.mark.parametrize(
    ("data_scope", "expected_keys"),
    [
        ("all-patient-baseline", {"patient:sub02", "patient:sub03"}),
        ("all-patient", {"patient:sub02", "patient:sub03"}),
        ("all-patient-health", {"patient:sub02", "patient:sub03", "health:health01"}),
        ("health-only", {"health:health01"}),
    ],
)
def test_ssl_data_scopes_match_requested_unlabeled_pools(data_scope, expected_keys):
    records = [
        _record(group="patient", subject_id="sub01", stage="基线", state="EO", supervised=True),
        _record(group="patient", subject_id="sub02", stage="基线", state="EO", supervised=True),
        _record(group="patient", subject_id="sub03", stage="基线", state="EO", supervised=False),
        _record(group="patient", subject_id="sub03", stage="最终", state="EC", supervised=False),
        _record(group="health", subject_id="health01", stage="health", state="EO"),
    ]

    selected = select_ssl_records(
        records,
        data_scope=data_scope,
        strict_loso_test_subject_id="sub01",
    )

    assert {record.subject_key for record in selected} == expected_keys
    assert all(record.subject_key != "patient:sub01" for record in selected)


def test_ssl_augmentations_are_deterministic_and_return_two_crops():
    array = np.arange(62 * 256, dtype=np.float32).reshape(62, 256)
    config = SSLAugmentationConfig(
        crop_samples=128,
        noise_std=0.01,
        amplitude_scale_range=(0.9, 1.1),
        channel_dropout_prob=0.1,
        time_mask_prob=0.2,
        time_mask_fraction=0.1,
    )

    first = make_ssl_views(array, seed=123, config=config)
    second = make_ssl_views(array, seed=123, config=config)
    different = make_ssl_views(array, seed=124, config=config)

    assert first.view_a.shape == (62, 128)
    assert first.view_b.shape == (62, 128)
    np.testing.assert_allclose(first.view_a, second.view_a)
    np.testing.assert_allclose(first.view_b, second.view_b)
    assert not np.allclose(first.view_a, different.view_a)


def test_ssl_losses_are_finite_and_positive():
    torch.manual_seed(0)
    embeddings_a = torch.randn(4, 8)
    embeddings_b = torch.randn(4, 8)
    contrastive_loss = NTXentLoss(temperature=0.2)(embeddings_a, embeddings_b)

    target = torch.randn(2, 62, 32)
    reconstructed = target + 0.5
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :, :8] = True
    reconstruction_loss = MaskedReconstructionLoss()(reconstructed, target, mask)

    assert torch.isfinite(contrastive_loss)
    assert float(contrastive_loss) > 0.0
    assert torch.isfinite(reconstruction_loss)
    assert float(reconstruction_loss) > 0.0


def test_ssl_model_outputs_projection_and_reconstruction_shapes():
    model = SSLTimeSeriesModel(n_channels=62, embedding_dim=16, projection_dim=8)
    batch = torch.randn(3, 62, 128)

    output = model(batch)

    assert output.sequence_features.shape == (3, 16, 128)
    assert output.projection.shape == (3, 8)
    assert output.reconstruction.shape == (3, 62, 128)


def test_run_ssl_pretraining_on_arrays_writes_epoch_history(tmp_path):
    arrays = [
        np.random.default_rng(0).normal(size=(62, 96)).astype(np.float32),
        np.random.default_rng(1).normal(size=(62, 96)).astype(np.float32),
        np.random.default_rng(2).normal(size=(62, 96)).astype(np.float32),
    ]
    config = SSLTrainingConfig(
        data_scope="supervised-baseline",
        objective="combined",
        epochs=2,
        batch_size=2,
        crop_samples=64,
        embedding_dim=8,
        projection_dim=4,
        lr=1e-3,
        device="cpu",
        seed=3,
    )

    history = run_ssl_pretraining_on_arrays(arrays, config)
    history_path = write_ssl_outputs(history, tmp_path, run_name="unit_ssl")

    assert history["epoch"].tolist() == [1, 2]
    assert np.isfinite(history["loss"].to_numpy()).all()
    assert np.isfinite(history["contrastive_loss"].to_numpy()).all()
    assert np.isfinite(history["reconstruction_loss"].to_numpy()).all()
    assert history_path == tmp_path / "results" / "ssl" / "ssl_pretraining_history_unit_ssl.csv"
    assert history_path.exists()
    saved = pd.read_csv(history_path)
    assert saved["epoch"].tolist() == [1, 2]


def test_train_ssl_script_help_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-B", "scripts/06_train_ssl.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--data-scope" in result.stdout
    assert "supervised-baseline" in result.stdout
    assert "all-patient-health" in result.stdout
    assert "--strict-loso-test-subject" in result.stdout
