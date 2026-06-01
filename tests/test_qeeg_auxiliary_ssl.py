from __future__ import annotations

import numpy as np
import torch

from eeg_recovery.training.train_feature_ssl import FeatureSSLPairRecord
from eeg_recovery.training.train_masked_ssl import (
    MaskedSSLConfig,
    _qeeg_auxiliary_loss,
    _standardize_qeeg_targets,
    run_masked_ssl_pretraining,
)


def _pair(index: int, *, qeeg_target: float | None) -> FeatureSSLPairRecord:
    rng = np.random.default_rng(index)
    return FeatureSSLPairRecord(
        group="patient" if qeeg_target is not None else "health",
        subject_id=f"sub{index:02d}" if qeeg_target is not None else f"health{index:02d}",
        subject_key=f"subject:{index}",
        stage="baseline",
        is_supervised_subject=qeeg_target is not None,
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
        qeeg_target=qeeg_target,
    )


def test_qeeg_auxiliary_loss_is_finite_and_masks_missing_targets():
    predictions = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0], [float("nan")], [2.0]], dtype=torch.float32)

    loss = _qeeg_auxiliary_loss(predictions, targets, huber_delta=1.0)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_qeeg_auxiliary_loss_returns_zero_when_all_targets_missing():
    predictions = torch.zeros(2, 1)
    targets = torch.full((2, 1), float("nan"))

    loss = _qeeg_auxiliary_loss(predictions, targets, huber_delta=1.0)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_qeeg_targets_are_fold_scaled_with_fit_subject_metadata():
    pairs = [_pair(1, qeeg_target=0.2), _pair(2, qeeg_target=0.4), _pair(3, qeeg_target=None)]

    targets, metadata = _standardize_qeeg_targets(pairs, feature_name="qeeg_ec_global_slow_fast_bsi")

    assert targets.shape == (3, 1)
    assert np.isfinite(targets[:2]).all()
    assert np.isnan(targets[2, 0])
    assert metadata["qeeg_feature_name"] == "qeeg_ec_global_slow_fast_bsi"
    assert metadata["qeeg_fit_subject_ids"] == "sub01;sub02"
    assert metadata["qeeg_scaler_hash"]


def test_masked_ssl_history_records_qeeg_auxiliary_metadata():
    pairs = [_pair(index, qeeg_target=float(index) / 10.0) for index in range(1, 4)]
    config = MaskedSSLConfig(
        psd_epochs=1,
        fc_epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        projection_dim=4,
        contrastive_weight=0.1,
        alignment_method="vicreg",
        qeeg_auxiliary_enabled=True,
        lambda_qeeg=0.05,
        qeeg_feature_name="qeeg_ec_global_slow_fast_bsi",
        device="cpu",
        seed=7,
    )

    _, history = run_masked_ssl_pretraining(pairs, config)

    assert "qeeg_auxiliary_enabled" in history.columns
    assert "lambda_qeeg" in history.columns
    assert "qeeg_auxiliary_loss" in history.columns
    assert set(history["qeeg_auxiliary_enabled"]) == {True}
    assert set(history["qeeg_feature_name"]) == {"qeeg_ec_global_slow_fast_bsi"}
    assert np.isfinite(history["qeeg_auxiliary_loss"].to_numpy()).all()
