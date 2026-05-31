from __future__ import annotations

import numpy as np
import torch

from eeg_recovery.models.multimodal_model import MultimodalEEGModel
from eeg_recovery.training.train_feature_ssl import FeatureSSLPairRecord
from eeg_recovery.training.train_masked_ssl import (
    MaskedSSLConfig,
    _fc_graph_smoothness_loss,
    run_masked_ssl_pretraining,
)


def test_fc_graph_smoothness_loss_is_zero_for_constant_edges():
    edge_index = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    reconstruction = torch.ones((2, 3, 2), dtype=torch.float32)

    loss = _fc_graph_smoothness_loss(reconstruction, edge_index)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_fc_graph_smoothness_loss_penalizes_edge_outliers():
    edge_index = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    smooth = torch.ones((1, 3, 1), dtype=torch.float32)
    outlier = torch.tensor([[[1.0], [1.0], [5.0]]], dtype=torch.float32)

    smooth_loss = _fc_graph_smoothness_loss(smooth, edge_index)
    outlier_loss = _fc_graph_smoothness_loss(outlier, edge_index)

    assert outlier_loss.item() > smooth_loss.item()
    assert torch.isfinite(outlier_loss)


def test_masked_ssl_history_records_graph_smoothness_loss_when_enabled():
    rng = np.random.default_rng(23)
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
        for index in range(3)
    ]
    config = MaskedSSLConfig(
        psd_epochs=1,
        fc_epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        fc_graph_smoothness_weight=0.1,
        device="cpu",
        seed=23,
    )

    pretrained_state, history = run_masked_ssl_pretraining(pairs, config)
    model = MultimodalEEGModel("psd-fc-wpli", fusion="gated", embedding_dim=4, dropout=0.0, encoder_kind="cnn")
    incompatible = model.load_state_dict(pretrained_state, strict=False)

    wpli_history = history[history["branch"] == "wpli_masked"]
    assert not wpli_history.empty
    assert "fc_graph_smoothness_loss" in history.columns
    assert "fc_graph_smoothness_weight" in history.columns
    assert np.isfinite(wpli_history["fc_graph_smoothness_loss"].to_numpy()).all()
    assert (wpli_history["fc_graph_smoothness_loss"] >= 0).all()
    assert set(wpli_history["fc_graph_smoothness_weight"]) == {0.1}
    assert wpli_history["objective"].str.contains("graph_smoothness").all()
    assert not incompatible.unexpected_keys
