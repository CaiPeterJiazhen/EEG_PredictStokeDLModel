from __future__ import annotations

import numpy as np
import torch

from eeg_recovery.models.multimodal_model import MultimodalEEGModel, QEEGGuidedMultimodalEEGModel
from eeg_recovery.training.train_supervised import SupervisedTrainingConfig, _build_model


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "psd_eo": torch.randn(batch_size, 62, 90),
        "psd_ec": torch.randn(batch_size, 62, 90),
        "wpli_eo": torch.randn(batch_size, 1891, 6),
        "wpli_ec": torch.randn(batch_size, 1891, 6),
        "qeeg_features": torch.randn(batch_size, 1),
    }


def test_qeeg_guided_model_forward_uses_low_dimensional_qeeg_branch():
    model = QEEGGuidedMultimodalEEGModel(
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=32,
        dropout=0.0,
        qeeg_input_dim=1,
        qeeg_hidden_dim=4,
    )

    embedding, aux = model.extract_embedding(_batch(), return_aux=True)
    probabilities = model(_batch())

    assert embedding.shape == (2, 68)
    assert aux["cnn_embedding"].shape == (2, 64)
    assert aux["qeeg_embedding"].shape == (2, 4)
    assert probabilities.shape == (2, 1)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    qeeg_linears = [module for module in model.qeeg_encoder.modules() if isinstance(module, torch.nn.Linear)]
    assert qeeg_linears[0].in_features == 1
    assert qeeg_linears[-1].out_features == 4


def test_qeeg_guided_model_rejects_high_dimensional_primary_qeeg_input():
    try:
        QEEGGuidedMultimodalEEGModel(
            feature_kind="psd-fc-wpli",
            fusion="gated",
            encoder_kind="linear",
            embedding_dim=8,
            qeeg_input_dim=60,
            primary_qeeg_only=True,
        )
    except ValueError as exc:
        assert "primary qEEG mode" in str(exc)
    else:
        raise AssertionError("Expected high-dimensional primary qEEG mode to be rejected.")


def test_qeeg_branch_disabled_builds_original_multimodal_model():
    disabled = _build_model(
        SupervisedTrainingConfig(
            architecture="multimodal",
            feature_kind="psd-fc-wpli",
            fusion="gated",
            encoder_kind="linear",
            embedding_dim=8,
            qeeg_features_enabled=False,
        )
    )
    enabled = _build_model(
        SupervisedTrainingConfig(
            architecture="multimodal",
            feature_kind="psd-fc-wpli",
            fusion="gated",
            encoder_kind="linear",
            embedding_dim=8,
            qeeg_features_enabled=True,
            qeeg_feature_names=("qeeg_ec_global_slow_fast_bsi",),
        )
    )

    assert isinstance(disabled, MultimodalEEGModel)
    assert not isinstance(disabled, QEEGGuidedMultimodalEEGModel)
    assert isinstance(enabled, QEEGGuidedMultimodalEEGModel)


def test_qeeg_feature_shape_is_one_dimensional_in_primary_config():
    model = QEEGGuidedMultimodalEEGModel(
        feature_kind="psd-fc-wpli",
        fusion="gated",
        encoder_kind="linear",
        embedding_dim=8,
        qeeg_input_dim=1,
    )
    bad = _batch()
    bad["qeeg_features"] = torch.from_numpy(np.zeros((2, 2), dtype=np.float32))

    try:
        model(bad)
    except ValueError as exc:
        assert "qeeg_features" in str(exc)
    else:
        raise AssertionError("Expected qEEG shape validation to fail.")
