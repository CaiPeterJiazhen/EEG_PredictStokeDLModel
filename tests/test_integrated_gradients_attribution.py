from __future__ import annotations

import torch
from torch import nn

from eeg_recovery.explainability.attribution import (
    attribution_correlation,
    manual_integrated_gradients,
    randomize_classifier_weights,
)


class _TinyLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classification_head = nn.Linear(28, 1, bias=False)
        with torch.no_grad():
            weights = torch.linspace(-1.0, 1.0, steps=28).reshape(1, 28)
            self.classification_head.weight.copy_(weights)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = torch.cat(
            [
                batch["psd_eo"].flatten(start_dim=1),
                batch["psd_ec"].flatten(start_dim=1),
                batch["wpli_eo"].flatten(start_dim=1),
                batch["wpli_ec"].flatten(start_dim=1),
            ],
            dim=1,
        )
        logits = self.classification_head(features)
        return {"classification_logits": logits, "classification_probability": torch.sigmoid(logits)}


def _batch() -> dict[str, torch.Tensor]:
    return {
        "psd_eo": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) / 10.0,
        "psd_ec": torch.ones((1, 2, 3), dtype=torch.float32) * 0.2,
        "wpli_eo": torch.ones((1, 4, 2), dtype=torch.float32) * 0.3,
        "wpli_ec": torch.ones((1, 4, 2), dtype=torch.float32) * 0.4,
        "y": torch.tensor([[1.0]]),
    }


def test_integrated_gradients_returns_input_shapes_without_nan() -> None:
    model = _TinyLogitModel()
    batch = _batch()

    attributions = manual_integrated_gradients(
        model,
        batch,
        input_keys=("psd_eo", "psd_ec", "wpli_eo", "wpli_ec"),
        steps=8,
    )

    for key in ("psd_eo", "psd_ec", "wpli_eo", "wpli_ec"):
        assert attributions[key].shape == batch[key].shape
        assert torch.isfinite(attributions[key]).all()


def test_integrated_gradients_does_not_change_model_predictions() -> None:
    model = _TinyLogitModel()
    batch = _batch()
    before = model(batch)["classification_logits"].detach().clone()

    manual_integrated_gradients(model, batch, input_keys=("psd_eo", "wpli_eo"), steps=4)

    after = model(batch)["classification_logits"].detach()
    assert torch.allclose(before, after)


def test_randomized_classifier_changes_attribution_correlation() -> None:
    model = _TinyLogitModel()
    batch = _batch()
    trained = manual_integrated_gradients(model, batch, input_keys=("psd_eo",), steps=8)["psd_eo"]
    randomized = randomize_classifier_weights(model, seed=3)
    randomized_attr = manual_integrated_gradients(randomized, batch, input_keys=("psd_eo",), steps=8)["psd_eo"]

    assert attribution_correlation(trained, randomized_attr) < 0.99
