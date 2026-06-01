from __future__ import annotations

import torch
from torch import nn

from eeg_recovery.training.optimizers import build_swa_model, update_swa_batch_norm


def test_swa_model_averages_weights() -> None:
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    swa_model = build_swa_model(model)

    with torch.no_grad():
        model.weight.fill_(3.0)
    swa_model.update_parameters(model)
    with torch.no_grad():
        model.weight.fill_(5.0)
    swa_model.update_parameters(model)

    torch.testing.assert_close(swa_model.module.weight, torch.full_like(model.weight, 4.0))


def test_swa_batch_norm_update_runs() -> None:
    class _BNModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm1d(2)
            self.out = nn.Linear(2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out(self.bn(x))

    model = _BNModel()
    batches = [torch.ones(4, 2), torch.zeros(4, 2)]

    update_swa_batch_norm(batches, model)

    assert torch.isfinite(model.bn.running_mean).all()
    assert torch.isfinite(model.bn.running_var).all()
