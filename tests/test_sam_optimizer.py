from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from eeg_recovery.training.optimizers import SAM
from eeg_recovery.training.train_supervised import SupervisedTrainingConfig, _build_transfer_optimizer


def test_sam_wrapper_performs_two_step_update_without_nan() -> None:
    torch.manual_seed(7)
    model = nn.Linear(3, 1)
    initial_weight = model.weight.detach().clone()
    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=0.01, rho=0.05, weight_decay=1e-5)
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        return loss

    loss = optimizer.step(closure)

    assert torch.isfinite(loss)
    assert torch.isfinite(model.weight).all()
    assert not torch.allclose(initial_weight, model.weight.detach())


def test_default_transfer_optimizer_keeps_adam_path_when_disabled() -> None:
    model = nn.Linear(4, 1)
    config = SupervisedTrainingConfig(optimizer_name="adam", lr=0.003, weight_decay=1e-4)

    optimizer = _build_transfer_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.Adam)
    assert not isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.003


def test_adamw_transfer_optimizer_path_is_explicit() -> None:
    model = nn.Linear(4, 1)
    config = SupervisedTrainingConfig(optimizer_name="adamw", lr=0.003, weight_decay=1e-4)

    optimizer = _build_transfer_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.003
