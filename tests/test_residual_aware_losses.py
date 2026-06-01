from __future__ import annotations

import torch

from eeg_recovery.training.residual_aware_losses import (
    pairwise_ranking_loss,
    residual_aware_multitask_loss,
)


def test_residual_aware_huber_loss_is_finite() -> None:
    outputs = {
        "classification_logits": torch.tensor([[0.1], [-0.3]], requires_grad=True),
        "residual_score": torch.tensor([[0.2], [-0.2]], requires_grad=True),
    }
    batch = {
        "y": torch.tensor([[1.0], [0.0]]),
        "signed_distance_z": torch.tensor([[0.5], [-0.5]]),
        "soft_y": torch.tensor([[0.7], [0.3]]),
    }

    loss, components = residual_aware_multitask_loss(
        outputs,
        batch,
        lambda_reg=0.3,
        lambda_rank=0.1,
        lambda_soft=0.2,
        rank_margin=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(components["loss_residual_regression"])


def test_pairwise_ranking_loss_is_finite() -> None:
    scores = torch.tensor([[0.2], [0.8], [-0.1]], requires_grad=True)
    signed_distance = torch.tensor([[0.0], [1.0], [-1.0]])

    loss = pairwise_ranking_loss(scores, signed_distance, rank_margin=0.5)
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.grad is not None


def test_pairwise_ranking_loss_returns_zero_without_pairs() -> None:
    scores = torch.tensor([[0.2]], requires_grad=True)
    signed_distance = torch.tensor([[0.0]])

    loss = pairwise_ranking_loss(scores, signed_distance, rank_margin=0.5)

    assert float(loss.item()) == 0.0
    assert loss.requires_grad
