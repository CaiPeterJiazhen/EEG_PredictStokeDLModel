from __future__ import annotations

import torch

from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    _asymmetric_focal_binary_cross_entropy,
    _hard_false_positive_penalty,
    _supervised_binary_loss,
    _weighted_binary_cross_entropy,
)


def test_asymmetric_focal_loss_is_finite_for_positive_and_negative_targets() -> None:
    probabilities = torch.tensor([[0.2], [0.8], [0.999999], [0.000001]], dtype=torch.float32)
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    loss = _asymmetric_focal_binary_cross_entropy(
        probabilities,
        targets,
        positive_class_weight=1.0,
        negative_class_weight=1.5,
        gamma_pos=1.0,
        gamma_neg=2.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_hard_false_positive_penalty_only_applies_to_negatives_above_margin() -> None:
    probabilities = torch.tensor([[0.7], [0.5], [0.8]], dtype=torch.float32)
    targets = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)

    penalty = _hard_false_positive_penalty(
        probabilities,
        targets,
        margin=0.6,
        penalty_weight=0.25,
    )

    torch.testing.assert_close(penalty, torch.tensor(0.25 * ((0.7 - 0.6) ** 2) / 3.0))


def test_hard_false_positive_penalty_is_stable_without_negative_targets() -> None:
    probabilities = torch.tensor([[0.9], [0.8]], dtype=torch.float32)
    targets = torch.ones_like(probabilities)

    penalty = _hard_false_positive_penalty(
        probabilities,
        targets,
        margin=0.6,
        penalty_weight=0.25,
    )

    assert torch.isfinite(penalty)
    assert penalty.item() == 0.0


def test_loss_name_bce_keeps_existing_unweighted_bce_behavior() -> None:
    probabilities = torch.tensor([[0.2], [0.8], [0.4]], dtype=torch.float32)
    targets = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float32)
    config = SupervisedTrainingConfig(
        loss_name="bce",
        positive_class_weight=2.0,
        negative_class_weight=3.0,
    )

    observed = _supervised_binary_loss(probabilities, targets, config)
    expected = torch.nn.functional.binary_cross_entropy(probabilities, targets)

    torch.testing.assert_close(observed, expected)


def test_loss_name_weighted_bce_uses_class_weights() -> None:
    probabilities = torch.tensor([[0.2], [0.8], [0.4]], dtype=torch.float32)
    targets = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float32)
    config = SupervisedTrainingConfig(
        loss_name="weighted_bce",
        positive_class_weight=2.0,
        negative_class_weight=3.0,
    )

    observed = _supervised_binary_loss(probabilities, targets, config)
    expected = _weighted_binary_cross_entropy(probabilities, targets, config)

    torch.testing.assert_close(observed, expected)


def test_asymmetric_focal_fp_margin_loss_can_backward() -> None:
    logits = torch.tensor([[-1.0], [1.0], [0.2]], dtype=torch.float32, requires_grad=True)
    probabilities = torch.sigmoid(logits)
    targets = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float32)
    config = SupervisedTrainingConfig(
        loss_name="asymmetric_focal_fp_margin",
        positive_class_weight=1.0,
        negative_class_weight=1.5,
        focal_gamma_pos=1.0,
        focal_gamma_neg=2.0,
        fp_margin=0.6,
        fp_penalty_weight=0.25,
    )

    loss = _supervised_binary_loss(probabilities, targets, config)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
