from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from eeg_recovery.training.train_supervised import (
    SupervisedTrainingConfig,
    _apply_modality_dropout_to_embeddings,
    _compute_supervised_loss,
    _manifold_mixup_loss,
    _mixup_embeddings_and_targets,
    _sample_mixup_lambda,
)


class _DummyEmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 2)
        self.classifier = nn.Linear(2, 1)

    def extract_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(batch["features"])

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.classifier(self.extract_embedding(batch)))


def _batch(n_samples: int = 4) -> dict[str, torch.Tensor]:
    features = torch.arange(float(n_samples * 4), dtype=torch.float32).reshape(n_samples, 4) / 10.0
    labels = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float32)[:n_samples]
    return {"features": features, "y": labels}


def test_sample_mixup_lambda_is_between_zero_and_one() -> None:
    for _ in range(20):
        lam = _sample_mixup_lambda(0.4, torch.device("cpu"))

        assert lam.ndim == 0
        assert 0.0 <= float(lam.item()) <= 1.0


def test_mixup_targets_are_soft_labels_with_expected_shape() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([[0.0], [1.0]])
    permutation = torch.tensor([1, 0])
    lam = torch.tensor(0.25)

    mixed_embeddings, mixed_targets = _mixup_embeddings_and_targets(
        embeddings,
        targets,
        lam=lam,
        permutation=permutation,
    )

    assert mixed_embeddings.shape == embeddings.shape
    assert mixed_targets.shape == targets.shape
    torch.testing.assert_close(mixed_targets, torch.tensor([[0.75], [0.25]]))


def test_manifold_mixup_loss_is_finite_and_backwardable() -> None:
    model = _DummyEmbeddingModel()
    batch = _batch()
    config = SupervisedTrainingConfig(mixup_enabled=True, mixup_alpha=0.4, mixup_weight=1.0)

    loss = _manifold_mixup_loss(model, batch, config)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.encoder.weight.grad is not None
    assert torch.isfinite(model.encoder.weight.grad).all()


def test_compute_supervised_loss_without_mixup_keeps_bce_behavior() -> None:
    model = _DummyEmbeddingModel()
    batch = _batch()
    config = SupervisedTrainingConfig(loss_name="bce", mixup_enabled=False)

    observed = _compute_supervised_loss(model, batch, config, training=True)
    expected = F.binary_cross_entropy(model(batch), batch["y"])

    torch.testing.assert_close(observed, expected)


def test_compute_supervised_loss_disables_mixup_for_validation() -> None:
    model = _DummyEmbeddingModel()
    batch = _batch()
    config = SupervisedTrainingConfig(
        loss_name="bce",
        mixup_enabled=True,
        mixup_alpha=0.4,
        mixup_weight=100.0,
    )

    observed = _compute_supervised_loss(model, batch, config, training=False)
    expected = F.binary_cross_entropy(model(batch), batch["y"])

    torch.testing.assert_close(observed, expected)


def test_mixup_batch_size_one_does_not_error() -> None:
    model = _DummyEmbeddingModel()
    batch = _batch(n_samples=1)
    config = SupervisedTrainingConfig(mixup_enabled=True, mixup_alpha=0.4, mixup_weight=1.0)

    loss = _compute_supervised_loss(model, batch, config, training=True)

    assert torch.isfinite(loss)


def test_modality_dropout_never_drops_all_branches() -> None:
    embeddings = torch.ones((12, 4), dtype=torch.float32)

    dropped = _apply_modality_dropout_to_embeddings(
        embeddings,
        feature_kind="psd-fc-wpli",
        embedding_dim=2,
        dropout_prob=1.0,
    )

    assert (dropped.reshape(12, 2, 2).abs().sum(dim=2).sum(dim=1) > 0).all()
    assert (dropped == 0).any()
