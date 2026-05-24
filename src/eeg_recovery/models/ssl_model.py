from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SSLModelOutput:
    sequence_features: torch.Tensor
    projection: torch.Tensor
    reconstruction: torch.Tensor


class SSLTimeSeriesModel(nn.Module):
    """Lightweight raw-EEG SSL encoder with projection and reconstruction heads."""

    def __init__(
        self,
        n_channels: int = 62,
        embedding_dim: int = 32,
        projection_dim: int = 16,
    ) -> None:
        super().__init__()
        if n_channels <= 0:
            raise ValueError("n_channels must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")

        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, embedding_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Conv1d(embedding_dim, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> SSLModelOutput:
        if features.ndim != 3:
            raise ValueError(f"Expected raw EEG tensor shape (batch, channels, samples), got {tuple(features.shape)}.")
        sequence_features = self.encoder(features.float())
        projection = self.projection_head(sequence_features)
        reconstruction = self.reconstruction_head(sequence_features)
        return SSLModelOutput(
            sequence_features=sequence_features,
            projection=projection,
            reconstruction=reconstruction,
        )


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy contrastive loss."""

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)

    def forward(self, projection_a: torch.Tensor, projection_b: torch.Tensor) -> torch.Tensor:
        if projection_a.shape != projection_b.shape:
            raise ValueError(
                f"Projection shapes must match, got {tuple(projection_a.shape)} and {tuple(projection_b.shape)}."
            )
        if projection_a.ndim != 2:
            raise ValueError(f"Expected projection tensors shaped (batch, features), got {tuple(projection_a.shape)}.")

        batch_size = projection_a.shape[0]
        if batch_size < 1:
            raise ValueError("NTXentLoss requires at least one sample.")

        projections = torch.cat([projection_a, projection_b], dim=0)
        projections = F.normalize(projections, dim=1)
        logits = projections @ projections.T / self.temperature
        logits.fill_diagonal_(-torch.inf)
        labels = torch.arange(2 * batch_size, device=projections.device)
        labels = (labels + batch_size) % (2 * batch_size)
        return F.cross_entropy(logits, labels)


class MaskedReconstructionLoss(nn.Module):
    """MSE over masked samples, falling back to full-tensor MSE if mask is empty."""

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if reconstruction.shape != target.shape:
            raise ValueError(
                f"Reconstruction and target shapes must match, got {tuple(reconstruction.shape)} and {tuple(target.shape)}."
            )
        if mask.shape != target.shape:
            raise ValueError(f"Mask shape must match target shape, got {tuple(mask.shape)} and {tuple(target.shape)}.")
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if not bool(mask.any()):
            return F.mse_loss(reconstruction, target)
        return F.mse_loss(reconstruction[mask], target[mask])


def build_transfer_optimizer(
    model: SSLTimeSeriesModel,
    *,
    freeze_encoder: bool,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Build optimizer param groups for frozen or fine-tuned SSL transfer."""

    if encoder_lr <= 0 or head_lr <= 0:
        raise ValueError("encoder_lr and head_lr must be positive.")
    for parameter in model.encoder.parameters():
        parameter.requires_grad = not freeze_encoder

    if freeze_encoder:
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("encoder.") and parameter.requires_grad
        ]
        return torch.optim.Adam(parameters, lr=head_lr, weight_decay=weight_decay)

    return torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("encoder.")
                ],
                "lr": head_lr,
            },
        ],
        weight_decay=weight_decay,
    )
