from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


PSD_SHAPE = (62, 90)
FC_SHAPE = (1891, 6)
ENCODER_KINDS = {"cnn", "linear"}


def _validate_encoder_kind(encoder_kind: str) -> str:
    if encoder_kind not in ENCODER_KINDS:
        raise ValueError("encoder_kind must be 'cnn' or 'linear'.")
    return encoder_kind


def _validate_positive_embedding_dim(embedding_dim: int) -> None:
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")


def _validate_feature_shape(features: torch.Tensor, expected: tuple[int, ...]) -> None:
    observed = tuple(features.shape[1:])
    if observed != expected:
        raise ValueError(f"Expected input shape (*, {expected}), got {tuple(features.shape)}.")


class _FlattenLinearEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Iterable[int],
        embedding_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(value) for value in input_shape)
        _validate_positive_embedding_dim(embedding_dim)
        input_dim = 1
        for value in self.input_shape:
            input_dim *= value
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_feature_shape(features, self.input_shape)
        return self.network(features.float())


class PSDConv2DEncoder(nn.Module):
    """Conv2D PSD encoder for tensors shaped as channel x frequency-bin."""

    def __init__(self, embedding_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        _validate_positive_embedding_dim(embedding_dim)
        self.input_shape = PSD_SHAPE
        self.network = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        feature_maps = self.forward_features(features)
        return self.network[11:](feature_maps)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return pre-pooling convolution maps for local masked reconstruction."""

        _validate_feature_shape(features, self.input_shape)
        return self.network[:11](features.float().unsqueeze(1))


class FCConv1DEncoder(nn.Module):
    """Conv1D FC encoder for tensors shaped as edge x frequency-band."""

    def __init__(self, embedding_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        _validate_positive_embedding_dim(embedding_dim)
        self.input_shape = FC_SHAPE
        self.network = nn.Sequential(
            nn.Conv1d(FC_SHAPE[1], 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        feature_maps = self.forward_features(features)
        return self.network[11:](feature_maps)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return pre-pooling edge maps for local masked reconstruction."""

        _validate_feature_shape(features, self.input_shape)
        return self.network[:11](features.float().transpose(1, 2))


class SharedPSDEncoder(nn.Module):
    """Shared EO/EC PSD encoder with selectable CNN or linear backend."""

    def __init__(
        self,
        embedding_dim: int = 16,
        dropout: float = 0.1,
        encoder_kind: str = "cnn",
    ) -> None:
        super().__init__()
        encoder_kind = _validate_encoder_kind(encoder_kind)
        self.encoder_kind = encoder_kind
        self.encoder = (
            PSDConv2DEncoder(embedding_dim=embedding_dim, dropout=dropout)
            if encoder_kind == "cnn"
            else _FlattenLinearEncoder(PSD_SHAPE, embedding_dim=embedding_dim, dropout=dropout)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)


class SharedFCEncoder(nn.Module):
    """Shared EO/EC connectivity encoder with selectable CNN or linear backend."""

    def __init__(
        self,
        embedding_dim: int = 16,
        dropout: float = 0.1,
        encoder_kind: str = "cnn",
    ) -> None:
        super().__init__()
        encoder_kind = _validate_encoder_kind(encoder_kind)
        self.encoder_kind = encoder_kind
        self.encoder = (
            FCConv1DEncoder(embedding_dim=embedding_dim, dropout=dropout)
            if encoder_kind == "cnn"
            else _FlattenLinearEncoder(FC_SHAPE, embedding_dim=embedding_dim, dropout=dropout)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)


class ClinicalMLP(nn.Module):
    """Compact clinical metadata MLP for optional late fusion experiments."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 8,
        hidden_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features.float())
