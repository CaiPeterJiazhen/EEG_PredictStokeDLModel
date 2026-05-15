from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


PSD_SHAPE = (62, 90)
FC_SHAPE = (1891, 6)


class _FlattenLinearEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Iterable[int],
        embedding_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(value) for value in input_shape)
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
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
        expected = self.input_shape
        observed = tuple(features.shape[1:])
        if observed != expected:
            raise ValueError(f"Expected input shape (*, {expected}), got {tuple(features.shape)}.")
        return self.network(features.float())


class SharedPSDEncoder(_FlattenLinearEncoder):
    """Small shared EO/EC PSD encoder for supervised 19-patient training."""

    def __init__(self, embedding_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__(PSD_SHAPE, embedding_dim=embedding_dim, dropout=dropout)


class SharedFCEncoder(_FlattenLinearEncoder):
    """Small shared EO/EC connectivity encoder for supervised 19-patient training."""

    def __init__(self, embedding_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__(FC_SHAPE, embedding_dim=embedding_dim, dropout=dropout)


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
