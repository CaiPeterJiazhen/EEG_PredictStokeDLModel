from __future__ import annotations

import torch
from torch import nn

from eeg_recovery.models.dual_state_model import DualStateEEGModel


class Fusion3DEEGModel(nn.Module):
    """Model wrapper for tensors shaped as state x feature dimensions."""

    def __init__(
        self,
        feature_kind: str = "psd",
        fusion: str = "concat",
        embedding_dim: int = 16,
        dropout: float = 0.1,
        encoder_kind: str = "cnn",
    ) -> None:
        super().__init__()
        self.dual_state_model = DualStateEEGModel(
            feature_kind=feature_kind,
            fusion=fusion,
            embedding_dim=embedding_dim,
            dropout=dropout,
            encoder_kind=encoder_kind,
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim < 3 or features.shape[1] != 2:
            raise ValueError(f"Expected input shape (batch, 2, ...), got {tuple(features.shape)}.")
        return self.dual_state_model(
            features[:, 0],
            features[:, 1],
            return_aux=return_aux,
        )
