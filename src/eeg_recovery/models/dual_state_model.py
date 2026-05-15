from __future__ import annotations

from typing import Any

import torch
from torch import nn

from eeg_recovery.models.encoders import SharedFCEncoder, SharedPSDEncoder


FC_FEATURE_KINDS = {"fc", "fc-wpli", "fc-icoh"}


class DualStateEEGModel(nn.Module):
    """Shared encoder model for separate EO and EC feature tensors."""

    def __init__(
        self,
        feature_kind: str = "psd",
        fusion: str = "concat",
        embedding_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_kind = _normalize_feature_kind(feature_kind)
        self.fusion = fusion
        self.encoder = _build_encoder(self.feature_kind, embedding_dim, dropout)

        if fusion == "concat":
            classifier_input_dim = embedding_dim * 2
            self.gate = None
        elif fusion == "gated":
            classifier_input_dim = embedding_dim
            self.gate = nn.Linear(embedding_dim * 2, 2)
        else:
            raise ValueError("fusion must be 'concat' or 'gated'.")

        hidden_dim = max(4, min(32, classifier_input_dim))
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        eo_features: torch.Tensor,
        ec_features: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eo_embedding = self.encoder(eo_features)
        ec_embedding = self.encoder(ec_features)
        fused, aux = self._fuse(eo_embedding, ec_embedding)
        probabilities = torch.sigmoid(self.classifier(fused))
        if return_aux:
            return probabilities, aux
        return probabilities

    def _fuse(
        self,
        eo_embedding: torch.Tensor,
        ec_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pair = torch.cat([eo_embedding, ec_embedding], dim=1)
        if self.fusion == "concat":
            return pair, {}
        if self.gate is None:
            raise RuntimeError("Gated fusion requested without a gate layer.")
        state_weights = torch.softmax(self.gate(pair), dim=1)
        stacked = torch.stack([eo_embedding, ec_embedding], dim=1)
        fused = (state_weights.unsqueeze(-1) * stacked).sum(dim=1)
        return fused, {"state_weights": state_weights}


def _normalize_feature_kind(feature_kind: str) -> str:
    if feature_kind == "psd":
        return "psd"
    if feature_kind in FC_FEATURE_KINDS:
        return "fc"
    raise ValueError("feature_kind must be 'psd', 'fc', 'fc-wpli', or 'fc-icoh'.")


def _build_encoder(feature_kind: str, embedding_dim: int, dropout: float) -> nn.Module:
    if feature_kind == "psd":
        return SharedPSDEncoder(embedding_dim=embedding_dim, dropout=dropout)
    if feature_kind == "fc":
        return SharedFCEncoder(embedding_dim=embedding_dim, dropout=dropout)
    raise ValueError(f"Unsupported feature_kind: {feature_kind}")
