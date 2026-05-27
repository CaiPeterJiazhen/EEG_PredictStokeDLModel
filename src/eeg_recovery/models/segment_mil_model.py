from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from eeg_recovery.models.encoders import SharedFCEncoder


class AttentionPool(nn.Module):
    """Attention pooling over variable-length segment embeddings."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 2:
            raise ValueError(f"Segment embeddings must be 2D, got {tuple(embeddings.shape)}.")
        if embeddings.shape[0] < 1:
            raise ValueError("Attention pooling requires at least one segment.")
        scores = self.network(embeddings).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        pooled = (weights.unsqueeze(1) * embeddings).sum(dim=0)
        return pooled, weights


class WPLISegmentAttentionMILModel(nn.Module):
    """Patient-level attention MIL model over baseline EO/EC wPLI segments."""

    def __init__(
        self,
        *,
        embedding_dim: int = 32,
        attention_hidden_dim: int = 16,
        dropout: float = 0.0,
        encoder_kind: str = "cnn",
        encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if dropout < 0:
            raise ValueError("dropout must be non-negative.")
        self.embedding_dim = int(embedding_dim)
        self.encoder_kind = encoder_kind
        self.encoder = SharedFCEncoder(
            embedding_dim=embedding_dim,
            dropout=dropout,
            encoder_kind=encoder_kind,
        )
        self.eo_attention = AttentionPool(embedding_dim, attention_hidden_dim)
        self.ec_attention = AttentionPool(embedding_dim, attention_hidden_dim)
        self.state_gate = nn.Linear(2 * embedding_dim, 2)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        if encoder_state_dict is not None:
            self.load_encoder_state_dict(encoder_state_dict)

    def load_encoder_state_dict(self, encoder_state_dict: Mapping[str, torch.Tensor]) -> None:
        incompatible = self.encoder.load_state_dict(dict(encoder_state_dict), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                "WPLI encoder checkpoint did not match model encoder: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor | object],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
        eo_segments = _as_segment_tensor(batch, "wpli_eo_segments")
        ec_segments = _as_segment_tensor(batch, "wpli_ec_segments")
        if eo_segments.shape[0] < 1:
            raise ValueError("wpli_eo_segments must contain at least one segment.")
        if ec_segments.shape[0] < 1:
            raise ValueError("wpli_ec_segments must contain at least one segment.")

        eo_embeddings = self.encoder(eo_segments)
        ec_embeddings = self.encoder(ec_segments)
        eo_patient_embedding, eo_attention_weights = self.eo_attention(eo_embeddings)
        ec_patient_embedding, ec_attention_weights = self.ec_attention(ec_embeddings)
        pair = torch.cat([eo_patient_embedding, ec_patient_embedding], dim=0)
        state_weights = torch.softmax(self.state_gate(pair.unsqueeze(0)).squeeze(0), dim=0)
        stacked = torch.stack([eo_patient_embedding, ec_patient_embedding], dim=0)
        patient_embedding = (state_weights.unsqueeze(1) * stacked).sum(dim=0)
        probability = torch.sigmoid(self.classifier(patient_embedding.unsqueeze(0)))
        aux: dict[str, torch.Tensor | int] = {
            "eo_attention_weights": eo_attention_weights,
            "ec_attention_weights": ec_attention_weights,
            "state_weights": state_weights,
            "n_eo_segments": int(eo_segments.shape[0]),
            "n_ec_segments": int(ec_segments.shape[0]),
        }
        return probability, aux


def _as_segment_tensor(batch: Mapping[str, torch.Tensor | object], key: str) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"MIL batch is missing required key {key!r}.")
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"MIL batch key {key!r} must be a torch.Tensor.")
    if value.ndim != 3 or tuple(value.shape[1:]) != (1891, 6):
        raise ValueError(f"{key} must be shaped (n_segments, 1891, 6), got {tuple(value.shape)}.")
    return value.float()

