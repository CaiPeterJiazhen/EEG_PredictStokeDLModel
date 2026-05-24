from __future__ import annotations

import torch
from torch import nn

from eeg_recovery.models.encoders import SharedFCEncoder, SharedPSDEncoder


def branches_for_feature_kind(feature_kind: str) -> tuple[str, ...]:
    if feature_kind == "psd":
        return ("psd",)
    if feature_kind in {"fc", "fc-wpli"}:
        return ("wpli",)
    if feature_kind == "fc-icoh":
        return ("icoh",)
    if feature_kind == "fc-both":
        return ("wpli", "icoh")
    if feature_kind == "psd-fc-wpli":
        return ("psd", "wpli")
    if feature_kind == "psd-fc-icoh":
        return ("psd", "icoh")
    if feature_kind == "psd-fc-both":
        return ("psd", "wpli", "icoh")
    raise ValueError(
        "feature_kind must be one of psd, fc-wpli, fc-icoh, fc-both, "
        "psd-fc-wpli, psd-fc-icoh, or psd-fc-both."
    )


class MultimodalEEGModel(nn.Module):
    """Late-fusion model with one lightweight dual-state branch per modality."""

    def __init__(
        self,
        feature_kind: str,
        fusion: str = "concat",
        embedding_dim: int = 16,
        dropout: float = 0.1,
        encoder_kind: str = "cnn",
    ) -> None:
        super().__init__()
        self.branches = branches_for_feature_kind(feature_kind)
        self.branch_models = nn.ModuleDict(
            {
                branch: _DualStateBranch(
                    branch=branch,
                    fusion=fusion,
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    encoder_kind=encoder_kind,
                )
                for branch in self.branches
            }
        )
        classifier_input_dim = embedding_dim * len(self.branches)
        hidden_dim = max(4, min(32, classifier_input_dim))
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedding, aux = self.extract_embedding(batch, return_aux=True)
        probabilities = torch.sigmoid(self.classifier(embedding))
        if return_aux:
            return probabilities, aux
        return probabilities

    def extract_embedding(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the branch-fused embedding before the classifier."""

        embeddings = []
        aux: dict[str, torch.Tensor] = {}
        for branch in self.branches:
            eo_key = f"{branch}_eo"
            ec_key = f"{branch}_ec"
            if eo_key not in batch or ec_key not in batch:
                raise KeyError(f"Multimodal batch is missing required keys {eo_key!r} and/or {ec_key!r}.")
            embedding, branch_aux = self.branch_models[branch](batch[eo_key], batch[ec_key])
            embeddings.append(embedding)
            if "state_weights" in branch_aux:
                aux[f"{branch}_state_weights"] = branch_aux["state_weights"]
        combined = torch.cat(embeddings, dim=1)
        if return_aux:
            return combined, aux
        return combined


class _DualStateBranch(nn.Module):
    def __init__(
        self,
        branch: str,
        fusion: str,
        embedding_dim: int,
        dropout: float,
        encoder_kind: str,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        if branch == "psd":
            self.encoder = SharedPSDEncoder(
                embedding_dim=embedding_dim,
                dropout=dropout,
                encoder_kind=encoder_kind,
            )
        else:
            self.encoder = SharedFCEncoder(
                embedding_dim=embedding_dim,
                dropout=dropout,
                encoder_kind=encoder_kind,
            )
        if fusion == "concat":
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.gate = None
        elif fusion == "gated":
            self.projection = nn.Identity()
            self.gate = nn.Linear(embedding_dim * 2, 2)
        else:
            raise ValueError("fusion must be 'concat' or 'gated'.")

    def forward(
        self,
        eo_features: torch.Tensor,
        ec_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eo_embedding = self.encoder(eo_features)
        ec_embedding = self.encoder(ec_features)
        pair = torch.cat([eo_embedding, ec_embedding], dim=1)
        if self.fusion == "concat":
            return self.projection(pair), {}
        if self.gate is None:
            raise RuntimeError("Gated fusion requested without a gate layer.")
        state_weights = torch.softmax(self.gate(pair), dim=1)
        stacked = torch.stack([eo_embedding, ec_embedding], dim=1)
        embedding = (state_weights.unsqueeze(-1) * stacked).sum(dim=1)
        return embedding, {"state_weights": state_weights}
