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
        embedding_adapter_dim: int = 0,
        embedding_adapter_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if embedding_adapter_dim < 0:
            raise ValueError("embedding_adapter_dim must be non-negative.")
        self.branches = branches_for_feature_kind(feature_kind)
        self.embedding_adapter_scale = float(embedding_adapter_scale)
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
        self.embedding_adapter = _make_embedding_adapter(classifier_input_dim, embedding_adapter_dim, dropout)
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

        branch_embeddings, aux = self.extract_branch_embeddings(batch, return_aux=True)
        combined = torch.cat([branch_embeddings[branch] for branch in self.branches], dim=1)
        if self.embedding_adapter is not None:
            combined = combined + self.embedding_adapter_scale * self.embedding_adapter(combined)
        if return_aux:
            return combined, aux
        return combined

    def extract_branch_embeddings(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return one fused EO/EC embedding per feature branch."""

        embeddings: dict[str, torch.Tensor] = {}
        aux: dict[str, torch.Tensor] = {}
        for branch in self.branches:
            eo_key = f"{branch}_eo"
            ec_key = f"{branch}_ec"
            if eo_key not in batch or ec_key not in batch:
                raise KeyError(f"Multimodal batch is missing required keys {eo_key!r} and/or {ec_key!r}.")
            embedding, branch_aux = self.branch_models[branch](batch[eo_key], batch[ec_key])
            embeddings[branch] = embedding
            if "state_weights" in branch_aux:
                aux[f"{branch}_state_weights"] = branch_aux["state_weights"]
        if return_aux:
            return embeddings, aux
        return embeddings


class QEEGGuidedMultimodalEEGModel(MultimodalEEGModel):
    """PSD/WPLI CNN with a compact fold-standardized qEEG biomarker branch."""

    def __init__(
        self,
        feature_kind: str,
        fusion: str = "concat",
        embedding_dim: int = 16,
        dropout: float = 0.1,
        encoder_kind: str = "cnn",
        qeeg_input_dim: int = 1,
        qeeg_hidden_dim: int = 4,
        qeeg_clip_value: float | None = 3.0,
        primary_qeeg_only: bool = True,
    ) -> None:
        if primary_qeeg_only and qeeg_input_dim != 1:
            raise ValueError("primary qEEG mode is locked to a single biomarker input.")
        if qeeg_input_dim <= 0:
            raise ValueError("qeeg_input_dim must be positive.")
        if qeeg_hidden_dim <= 0:
            raise ValueError("qeeg_hidden_dim must be positive.")
        super().__init__(
            feature_kind=feature_kind,
            fusion=fusion,
            embedding_dim=embedding_dim,
            dropout=dropout,
            encoder_kind=encoder_kind,
        )
        self.qeeg_input_dim = int(qeeg_input_dim)
        self.qeeg_hidden_dim = int(qeeg_hidden_dim)
        self.qeeg_clip_value = qeeg_clip_value
        self.primary_qeeg_only = bool(primary_qeeg_only)
        self.qeeg_encoder = nn.Sequential(
            nn.Linear(self.qeeg_input_dim, self.qeeg_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.qeeg_hidden_dim, self.qeeg_hidden_dim),
            nn.ReLU(),
        )
        cnn_embedding_dim = embedding_dim * len(self.branches)
        classifier_input_dim = cnn_embedding_dim + self.qeeg_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def extract_embedding(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cnn_embedding, aux = super().extract_embedding(batch, return_aux=True)
        qeeg_features = self._qeeg_features(batch)
        qeeg_embedding = self.qeeg_encoder(qeeg_features)
        combined = torch.cat([cnn_embedding, qeeg_embedding], dim=1)
        if return_aux:
            aux["cnn_embedding"] = cnn_embedding
            aux["qeeg_embedding"] = qeeg_embedding
            return combined, aux
        return combined

    def _qeeg_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "qeeg_features" not in batch:
            raise KeyError("QEEG-guided model requires batch key 'qeeg_features'.")
        qeeg = batch["qeeg_features"].float()
        if qeeg.ndim != 2 or qeeg.shape[1] != self.qeeg_input_dim:
            raise ValueError(
                "qeeg_features tensor must have shape (batch, qeeg_input_dim); "
                f"got {tuple(qeeg.shape)} for qeeg_input_dim={self.qeeg_input_dim}."
            )
        if self.qeeg_clip_value is not None:
            clip_value = float(self.qeeg_clip_value)
            qeeg = qeeg.clamp(-clip_value, clip_value)
        if not torch.isfinite(qeeg).all():
            raise ValueError("qeeg_features contains non-finite values.")
        return qeeg


class EEGSummaryGatedMultimodalEEGModel(MultimodalEEGModel):
    """PSD/WPLI CNN with a named EEG-summary branch and modality gates."""

    def __init__(
        self,
        feature_kind: str,
        fusion: str = "concat",
        embedding_dim: int = 16,
        dropout: float = 0.1,
        encoder_kind: str = "cnn",
        summary_input_dim: int = 1,
        summary_feature_names: tuple[str, ...] = ("eeg_summary",),
    ) -> None:
        super().__init__(
            feature_kind=feature_kind,
            fusion=fusion,
            embedding_dim=embedding_dim,
            dropout=dropout,
            encoder_kind=encoder_kind,
        )
        if summary_input_dim <= 0:
            raise ValueError("summary_input_dim must be positive.")
        if len(summary_feature_names) != summary_input_dim:
            raise ValueError("summary_feature_names length must match summary_input_dim.")
        self.summary_input_dim = int(summary_input_dim)
        self.summary_feature_names = tuple(summary_feature_names)
        self.modality_names = (*self.branches, "eeg_summary")
        self.summary_encoder = _make_summary_encoder(self.summary_input_dim, embedding_dim, dropout)
        gated_input_dim = embedding_dim * len(self.modality_names)
        self.modality_gate = nn.Linear(gated_input_dim, len(self.modality_names))
        hidden_dim = max(4, min(32, gated_input_dim))
        self.classifier = nn.Sequential(
            nn.Linear(gated_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def extract_embedding(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branch_embeddings, aux = self.extract_branch_embeddings(batch, return_aux=True)
        if "eeg_summary" not in batch:
            raise KeyError("EEG summary model requires batch key 'eeg_summary'.")
        summary = batch["eeg_summary"].float()
        if summary.ndim != 2 or summary.shape[1] != self.summary_input_dim:
            raise ValueError(
                "eeg_summary tensor must have shape (batch, summary_input_dim); "
                f"got {tuple(summary.shape)}."
            )
        summary_embedding = self.summary_encoder(summary)
        embeddings = [branch_embeddings[branch] for branch in self.branches]
        embeddings.append(summary_embedding)
        raw_combined = torch.cat(embeddings, dim=1)
        modality_weights = torch.softmax(self.modality_gate(raw_combined), dim=1)
        weighted_embeddings = [
            embedding * modality_weights[:, index : index + 1]
            for index, embedding in enumerate(embeddings)
        ]
        combined = torch.cat(weighted_embeddings, dim=1)
        if return_aux:
            aux["modality_weights"] = modality_weights
            aux["eeg_summary_feature_importance"] = self.eeg_summary_feature_importance().to(
                dtype=combined.dtype,
                device=combined.device,
            )
            return combined, aux
        return combined

    def eeg_summary_feature_importance(self) -> torch.Tensor:
        first_linear = next(
            module
            for module in self.summary_encoder.modules()
            if isinstance(module, nn.Linear)
        )
        weights = first_linear.weight.detach().abs().mean(dim=0)
        total = weights.sum()
        if total <= 0:
            return torch.full_like(weights, 1.0 / float(weights.numel()))
        return weights / total


class SSLBridgeMultimodalEEGModel(nn.Module):
    """Classifier that keeps a frozen SSL encoder copy beside the trainable CNN."""

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
        self.ssl_branch_models = nn.ModuleDict(
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
        self.freeze_ssl_branch_models()
        base_dim = embedding_dim * len(self.branches)
        classifier_input_dim = base_dim * 3
        hidden_dim = max(4, min(32, classifier_input_dim))
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode: bool = True) -> SSLBridgeMultimodalEEGModel:
        super().train(mode)
        self.ssl_branch_models.eval()
        return self

    def freeze_ssl_branch_models(self) -> None:
        for parameter in self.ssl_branch_models.parameters():
            parameter.requires_grad = False
        self.ssl_branch_models.eval()

    def load_ssl_bridge_state(self, state: dict[str, torch.Tensor]):
        mapped_state: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            mapped_state[key] = value
            if key.startswith("branch_models."):
                mapped_state[key.replace("branch_models.", "ssl_branch_models.", 1)] = value
        incompatible = self.load_state_dict(mapped_state, strict=False)
        self.freeze_ssl_branch_models()
        return incompatible

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
        trainable_embedding, ssl_embedding, aux = self.extract_bridge_embeddings(batch, return_aux=True)
        bridge_embedding = torch.cat(
            [trainable_embedding, ssl_embedding, torch.abs(trainable_embedding - ssl_embedding)],
            dim=1,
        )
        if return_aux:
            return bridge_embedding, aux
        return bridge_embedding

    def extract_bridge_embeddings(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return trainable and frozen SSL embeddings before bridge concatenation."""

        trainable_embedding, aux = self._extract_from_branch_models(self.branch_models, batch)
        with torch.no_grad():
            ssl_embedding, ssl_aux = self._extract_from_branch_models(self.ssl_branch_models, batch)
        if return_aux:
            aux.update({f"ssl_{key}": value for key, value in ssl_aux.items()})
            return trainable_embedding, ssl_embedding, aux
        return trainable_embedding, ssl_embedding

    def _extract_from_branch_models(
        self,
        branch_models: nn.ModuleDict,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = []
        aux: dict[str, torch.Tensor] = {}
        for branch in self.branches:
            eo_key = f"{branch}_eo"
            ec_key = f"{branch}_ec"
            if eo_key not in batch or ec_key not in batch:
                raise KeyError(f"Multimodal batch is missing required keys {eo_key!r} and/or {ec_key!r}.")
            embedding, branch_aux = branch_models[branch](batch[eo_key], batch[ec_key])
            embeddings.append(embedding)
            if "state_weights" in branch_aux:
                aux[f"{branch}_state_weights"] = branch_aux["state_weights"]
        return torch.cat(embeddings, dim=1), aux


class SSLTwoHeadMultimodalEEGModel(nn.Module):
    """Two-head transfer model with trainable CNN and frozen SSL representations."""

    def __init__(
        self,
        feature_kind: str,
        fusion: str = "concat",
        embedding_dim: int = 16,
        dropout: float = 0.1,
        encoder_kind: str = "cnn",
        ssl_fusion_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0 <= ssl_fusion_weight <= 1:
            raise ValueError("ssl_fusion_weight must be in [0, 1].")
        self.branches = branches_for_feature_kind(feature_kind)
        self.ssl_fusion_weight = float(ssl_fusion_weight)
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
        self.ssl_branch_models = nn.ModuleDict(
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
        self.freeze_ssl_branch_models()
        base_dim = embedding_dim * len(self.branches)
        self.trainable_classifier = _make_binary_classifier(base_dim, dropout)
        self.ssl_classifier = _make_binary_classifier(base_dim, dropout)

    def train(self, mode: bool = True) -> SSLTwoHeadMultimodalEEGModel:
        super().train(mode)
        self.ssl_branch_models.eval()
        return self

    def freeze_ssl_branch_models(self) -> None:
        for parameter in self.ssl_branch_models.parameters():
            parameter.requires_grad = False
        self.ssl_branch_models.eval()

    def load_ssl_bridge_state(self, state: dict[str, torch.Tensor]):
        mapped_state: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            mapped_state[key] = value
            if key.startswith("branch_models."):
                mapped_state[key.replace("branch_models.", "ssl_branch_models.", 1)] = value
        incompatible = self.load_state_dict(mapped_state, strict=False)
        self.freeze_ssl_branch_models()
        return incompatible

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        trainable_embedding, ssl_embedding, aux = self.extract_bridge_embeddings(batch, return_aux=True)
        trainable_logits = self.trainable_classifier(trainable_embedding)
        ssl_logits = self.ssl_classifier(ssl_embedding)
        fused_logits = (1.0 - self.ssl_fusion_weight) * trainable_logits + self.ssl_fusion_weight * ssl_logits
        probabilities = torch.sigmoid(fused_logits)
        if return_aux:
            aux.update(
                {
                    "trainable_logits": trainable_logits,
                    "ssl_logits": ssl_logits,
                    "trainable_probabilities": torch.sigmoid(trainable_logits),
                    "ssl_probabilities": torch.sigmoid(ssl_logits),
                }
            )
            return probabilities, aux
        return probabilities

    def extract_bridge_embeddings(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        trainable_embedding, aux = self._extract_from_branch_models(self.branch_models, batch)
        with torch.no_grad():
            ssl_embedding, ssl_aux = self._extract_from_branch_models(self.ssl_branch_models, batch)
        if return_aux:
            aux.update({f"ssl_{key}": value for key, value in ssl_aux.items()})
            return trainable_embedding, ssl_embedding, aux
        return trainable_embedding, ssl_embedding

    def _extract_from_branch_models(
        self,
        branch_models: nn.ModuleDict,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = []
        aux: dict[str, torch.Tensor] = {}
        for branch in self.branches:
            eo_key = f"{branch}_eo"
            ec_key = f"{branch}_ec"
            if eo_key not in batch or ec_key not in batch:
                raise KeyError(f"Multimodal batch is missing required keys {eo_key!r} and/or {ec_key!r}.")
            embedding, branch_aux = branch_models[branch](batch[eo_key], batch[ec_key])
            embeddings.append(embedding)
            if "state_weights" in branch_aux:
                aux[f"{branch}_state_weights"] = branch_aux["state_weights"]
        return torch.cat(embeddings, dim=1), aux


class SSLResidualMultimodalEEGModel(nn.Module):
    """Low-gate residual transfer model using frozen SSL and bridge heads."""

    def __init__(
        self,
        feature_kind: str,
        fusion: str = "concat",
        embedding_dim: int = 16,
        dropout: float = 0.1,
        encoder_kind: str = "cnn",
        ssl_residual_weight: float = 0.14,
        bridge_residual_weight: float = 0.04,
        preload_trainable_branch: bool = True,
    ) -> None:
        super().__init__()
        if not 0 <= ssl_residual_weight <= 1:
            raise ValueError("ssl_residual_weight must be in [0, 1].")
        if not 0 <= bridge_residual_weight <= 1:
            raise ValueError("bridge_residual_weight must be in [0, 1].")
        if ssl_residual_weight + bridge_residual_weight > 1:
            raise ValueError("ssl_residual_weight + bridge_residual_weight must be <= 1.")
        self.branches = branches_for_feature_kind(feature_kind)
        self.ssl_residual_weight = float(ssl_residual_weight)
        self.bridge_residual_weight = float(bridge_residual_weight)
        self.trainable_weight = 1.0 - self.ssl_residual_weight - self.bridge_residual_weight
        self.preload_trainable_branch = bool(preload_trainable_branch)
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
        self.ssl_branch_models = nn.ModuleDict(
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
        self.freeze_ssl_branch_models()
        base_dim = embedding_dim * len(self.branches)
        self.trainable_classifier = _make_binary_classifier(base_dim, dropout)
        self.ssl_classifier = _make_binary_classifier(base_dim, dropout)
        self.bridge_classifier = _make_binary_classifier(base_dim * 3, dropout)

    def set_residual_weights(self, *, ssl_residual_weight: float, bridge_residual_weight: float) -> None:
        if not 0 <= ssl_residual_weight <= 1:
            raise ValueError("ssl_residual_weight must be in [0, 1].")
        if not 0 <= bridge_residual_weight <= 1:
            raise ValueError("bridge_residual_weight must be in [0, 1].")
        if ssl_residual_weight + bridge_residual_weight > 1:
            raise ValueError("ssl_residual_weight + bridge_residual_weight must be <= 1.")
        self.ssl_residual_weight = float(ssl_residual_weight)
        self.bridge_residual_weight = float(bridge_residual_weight)
        self.trainable_weight = 1.0 - self.ssl_residual_weight - self.bridge_residual_weight

    def train(self, mode: bool = True) -> SSLResidualMultimodalEEGModel:
        super().train(mode)
        self.ssl_branch_models.eval()
        return self

    def freeze_ssl_branch_models(self) -> None:
        for parameter in self.ssl_branch_models.parameters():
            parameter.requires_grad = False
        self.ssl_branch_models.eval()

    def load_ssl_bridge_state(self, state: dict[str, torch.Tensor]):
        mapped_state: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key.startswith("branch_models."):
                if self.preload_trainable_branch:
                    mapped_state[key] = value
                mapped_state[key.replace("branch_models.", "ssl_branch_models.", 1)] = value
            elif self.preload_trainable_branch:
                mapped_state[key] = value
        incompatible = self.load_state_dict(mapped_state, strict=False)
        self.freeze_ssl_branch_models()
        return incompatible

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        trainable_embedding, ssl_embedding, aux = self.extract_bridge_embeddings(batch, return_aux=True)
        bridge_embedding = torch.cat(
            [trainable_embedding, ssl_embedding, torch.abs(trainable_embedding - ssl_embedding)],
            dim=1,
        )
        trainable_logits = self.trainable_classifier(trainable_embedding)
        ssl_logits = self.ssl_classifier(ssl_embedding)
        bridge_logits = self.bridge_classifier(bridge_embedding)
        trainable_probabilities = torch.sigmoid(trainable_logits)
        ssl_probabilities = torch.sigmoid(ssl_logits)
        bridge_probabilities = torch.sigmoid(bridge_logits)
        probabilities = (
            self.trainable_weight * trainable_probabilities
            + self.ssl_residual_weight * ssl_probabilities
            + self.bridge_residual_weight * bridge_probabilities
        )
        if return_aux:
            aux.update(
                {
                    "trainable_logits": trainable_logits,
                    "ssl_logits": ssl_logits,
                    "bridge_logits": bridge_logits,
                    "trainable_probabilities": trainable_probabilities,
                    "ssl_probabilities": ssl_probabilities,
                    "bridge_probabilities": bridge_probabilities,
                }
            )
            return probabilities, aux
        return probabilities

    def extract_bridge_embeddings(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        trainable_embedding, aux = self._extract_from_branch_models(self.branch_models, batch)
        with torch.no_grad():
            ssl_embedding, ssl_aux = self._extract_from_branch_models(self.ssl_branch_models, batch)
        if return_aux:
            aux.update({f"ssl_{key}": value for key, value in ssl_aux.items()})
            return trainable_embedding, ssl_embedding, aux
        return trainable_embedding, ssl_embedding

    def _extract_from_branch_models(
        self,
        branch_models: nn.ModuleDict,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = []
        aux: dict[str, torch.Tensor] = {}
        for branch in self.branches:
            eo_key = f"{branch}_eo"
            ec_key = f"{branch}_ec"
            if eo_key not in batch or ec_key not in batch:
                raise KeyError(f"Multimodal batch is missing required keys {eo_key!r} and/or {ec_key!r}.")
            embedding, branch_aux = branch_models[branch](batch[eo_key], batch[ec_key])
            embeddings.append(embedding)
            if "state_weights" in branch_aux:
                aux[f"{branch}_state_weights"] = branch_aux["state_weights"]
        return torch.cat(embeddings, dim=1), aux


class FrozenMainSSLResidualMultimodalEEGModel(nn.Module):
    """Frozen no-SSL main CNN with small trainable Barlow residual heads."""

    def __init__(
        self,
        feature_kind: str,
        fusion: str = "concat",
        embedding_dim: int = 16,
        dropout: float = 0.1,
        encoder_kind: str = "cnn",
        ssl_residual_weight: float = 0.14,
        bridge_residual_weight: float = 0.04,
    ) -> None:
        super().__init__()
        if not 0 <= ssl_residual_weight <= 1:
            raise ValueError("ssl_residual_weight must be in [0, 1].")
        if not 0 <= bridge_residual_weight <= 1:
            raise ValueError("bridge_residual_weight must be in [0, 1].")
        if ssl_residual_weight + bridge_residual_weight > 1:
            raise ValueError("ssl_residual_weight + bridge_residual_weight must be <= 1.")
        self.branches = branches_for_feature_kind(feature_kind)
        self.ssl_residual_weight = float(ssl_residual_weight)
        self.bridge_residual_weight = float(bridge_residual_weight)
        self.trainable_weight = 1.0 - self.ssl_residual_weight - self.bridge_residual_weight
        self.main_model = MultimodalEEGModel(
            feature_kind=feature_kind,
            fusion=fusion,
            embedding_dim=embedding_dim,
            dropout=dropout,
            encoder_kind=encoder_kind,
        )
        self.ssl_branch_models = nn.ModuleDict(
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
        self.freeze_main_model()
        self.freeze_ssl_branch_models()
        base_dim = embedding_dim * len(self.branches)
        self.ssl_classifier = _make_binary_classifier(base_dim, dropout)
        self.bridge_classifier = _make_binary_classifier(base_dim * 3, dropout)

    def set_residual_weights(self, *, ssl_residual_weight: float, bridge_residual_weight: float) -> None:
        if not 0 <= ssl_residual_weight <= 1:
            raise ValueError("ssl_residual_weight must be in [0, 1].")
        if not 0 <= bridge_residual_weight <= 1:
            raise ValueError("bridge_residual_weight must be in [0, 1].")
        if ssl_residual_weight + bridge_residual_weight > 1:
            raise ValueError("ssl_residual_weight + bridge_residual_weight must be <= 1.")
        self.ssl_residual_weight = float(ssl_residual_weight)
        self.bridge_residual_weight = float(bridge_residual_weight)
        self.trainable_weight = 1.0 - self.ssl_residual_weight - self.bridge_residual_weight

    def train(self, mode: bool = True) -> FrozenMainSSLResidualMultimodalEEGModel:
        super().train(mode)
        self.main_model.eval()
        self.ssl_branch_models.eval()
        return self

    def freeze_main_model(self) -> None:
        for parameter in self.main_model.parameters():
            parameter.requires_grad = False
        self.main_model.eval()

    def freeze_ssl_branch_models(self) -> None:
        for parameter in self.ssl_branch_models.parameters():
            parameter.requires_grad = False
        self.ssl_branch_models.eval()

    def load_frozen_main_state(self, state: dict[str, torch.Tensor]):
        incompatible = self.main_model.load_state_dict(dict(state), strict=False)
        self.freeze_main_model()
        return incompatible

    def load_ssl_bridge_state(self, state: dict[str, torch.Tensor]):
        mapped_state: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key.startswith("branch_models."):
                mapped_state[key.replace("branch_models.", "ssl_branch_models.", 1)] = value
        incompatible = self.load_state_dict(mapped_state, strict=False)
        self.freeze_ssl_branch_models()
        return incompatible

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        main_embedding, ssl_embedding, aux = self.extract_bridge_embeddings(batch, return_aux=True)
        bridge_embedding = torch.cat(
            [main_embedding, ssl_embedding, torch.abs(main_embedding - ssl_embedding)],
            dim=1,
        )
        with torch.no_grad():
            main_probabilities = self.main_model(batch)
        ssl_probabilities = torch.sigmoid(self.ssl_classifier(ssl_embedding))
        bridge_probabilities = torch.sigmoid(self.bridge_classifier(bridge_embedding))
        probabilities = (
            self.trainable_weight * main_probabilities
            + self.ssl_residual_weight * ssl_probabilities
            + self.bridge_residual_weight * bridge_probabilities
        )
        if return_aux:
            aux.update(
                {
                    "trainable_probabilities": main_probabilities,
                    "ssl_probabilities": ssl_probabilities,
                    "bridge_probabilities": bridge_probabilities,
                }
            )
            return probabilities, aux
        return probabilities

    def extract_bridge_embeddings(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            main_embedding = self.main_model.extract_embedding(batch)
            ssl_embedding, ssl_aux = self._extract_from_branch_models(self.ssl_branch_models, batch)
        aux = {f"ssl_{key}": value for key, value in ssl_aux.items()}
        if return_aux:
            return main_embedding, ssl_embedding, aux
        return main_embedding, ssl_embedding

    def _extract_from_branch_models(
        self,
        branch_models: nn.ModuleDict,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = []
        aux: dict[str, torch.Tensor] = {}
        for branch in self.branches:
            eo_key = f"{branch}_eo"
            ec_key = f"{branch}_ec"
            if eo_key not in batch or ec_key not in batch:
                raise KeyError(f"Multimodal batch is missing required keys {eo_key!r} and/or {ec_key!r}.")
            embedding, branch_aux = branch_models[branch](batch[eo_key], batch[ec_key])
            embeddings.append(embedding)
            if "state_weights" in branch_aux:
                aux[f"{branch}_state_weights"] = branch_aux["state_weights"]
        return torch.cat(embeddings, dim=1), aux


class FrozenMainSSLLogitDeltaMultimodalEEGModel(nn.Module):
    """Frozen no-SSL CNN whose logits receive a bounded Barlow bridge correction."""

    def __init__(
        self,
        feature_kind: str,
        fusion: str = "concat",
        embedding_dim: int = 16,
        dropout: float = 0.1,
        encoder_kind: str = "cnn",
        logit_delta_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if logit_delta_scale < 0:
            raise ValueError("logit_delta_scale must be non-negative.")
        self.branches = branches_for_feature_kind(feature_kind)
        self.logit_delta_scale = float(logit_delta_scale)
        self.main_model = MultimodalEEGModel(
            feature_kind=feature_kind,
            fusion=fusion,
            embedding_dim=embedding_dim,
            dropout=dropout,
            encoder_kind=encoder_kind,
        )
        self.ssl_branch_models = nn.ModuleDict(
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
        self.freeze_main_model()
        self.freeze_ssl_branch_models()
        base_dim = embedding_dim * len(self.branches)
        self.delta_head = _make_zero_initialized_binary_head(base_dim * 3, dropout)

    def train(self, mode: bool = True) -> FrozenMainSSLLogitDeltaMultimodalEEGModel:
        super().train(mode)
        self.main_model.eval()
        self.ssl_branch_models.eval()
        return self

    def freeze_main_model(self) -> None:
        for parameter in self.main_model.parameters():
            parameter.requires_grad = False
        self.main_model.eval()

    def freeze_ssl_branch_models(self) -> None:
        for parameter in self.ssl_branch_models.parameters():
            parameter.requires_grad = False
        self.ssl_branch_models.eval()

    def load_frozen_main_state(self, state: dict[str, torch.Tensor]):
        incompatible = self.main_model.load_state_dict(dict(state), strict=False)
        self.freeze_main_model()
        return incompatible

    def load_ssl_bridge_state(self, state: dict[str, torch.Tensor]):
        mapped_state: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key.startswith("branch_models."):
                mapped_state[key.replace("branch_models.", "ssl_branch_models.", 1)] = value
        incompatible = self.load_state_dict(mapped_state, strict=False)
        self.freeze_ssl_branch_models()
        return incompatible

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        main_embedding, ssl_embedding, aux = self.extract_bridge_embeddings(batch, return_aux=True)
        bridge_embedding = torch.cat(
            [main_embedding, ssl_embedding, torch.abs(main_embedding - ssl_embedding)],
            dim=1,
        )
        with torch.no_grad():
            main_probabilities = self.main_model(batch)
        main_logits = torch.logit(main_probabilities.clamp(1e-6, 1.0 - 1e-6))
        raw_delta = self.delta_head(bridge_embedding)
        logit_delta = self.logit_delta_scale * torch.tanh(raw_delta)
        probabilities = torch.sigmoid(main_logits + logit_delta)
        if return_aux:
            aux.update(
                {
                    "trainable_probabilities": main_probabilities,
                    "ssl_probabilities": probabilities,
                    "bridge_probabilities": probabilities,
                    "main_logits": main_logits,
                    "raw_logit_delta": raw_delta,
                    "logit_delta": logit_delta,
                }
            )
            return probabilities, aux
        return probabilities

    def extract_bridge_embeddings(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            main_embedding = self.main_model.extract_embedding(batch)
            ssl_embedding, ssl_aux = self._extract_from_branch_models(self.ssl_branch_models, batch)
        aux = {f"ssl_{key}": value for key, value in ssl_aux.items()}
        if return_aux:
            return main_embedding, ssl_embedding, aux
        return main_embedding, ssl_embedding

    def _extract_from_branch_models(
        self,
        branch_models: nn.ModuleDict,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = []
        aux: dict[str, torch.Tensor] = {}
        for branch in self.branches:
            eo_key = f"{branch}_eo"
            ec_key = f"{branch}_ec"
            if eo_key not in batch or ec_key not in batch:
                raise KeyError(f"Multimodal batch is missing required keys {eo_key!r} and/or {ec_key!r}.")
            embedding, branch_aux = branch_models[branch](batch[eo_key], batch[ec_key])
            embeddings.append(embedding)
            if "state_weights" in branch_aux:
                aux[f"{branch}_state_weights"] = branch_aux["state_weights"]
        return torch.cat(embeddings, dim=1), aux


def _make_binary_classifier(input_dim: int, dropout: float) -> nn.Sequential:
    hidden_dim = max(4, min(32, input_dim))
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def _make_zero_initialized_binary_head(input_dim: int, dropout: float) -> nn.Sequential:
    head = _make_binary_classifier(input_dim, dropout)
    final_linear = head[-1]
    if not isinstance(final_linear, nn.Linear):
        raise RuntimeError("Binary head final layer must be linear.")
    nn.init.zeros_(final_linear.weight)
    nn.init.zeros_(final_linear.bias)
    return head


def _make_embedding_adapter(
    input_dim: int,
    adapter_dim: int,
    dropout: float,
) -> nn.Sequential | None:
    if adapter_dim == 0:
        return None
    adapter = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, adapter_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(adapter_dim, input_dim),
    )
    final_linear = adapter[-1]
    if not isinstance(final_linear, nn.Linear):
        raise RuntimeError("Embedding adapter final layer must be linear.")
    nn.init.zeros_(final_linear.weight)
    nn.init.zeros_(final_linear.bias)
    return adapter


def _make_summary_encoder(input_dim: int, embedding_dim: int, dropout: float) -> nn.Sequential:
    hidden_dim = max(4, min(32, input_dim * 2))
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, embedding_dim),
        nn.ReLU(),
    )


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
