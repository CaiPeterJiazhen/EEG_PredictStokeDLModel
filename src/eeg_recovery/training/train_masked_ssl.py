from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from eeg_recovery.models.encoders import FCConv1DEncoder, FC_SHAPE, PSDConv2DEncoder, PSD_SHAPE
from eeg_recovery.models.multimodal_model import MultimodalEEGModel
from eeg_recovery.models.ssl_model import NTXentLoss
from eeg_recovery.training.train_feature_ssl import FeatureSSLPairRecord, _augment_feature_batch
from eeg_recovery.training.train_structured_ssl import build_fc_edge_index
from eeg_recovery.training.train_supervised import (
    SupervisedFeatureRecord,
    _fit_state_scaler,
    _make_batch,
    resolve_device,
)


ALIGNMENT_METHODS = {"ntxent", "vicreg", "barlow", "byol"}


@dataclass(frozen=True)
class MaskedSSLConfig:
    psd_epochs: int = 20
    fc_epochs: int = 20
    batch_size: int = 8
    embedding_dim: int = 32
    hidden_dim: int = 128
    lr: float = 1e-3
    dropout: float = 0.0
    psd_channel_mask_prob: float = 0.15
    psd_frequency_mask_prob: float = 0.15
    psd_element_mask_prob: float = 0.02
    fc_node_mask_prob: float = 0.05
    fc_edge_mask_prob: float = 0.15
    fc_band_mask_prob: float = 0.05
    eo_ec_consistency_weight: float = 0.0
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.2
    projection_dim: int = 16
    contrastive_noise_std: float = 0.02
    contrastive_feature_mask_prob: float = 0.05
    alignment_method: str = "ntxent"
    vicreg_invariance_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    vicreg_variance_target: float = 1.0
    vicreg_eps: float = 1e-4
    barlow_offdiag_weight: float = 0.005
    barlow_eps: float = 1e-9
    byol_momentum: float = 0.99
    byol_predictor_hidden_dim: int = 64
    device: str = "auto"
    seed: int = 42


def build_masked_psd_view(
    psd: np.ndarray,
    *,
    seed: int,
    channel_mask_prob: float = 0.15,
    frequency_mask_prob: float = 0.15,
    element_mask_prob: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask PSD channel-frequency entries and return the masked view plus target mask."""

    array = np.asarray(psd, dtype=np.float32)
    if array.shape != PSD_SHAPE:
        raise ValueError(f"Expected PSD shape {PSD_SHAPE}, got {array.shape}.")
    _validate_probability("channel_mask_prob", channel_mask_prob)
    _validate_probability("frequency_mask_prob", frequency_mask_prob)
    _validate_probability("element_mask_prob", element_mask_prob)

    rng = np.random.default_rng(seed)
    mask = np.zeros(array.shape, dtype=np.bool_)
    if channel_mask_prob > 0:
        mask[rng.random(array.shape[0]) < channel_mask_prob, :] = True
    if frequency_mask_prob > 0:
        mask[:, rng.random(array.shape[1]) < frequency_mask_prob] = True
    if element_mask_prob > 0:
        mask |= rng.random(array.shape) < element_mask_prob
    _ensure_nonempty_mask(mask, rng)
    masked = np.array(array, copy=True)
    masked[mask] = 0.0
    return masked.astype(np.float32, copy=False), mask


def build_masked_fc_view(
    fc: np.ndarray,
    *,
    edge_index: np.ndarray,
    seed: int,
    node_mask_prob: float = 0.05,
    edge_mask_prob: float = 0.15,
    band_mask_prob: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask FC graph edges/bands and return the masked view plus target mask."""

    array = np.asarray(fc, dtype=np.float32)
    if array.shape != FC_SHAPE and edge_index.shape[0] != array.shape[0]:
        raise ValueError(
            f"Expected FC shape {FC_SHAPE} or matching edge count, got fc={array.shape}, "
            f"edge_index={edge_index.shape}."
        )
    if edge_index.shape != (array.shape[0], 2):
        raise ValueError(f"edge_index shape must be ({array.shape[0]}, 2), got {edge_index.shape}.")
    _validate_probability("node_mask_prob", node_mask_prob)
    _validate_probability("edge_mask_prob", edge_mask_prob)
    _validate_probability("band_mask_prob", band_mask_prob)

    rng = np.random.default_rng(seed)
    mask = np.zeros(array.shape, dtype=np.bool_)
    n_nodes = int(edge_index.max()) + 1
    if node_mask_prob > 0:
        dropped_nodes = rng.random(n_nodes) < node_mask_prob
        incident = dropped_nodes[edge_index[:, 0]] | dropped_nodes[edge_index[:, 1]]
        mask[incident, :] = True
    if edge_mask_prob > 0:
        mask[rng.random(array.shape[0]) < edge_mask_prob, :] = True
    if band_mask_prob > 0:
        mask[:, rng.random(array.shape[1]) < band_mask_prob] = True
    _ensure_nonempty_mask(mask, rng)
    masked = np.array(array, copy=True)
    masked[mask] = 0.0
    return masked.astype(np.float32, copy=False), mask


def run_masked_ssl_pretraining(
    pairs: Iterable[FeatureSSLPairRecord],
    config: MaskedSSLConfig,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    _validate_masked_ssl_config(config)
    pairs = list(pairs)
    if not pairs:
        raise ValueError("Masked SSL pretraining requires at least one EO/EC pair.")
    records = [_pair_to_supervised_record(pair) for pair in pairs]
    scaler = _fit_state_scaler(records)
    if not isinstance(scaler, dict):
        raise ValueError("Masked SSL requires branch-specific scalers.")

    device = resolve_device(config.device)
    if config.contrastive_weight > 0:
        return _pretrain_multitask_masked_contrastive(pairs, scaler, config, device)

    psd_pair_states = _scaled_branch_pair_states(pairs, scaler, "psd")
    wpli_pair_states = _scaled_branch_pair_states(pairs, scaler, "wpli")
    psd_state, psd_history = _pretrain_psd_masked_autoencoder(psd_pair_states, config, device)
    fc_state, fc_history = _pretrain_fc_masked_autoencoder(wpli_pair_states, config, device)

    state: dict[str, torch.Tensor] = {}
    state.update({f"branch_models.psd.encoder.encoder.{key}": value for key, value in psd_state.items()})
    state.update({f"branch_models.wpli.encoder.encoder.{key}": value for key, value in fc_state.items()})
    return state, pd.DataFrame([*psd_history, *fc_history])


class _MultitaskMaskedContrastiveModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        projection_dim: int,
        dropout: float,
        predictor_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = MultimodalEEGModel(
            "psd-fc-wpli",
            fusion="gated",
            embedding_dim=embedding_dim,
            dropout=dropout,
            encoder_kind="cnn",
        )
        self.projection_head = nn.Linear(embedding_dim * len(self.backbone.branches), projection_dim)
        predictor_hidden_dim = predictor_hidden_dim or max(hidden_dim, projection_dim)
        self.predictor_head = nn.Sequential(
            nn.Linear(projection_dim, predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(predictor_hidden_dim, projection_dim),
        )
        self.psd_decoder = nn.Sequential(
            nn.Conv2d(16, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.fc_decoder = nn.Sequential(
            nn.Conv1d(16, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, FC_SHAPE[1], kernel_size=1),
        )

    def project(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.projection_head(self.backbone.extract_embedding(batch))

    def predict(self, projection: torch.Tensor) -> torch.Tensor:
        return self.predictor_head(projection)

    def reconstruct_psd(self, features: torch.Tensor) -> torch.Tensor:
        feature_maps = self._psd_encoder().forward_features(features)
        return self.psd_decoder(feature_maps).squeeze(1)

    def reconstruct_wpli(self, features: torch.Tensor) -> torch.Tensor:
        feature_maps = self._wpli_encoder().forward_features(features)
        return self.fc_decoder(feature_maps).transpose(1, 2)

    def psd_consistency_loss(self, pair_states: list[tuple[np.ndarray, np.ndarray]], device: torch.device) -> torch.Tensor:
        return _pair_consistency_loss(self._psd_encoder(), pair_states, device)

    def wpli_consistency_loss(self, pair_states: list[tuple[np.ndarray, np.ndarray]], device: torch.device) -> torch.Tensor:
        return _pair_consistency_loss(self._wpli_encoder(), pair_states, device)

    def pretrained_state(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.backbone.state_dict().items()
            if key.startswith("branch_models.")
        }

    def _psd_encoder(self) -> PSDConv2DEncoder:
        encoder = self.backbone.branch_models["psd"].encoder.encoder
        if not isinstance(encoder, PSDConv2DEncoder):
            raise TypeError("Multi-task masked SSL requires the CNN PSD encoder.")
        return encoder

    def _wpli_encoder(self) -> FCConv1DEncoder:
        encoder = self.backbone.branch_models["wpli"].encoder.encoder
        if not isinstance(encoder, FCConv1DEncoder):
            raise TypeError("Multi-task masked SSL requires the CNN FC encoder.")
        return encoder


def _pretrain_multitask_masked_contrastive(
    pairs: list[FeatureSSLPairRecord],
    scaler: dict[str, tuple[np.ndarray, np.ndarray]],
    config: MaskedSSLConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    records = [_pair_to_supervised_record(pair) for pair in pairs]
    psd_pair_states = _scaled_branch_pair_states(pairs, scaler, "psd")
    wpli_pair_states = _scaled_branch_pair_states(pairs, scaler, "wpli")
    _validate_arrays([state for pair in psd_pair_states for state in pair], PSD_SHAPE, "PSD")
    _validate_arrays([state for pair in wpli_pair_states for state in pair], FC_SHAPE, "wPLI")

    edge_index = build_fc_edge_index()
    model = _MultitaskMaskedContrastiveModel(
        config.embedding_dim,
        config.hidden_dim,
        config.projection_dim,
        config.dropout,
        config.byol_predictor_hidden_dim,
    ).to(device)
    target_model = _build_byol_target_model(model) if config.alignment_method == "byol" else None
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    ntxent_loss_fn = NTXentLoss(temperature=config.contrastive_temperature)
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []
    n_epochs = max(config.psd_epochs, config.fc_epochs)

    for epoch in range(1, n_epochs + 1):
        order = rng.permutation(len(records))
        losses: list[float] = []
        psd_reconstruction_losses: list[float] = []
        fc_reconstruction_losses: list[float] = []
        alignment_losses: list[float] = []
        vicreg_invariance_losses: list[float] = []
        vicreg_variance_losses: list[float] = []
        vicreg_covariance_losses: list[float] = []
        barlow_on_diag_losses: list[float] = []
        barlow_off_diag_losses: list[float] = []
        byol_losses: list[float] = []
        consistency_losses: list[float] = []
        model.train()
        for start in range(0, len(order), config.batch_size):
            batch_indices = [int(index) for index in order[start : start + config.batch_size]]
            batch_records = [records[index] for index in batch_indices]
            batch_psd_pairs = [psd_pair_states[index] for index in batch_indices]
            batch_wpli_pairs = [wpli_pair_states[index] for index in batch_indices]
            batch = _make_batch(batch_records, scaler, device, "multimodal")
            view_a = _augment_feature_batch(
                batch,
                noise_std=config.contrastive_noise_std,
                feature_mask_prob=config.contrastive_feature_mask_prob,
                seed=config.seed + epoch * 100_000 + start,
            )
            view_b = _augment_feature_batch(
                batch,
                noise_std=config.contrastive_noise_std,
                feature_mask_prob=config.contrastive_feature_mask_prob,
                seed=config.seed + epoch * 100_000 + start + 1,
            )

            optimizer.zero_grad()
            projection_a = model.project(view_a)
            projection_b = model.project(view_b)
            alignment_components = (
                _byol_alignment_components(model, target_model, view_a, view_b)
                if config.alignment_method == "byol"
                else _alignment_loss_components(
                    projection_a,
                    projection_b,
                    config,
                    ntxent_loss_fn,
                )
            )
            psd_reconstruction_loss = (
                _multitask_psd_reconstruction_loss(
                    model,
                    batch_psd_pairs,
                    config,
                    device,
                    epoch,
                    start,
                )
                if epoch <= config.psd_epochs
                else torch.zeros((), device=device)
            )
            fc_reconstruction_loss = (
                _multitask_fc_reconstruction_loss(
                    model,
                    batch_wpli_pairs,
                    edge_index,
                    config,
                    device,
                    epoch,
                    start,
                )
                if epoch <= config.fc_epochs
                else torch.zeros((), device=device)
            )
            consistency_loss = _multitask_consistency_loss(
                model,
                batch_psd_pairs,
                batch_wpli_pairs,
                config,
                device,
            )
            loss = (
                psd_reconstruction_loss
                + fc_reconstruction_loss
                + config.contrastive_weight * alignment_components["loss"]
                + config.eo_ec_consistency_weight * consistency_loss
            )
            loss.backward()
            optimizer.step()
            if target_model is not None:
                _update_byol_target_model(target_model, model, config.byol_momentum)

            losses.append(float(loss.detach().cpu().item()))
            psd_reconstruction_losses.append(float(psd_reconstruction_loss.detach().cpu().item()))
            fc_reconstruction_losses.append(float(fc_reconstruction_loss.detach().cpu().item()))
            alignment_losses.append(float(alignment_components["loss"].detach().cpu().item()))
            vicreg_invariance_losses.append(float(alignment_components["invariance"].detach().cpu().item()))
            vicreg_variance_losses.append(float(alignment_components["variance"].detach().cpu().item()))
            vicreg_covariance_losses.append(float(alignment_components["covariance"].detach().cpu().item()))
            barlow_on_diag_losses.append(float(alignment_components["on_diag"].detach().cpu().item()))
            barlow_off_diag_losses.append(float(alignment_components["off_diag"].detach().cpu().item()))
            byol_losses.append(float(alignment_components["byol"].detach().cpu().item()))
            consistency_losses.append(float(consistency_loss.detach().cpu().item()))
        rows.append(
            _multitask_history_row(
                epoch,
                losses,
                psd_reconstruction_losses,
                fc_reconstruction_losses,
                alignment_losses,
                vicreg_invariance_losses,
                vicreg_variance_losses,
                vicreg_covariance_losses,
                barlow_on_diag_losses,
                barlow_off_diag_losses,
                byol_losses,
                consistency_losses,
                len(records),
                config,
            )
        )

    return model.pretrained_state(), pd.DataFrame(rows)


def _alignment_loss_components(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    config: MaskedSSLConfig,
    ntxent_loss_fn: NTXentLoss,
) -> dict[str, torch.Tensor]:
    if config.alignment_method == "ntxent":
        loss = ntxent_loss_fn(projection_a, projection_b)
        return {
            "loss": loss,
            "invariance": torch.zeros((), device=projection_a.device),
            "variance": torch.zeros((), device=projection_a.device),
            "covariance": torch.zeros((), device=projection_a.device),
            "on_diag": torch.zeros((), device=projection_a.device),
            "off_diag": torch.zeros((), device=projection_a.device),
            "byol": torch.zeros((), device=projection_a.device),
        }
    if config.alignment_method == "vicreg":
        if projection_a.shape[0] < 2:
            zero = torch.zeros((), device=projection_a.device)
            return {
                "loss": zero,
                "invariance": zero,
                "variance": zero,
                "covariance": zero,
                "on_diag": zero,
                "off_diag": zero,
                "byol": zero,
            }
        components = _vicreg_loss(
            projection_a,
            projection_b,
            invariance_weight=config.vicreg_invariance_weight,
            variance_weight=config.vicreg_variance_weight,
            covariance_weight=config.vicreg_covariance_weight,
            variance_target=config.vicreg_variance_target,
            eps=config.vicreg_eps,
        )
        zero = torch.zeros((), device=projection_a.device)
        return {**components, "on_diag": zero, "off_diag": zero, "byol": zero}
    if config.alignment_method == "barlow":
        if projection_a.shape[0] < 2:
            zero = torch.zeros((), device=projection_a.device)
            return {
                "loss": zero,
                "invariance": zero,
                "variance": zero,
                "covariance": zero,
                "on_diag": zero,
                "off_diag": zero,
                "byol": zero,
            }
        components = _barlow_twins_loss(
            projection_a,
            projection_b,
            offdiag_weight=config.barlow_offdiag_weight,
            eps=config.barlow_eps,
        )
        zero = torch.zeros((), device=projection_a.device)
        return {
            "loss": components["loss"],
            "invariance": zero,
            "variance": zero,
            "covariance": zero,
            "on_diag": components["on_diag"],
            "off_diag": components["off_diag"],
            "byol": zero,
        }
    raise ValueError(f"Unsupported alignment_method: {config.alignment_method}")


def _vicreg_loss(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    *,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    variance_target: float = 1.0,
    eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    if projection_a.shape != projection_b.shape:
        raise ValueError(
            f"Projection shapes must match, got {tuple(projection_a.shape)} and {tuple(projection_b.shape)}."
        )
    if projection_a.ndim != 2:
        raise ValueError(f"Expected projection tensors shaped (batch, features), got {tuple(projection_a.shape)}.")
    if projection_a.shape[0] < 2:
        raise ValueError("VICReg loss requires at least two samples.")
    if variance_target <= 0 or eps <= 0:
        raise ValueError("variance_target and eps must be positive.")

    projection_a = projection_a.float()
    projection_b = projection_b.float()
    invariance = F.mse_loss(projection_a, projection_b)
    variance = 0.5 * (
        F.relu(variance_target - torch.sqrt(projection_a.var(dim=0, unbiased=False) + eps)).mean()
        + F.relu(variance_target - torch.sqrt(projection_b.var(dim=0, unbiased=False) + eps)).mean()
    )
    covariance = 0.5 * (
        _off_diagonal_covariance_loss(projection_a)
        + _off_diagonal_covariance_loss(projection_b)
    )
    loss = (
        invariance_weight * invariance
        + variance_weight * variance
        + covariance_weight * covariance
    )
    return {
        "loss": loss,
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
    }


def _off_diagonal_covariance_loss(projection: torch.Tensor) -> torch.Tensor:
    batch_size, feature_dim = projection.shape
    centered = projection - projection.mean(dim=0)
    covariance = centered.T @ centered / max(batch_size - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return off_diagonal.pow(2).sum() / feature_dim


def _barlow_twins_loss(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    *,
    offdiag_weight: float = 0.005,
    eps: float = 1e-9,
) -> dict[str, torch.Tensor]:
    if projection_a.shape != projection_b.shape:
        raise ValueError(
            f"Projection shapes must match, got {tuple(projection_a.shape)} and {tuple(projection_b.shape)}."
        )
    if projection_a.ndim != 2:
        raise ValueError(f"Expected projection tensors shaped (batch, features), got {tuple(projection_a.shape)}.")
    if projection_a.shape[0] < 2:
        raise ValueError("Barlow Twins loss requires at least two samples.")
    if offdiag_weight < 0:
        raise ValueError("offdiag_weight must be non-negative.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    projection_a = _standardize_projection(projection_a.float(), eps)
    projection_b = _standardize_projection(projection_b.float(), eps)
    cross_correlation = projection_a.T @ projection_b / projection_a.shape[0]
    on_diag = torch.diagonal(cross_correlation).add(-1).pow(2).sum()
    off_diag = (cross_correlation - torch.diag(torch.diagonal(cross_correlation))).pow(2).sum()
    loss = on_diag + offdiag_weight * off_diag
    return {"loss": loss, "on_diag": on_diag, "off_diag": off_diag}


def _standardize_projection(projection: torch.Tensor, eps: float) -> torch.Tensor:
    return (projection - projection.mean(dim=0)) / (projection.std(dim=0, unbiased=False) + eps)


def _byol_alignment_components(
    model: _MultitaskMaskedContrastiveModel,
    target_model: _MultitaskMaskedContrastiveModel | None,
    view_a: dict[str, torch.Tensor],
    view_b: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if target_model is None:
        raise ValueError("BYOL alignment requires a target model.")
    prediction_a = model.predict(model.project(view_a))
    prediction_b = model.predict(model.project(view_b))
    with torch.no_grad():
        target_a = target_model.project(view_a)
        target_b = target_model.project(view_b)
    components = _byol_loss(prediction_a, target_b, prediction_b, target_a)
    zero = torch.zeros((), device=prediction_a.device)
    return {
        "loss": components["loss"],
        "invariance": zero,
        "variance": zero,
        "covariance": zero,
        "on_diag": zero,
        "off_diag": zero,
        "byol": components["loss"],
    }


def _byol_loss(
    prediction_a: torch.Tensor,
    target_b: torch.Tensor,
    prediction_b: torch.Tensor,
    target_a: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if prediction_a.shape != target_b.shape or prediction_b.shape != target_a.shape:
        raise ValueError("BYOL prediction and target shapes must match.")
    loss = 0.5 * (
        _negative_cosine_similarity(prediction_a, target_b.detach())
        + _negative_cosine_similarity(prediction_b, target_a.detach())
    )
    return {"loss": loss}


def _negative_cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction.float(), dim=1)
    target = F.normalize(target.float(), dim=1)
    return 2 - 2 * (prediction * target).sum(dim=1).mean()


def _build_byol_target_model(
    model: _MultitaskMaskedContrastiveModel,
) -> _MultitaskMaskedContrastiveModel:
    target_model = copy.deepcopy(model)
    target_model.eval()
    for parameter in target_model.parameters():
        parameter.requires_grad = False
    return target_model


@torch.no_grad()
def _update_byol_target_model(
    target_model: _MultitaskMaskedContrastiveModel,
    online_model: _MultitaskMaskedContrastiveModel,
    momentum: float,
) -> None:
    for target_parameter, online_parameter in zip(target_model.parameters(), online_model.parameters(), strict=True):
        target_parameter.data.mul_(momentum).add_(online_parameter.data, alpha=1.0 - momentum)
    for target_buffer, online_buffer in zip(target_model.buffers(), online_model.buffers(), strict=True):
        target_buffer.data.copy_(online_buffer.data)


def _multitask_psd_reconstruction_loss(
    model: _MultitaskMaskedContrastiveModel,
    batch_pairs: list[tuple[np.ndarray, np.ndarray]],
    config: MaskedSSLConfig,
    device: torch.device,
    epoch: int,
    start: int,
) -> torch.Tensor:
    batch_arrays = [state for pair in batch_pairs for state in pair]
    masked_views, masks = zip(
        *[
            build_masked_psd_view(
                array,
                seed=config.seed + epoch * 100_000 + start + index,
                channel_mask_prob=config.psd_channel_mask_prob,
                frequency_mask_prob=config.psd_frequency_mask_prob,
                element_mask_prob=config.psd_element_mask_prob,
            )
            for index, array in enumerate(batch_arrays)
        ],
        strict=True,
    )
    target = torch.as_tensor(np.stack(batch_arrays), device=device)
    masked = torch.as_tensor(np.stack(masked_views), device=device)
    mask = torch.as_tensor(np.stack(masks), device=device)
    return _masked_mse(model.reconstruct_psd(masked), target, mask)


def _multitask_fc_reconstruction_loss(
    model: _MultitaskMaskedContrastiveModel,
    batch_pairs: list[tuple[np.ndarray, np.ndarray]],
    edge_index: np.ndarray,
    config: MaskedSSLConfig,
    device: torch.device,
    epoch: int,
    start: int,
) -> torch.Tensor:
    batch_arrays = [state for pair in batch_pairs for state in pair]
    masked_views, masks = zip(
        *[
            build_masked_fc_view(
                array,
                edge_index=edge_index,
                seed=config.seed + epoch * 100_000 + start + index,
                node_mask_prob=config.fc_node_mask_prob,
                edge_mask_prob=config.fc_edge_mask_prob,
                band_mask_prob=config.fc_band_mask_prob,
            )
            for index, array in enumerate(batch_arrays)
        ],
        strict=True,
    )
    target = torch.as_tensor(np.stack(batch_arrays), device=device)
    masked = torch.as_tensor(np.stack(masked_views), device=device)
    mask = torch.as_tensor(np.stack(masks), device=device)
    return _masked_mse(model.reconstruct_wpli(masked), target, mask)


def _multitask_consistency_loss(
    model: _MultitaskMaskedContrastiveModel,
    psd_pairs: list[tuple[np.ndarray, np.ndarray]],
    wpli_pairs: list[tuple[np.ndarray, np.ndarray]],
    config: MaskedSSLConfig,
    device: torch.device,
) -> torch.Tensor:
    if config.eo_ec_consistency_weight == 0:
        return torch.zeros((), device=device)
    return 0.5 * (
        model.psd_consistency_loss(psd_pairs, device)
        + model.wpli_consistency_loss(wpli_pairs, device)
    )


class _PSDMaskedAutoencoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = PSDConv2DEncoder(embedding_dim=embedding_dim, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Conv2d(16, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decoder(self.encoder.forward_features(features))
        return reconstruction.squeeze(1)


class _FCMaskedAutoencoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = FCConv1DEncoder(embedding_dim=embedding_dim, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Conv1d(16, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, FC_SHAPE[1], kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decoder(self.encoder.forward_features(features))
        return reconstruction.transpose(1, 2)


def _pretrain_psd_masked_autoencoder(
    pair_states: list[tuple[np.ndarray, np.ndarray]],
    config: MaskedSSLConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    arrays = [state for pair in pair_states for state in pair]
    _validate_arrays(arrays, PSD_SHAPE, "PSD")
    model = _PSDMaskedAutoencoder(config.embedding_dim, config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []
    for epoch in range(1, config.psd_epochs + 1):
        order = rng.permutation(len(pair_states))
        losses: list[float] = []
        reconstruction_losses: list[float] = []
        consistency_losses: list[float] = []
        model.train()
        for start in range(0, len(order), config.batch_size):
            batch_pairs = [pair_states[int(index)] for index in order[start : start + config.batch_size]]
            batch_arrays = [state for pair in batch_pairs for state in pair]
            masked_views, masks = zip(
                *[
                    build_masked_psd_view(
                        array,
                        seed=config.seed + epoch * 100_000 + start + index,
                        channel_mask_prob=config.psd_channel_mask_prob,
                        frequency_mask_prob=config.psd_frequency_mask_prob,
                        element_mask_prob=config.psd_element_mask_prob,
                    )
                    for index, array in enumerate(batch_arrays)
                ],
                strict=True,
            )
            target = torch.as_tensor(np.stack(batch_arrays), device=device)
            masked = torch.as_tensor(np.stack(masked_views), device=device)
            mask = torch.as_tensor(np.stack(masks), device=device)
            optimizer.zero_grad()
            reconstruction_loss = _masked_mse(model(masked), target, mask)
            consistency_loss = _pair_consistency_loss(model.encoder, batch_pairs, device)
            loss = reconstruction_loss + config.eo_ec_consistency_weight * consistency_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            reconstruction_losses.append(float(reconstruction_loss.detach().cpu().item()))
            consistency_losses.append(float(consistency_loss.detach().cpu().item()))
        rows.append(
            _history_row(
                "psd_masked",
                epoch,
                losses,
                reconstruction_losses,
                consistency_losses,
                len(arrays),
                config,
            )
        )
    return _encoder_state(model.encoder), rows


def _pretrain_fc_masked_autoencoder(
    pair_states: list[tuple[np.ndarray, np.ndarray]],
    config: MaskedSSLConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    arrays = [state for pair in pair_states for state in pair]
    _validate_arrays(arrays, FC_SHAPE, "wPLI")
    edge_index = build_fc_edge_index()
    model = _FCMaskedAutoencoder(config.embedding_dim, config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    rng = np.random.default_rng(config.seed + 31)
    rows: list[dict[str, object]] = []
    for epoch in range(1, config.fc_epochs + 1):
        order = rng.permutation(len(pair_states))
        losses: list[float] = []
        reconstruction_losses: list[float] = []
        consistency_losses: list[float] = []
        model.train()
        for start in range(0, len(order), config.batch_size):
            batch_pairs = [pair_states[int(index)] for index in order[start : start + config.batch_size]]
            batch_arrays = [state for pair in batch_pairs for state in pair]
            masked_views, masks = zip(
                *[
                    build_masked_fc_view(
                        array,
                        edge_index=edge_index,
                        seed=config.seed + epoch * 100_000 + start + index,
                        node_mask_prob=config.fc_node_mask_prob,
                        edge_mask_prob=config.fc_edge_mask_prob,
                        band_mask_prob=config.fc_band_mask_prob,
                    )
                    for index, array in enumerate(batch_arrays)
                ],
                strict=True,
            )
            target = torch.as_tensor(np.stack(batch_arrays), device=device)
            masked = torch.as_tensor(np.stack(masked_views), device=device)
            mask = torch.as_tensor(np.stack(masks), device=device)
            optimizer.zero_grad()
            reconstruction_loss = _masked_mse(model(masked), target, mask)
            consistency_loss = _pair_consistency_loss(model.encoder, batch_pairs, device)
            loss = reconstruction_loss + config.eo_ec_consistency_weight * consistency_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            reconstruction_losses.append(float(reconstruction_loss.detach().cpu().item()))
            consistency_losses.append(float(consistency_loss.detach().cpu().item()))
        rows.append(
            _history_row(
                "wpli_masked",
                epoch,
                losses,
                reconstruction_losses,
                consistency_losses,
                len(arrays),
                config,
            )
        )
    return _encoder_state(model.encoder), rows


def _pair_consistency_loss(
    encoder: nn.Module,
    pair_states: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> torch.Tensor:
    eo = torch.as_tensor(np.stack([pair[0] for pair in pair_states]), device=device)
    ec = torch.as_tensor(np.stack([pair[1] for pair in pair_states]), device=device)
    return _embedding_consistency_loss(encoder(eo), encoder(ec))


def _embedding_consistency_loss(eo_embedding: torch.Tensor, ec_embedding: torch.Tensor) -> torch.Tensor:
    eo_normalized = F.normalize(eo_embedding.float(), dim=1)
    ec_normalized = F.normalize(ec_embedding.float(), dim=1)
    return F.mse_loss(eo_normalized, ec_normalized)


def _masked_mse(reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=reconstruction.dtype)
    squared = (reconstruction - target).pow(2) * mask_float
    return squared.sum() / mask_float.sum().clamp_min(1.0)


def _scaled_branch_pair_states(
    pairs: list[FeatureSSLPairRecord],
    scaler: dict[str, tuple[np.ndarray, np.ndarray]],
    branch: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if branch not in scaler:
        raise ValueError(f"Masked SSL requires branch {branch!r}.")
    mean, std = scaler[branch]
    pair_states: list[tuple[np.ndarray, np.ndarray]] = []
    for pair in pairs:
        states = pair.modalities.get(branch)
        if states is None:
            raise ValueError(f"Masked SSL requires branch {branch!r} for {pair.subject_key} {pair.stage}.")
        pair_states.append(
            (
                ((states[0] - mean) / std).astype(np.float32),
                ((states[1] - mean) / std).astype(np.float32),
            )
        )
    return pair_states


def _pair_to_supervised_record(pair: FeatureSSLPairRecord) -> SupervisedFeatureRecord:
    psd_states = pair.modalities.get("psd")
    if psd_states is None:
        raise ValueError(f"Masked SSL requires PSD branch for {pair.subject_key} {pair.stage}.")
    return SupervisedFeatureRecord(
        subject_id=pair.subject_id,
        label=0,
        eo=psd_states[0],
        ec=psd_states[1],
        modalities=pair.modalities,
    )


def _validate_arrays(arrays: list[np.ndarray], shape: tuple[int, ...], name: str) -> None:
    if not arrays:
        raise ValueError(f"{name} masked SSL requires at least one feature array.")
    if any(np.asarray(array).shape != shape for array in arrays):
        raise ValueError(f"All {name} arrays must have shape {shape}.")


def _encoder_state(encoder: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}


def _history_row(
    branch: str,
    epoch: int,
    losses: list[float],
    reconstruction_losses: list[float],
    consistency_losses: list[float],
    n_samples: int,
    config: MaskedSSLConfig,
) -> dict[str, object]:
    objective = (
        "masked_local_reconstruction_with_eo_ec_consistency"
        if config.eo_ec_consistency_weight > 0
        else "masked_local_reconstruction"
    )
    return {
        "branch": branch,
        "epoch": epoch,
        "loss": float(np.mean(losses)),
        "reconstruction_loss": float(np.mean(reconstruction_losses)),
        "consistency_loss": float(np.mean(consistency_losses)),
        "objective": objective,
        "n_samples": n_samples,
        "batch_size": config.batch_size,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "learning_rate": config.lr,
        "eo_ec_consistency_weight": config.eo_ec_consistency_weight,
        "psd_channel_mask_prob": config.psd_channel_mask_prob,
        "psd_frequency_mask_prob": config.psd_frequency_mask_prob,
        "psd_element_mask_prob": config.psd_element_mask_prob,
        "fc_node_mask_prob": config.fc_node_mask_prob,
        "fc_edge_mask_prob": config.fc_edge_mask_prob,
        "fc_band_mask_prob": config.fc_band_mask_prob,
        "contrastive_weight": config.contrastive_weight,
        "contrastive_temperature": config.contrastive_temperature,
        "projection_dim": config.projection_dim,
        "contrastive_noise_std": config.contrastive_noise_std,
        "contrastive_feature_mask_prob": config.contrastive_feature_mask_prob,
        "alignment_method": config.alignment_method,
        "vicreg_invariance_weight": config.vicreg_invariance_weight,
        "vicreg_variance_weight": config.vicreg_variance_weight,
        "vicreg_covariance_weight": config.vicreg_covariance_weight,
        "vicreg_variance_target": config.vicreg_variance_target,
        "barlow_offdiag_weight": config.barlow_offdiag_weight,
        "byol_momentum": config.byol_momentum,
        "byol_predictor_hidden_dim": config.byol_predictor_hidden_dim,
    }


def _multitask_history_row(
    epoch: int,
    losses: list[float],
    psd_reconstruction_losses: list[float],
    fc_reconstruction_losses: list[float],
    alignment_losses: list[float],
    vicreg_invariance_losses: list[float],
    vicreg_variance_losses: list[float],
    vicreg_covariance_losses: list[float],
    barlow_on_diag_losses: list[float],
    barlow_off_diag_losses: list[float],
    byol_losses: list[float],
    consistency_losses: list[float],
    n_pairs: int,
    config: MaskedSSLConfig,
) -> dict[str, object]:
    objective = (
        "masked_local_reconstruction_with_vicreg_alignment"
        if config.alignment_method == "vicreg"
        else "masked_local_reconstruction_with_barlow_alignment"
        if config.alignment_method == "barlow"
        else "masked_local_reconstruction_with_byol_alignment"
        if config.alignment_method == "byol"
        else "masked_local_reconstruction_with_feature_contrastive"
    )
    if config.eo_ec_consistency_weight > 0:
        objective = f"{objective}_and_eo_ec_consistency"
    alignment_loss = float(np.mean(alignment_losses))
    return {
        "branch": "psd_wpli_multitask",
        "epoch": epoch,
        "loss": float(np.mean(losses)),
        "reconstruction_loss": float(
            np.mean(
                [
                    psd_value + fc_value
                    for psd_value, fc_value in zip(psd_reconstruction_losses, fc_reconstruction_losses, strict=True)
                ]
            )
        ),
        "psd_reconstruction_loss": float(np.mean(psd_reconstruction_losses)),
        "fc_reconstruction_loss": float(np.mean(fc_reconstruction_losses)),
        "contrastive_loss": alignment_loss,
        "alignment_loss": alignment_loss,
        "alignment_method": config.alignment_method,
        "vicreg_invariance_loss": float(np.mean(vicreg_invariance_losses)),
        "vicreg_variance_loss": float(np.mean(vicreg_variance_losses)),
        "vicreg_covariance_loss": float(np.mean(vicreg_covariance_losses)),
        "barlow_on_diag_loss": float(np.mean(barlow_on_diag_losses)),
        "barlow_off_diag_loss": float(np.mean(barlow_off_diag_losses)),
        "byol_loss": float(np.mean(byol_losses)),
        "consistency_loss": float(np.mean(consistency_losses)),
        "objective": objective,
        "n_pairs": n_pairs,
        "n_samples": n_pairs * 2,
        "batch_size": config.batch_size,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "projection_dim": config.projection_dim,
        "learning_rate": config.lr,
        "eo_ec_consistency_weight": config.eo_ec_consistency_weight,
        "contrastive_weight": config.contrastive_weight,
        "contrastive_temperature": config.contrastive_temperature,
        "contrastive_noise_std": config.contrastive_noise_std,
        "contrastive_feature_mask_prob": config.contrastive_feature_mask_prob,
        "vicreg_invariance_weight": config.vicreg_invariance_weight,
        "vicreg_variance_weight": config.vicreg_variance_weight,
        "vicreg_covariance_weight": config.vicreg_covariance_weight,
        "vicreg_variance_target": config.vicreg_variance_target,
        "barlow_offdiag_weight": config.barlow_offdiag_weight,
        "byol_momentum": config.byol_momentum,
        "byol_predictor_hidden_dim": config.byol_predictor_hidden_dim,
        "psd_channel_mask_prob": config.psd_channel_mask_prob,
        "psd_frequency_mask_prob": config.psd_frequency_mask_prob,
        "psd_element_mask_prob": config.psd_element_mask_prob,
        "fc_node_mask_prob": config.fc_node_mask_prob,
        "fc_edge_mask_prob": config.fc_edge_mask_prob,
        "fc_band_mask_prob": config.fc_band_mask_prob,
    }


def _ensure_nonempty_mask(mask: np.ndarray, rng: np.random.Generator) -> None:
    if mask.any():
        return
    flat_index = int(rng.integers(0, mask.size))
    mask.reshape(-1)[flat_index] = True


def _validate_masked_ssl_config(config: MaskedSSLConfig) -> None:
    if config.psd_epochs < 1 or config.fc_epochs < 1:
        raise ValueError("psd_epochs and fc_epochs must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if config.embedding_dim < 1 or config.hidden_dim < 1:
        raise ValueError("embedding_dim and hidden_dim must be positive.")
    if config.lr <= 0:
        raise ValueError("lr must be positive.")
    if config.dropout < 0:
        raise ValueError("dropout must be non-negative.")
    if config.eo_ec_consistency_weight < 0:
        raise ValueError("eo_ec_consistency_weight must be non-negative.")
    if config.contrastive_weight < 0:
        raise ValueError("contrastive_weight must be non-negative.")
    if config.alignment_method not in ALIGNMENT_METHODS:
        raise ValueError("alignment_method must be 'ntxent', 'vicreg', 'barlow', or 'byol'.")
    if config.contrastive_temperature <= 0:
        raise ValueError("contrastive_temperature must be positive.")
    if config.projection_dim < 1:
        raise ValueError("projection_dim must be positive.")
    if config.contrastive_noise_std < 0:
        raise ValueError("contrastive_noise_std must be non-negative.")
    if config.vicreg_invariance_weight < 0 or config.vicreg_variance_weight < 0 or config.vicreg_covariance_weight < 0:
        raise ValueError("VICReg loss weights must be non-negative.")
    if config.vicreg_variance_target <= 0:
        raise ValueError("vicreg_variance_target must be positive.")
    if config.vicreg_eps <= 0:
        raise ValueError("vicreg_eps must be positive.")
    if config.barlow_offdiag_weight < 0:
        raise ValueError("barlow_offdiag_weight must be non-negative.")
    if config.barlow_eps <= 0:
        raise ValueError("barlow_eps must be positive.")
    if not 0 <= config.byol_momentum < 1:
        raise ValueError("byol_momentum must be in [0, 1).")
    if config.byol_predictor_hidden_dim < 1:
        raise ValueError("byol_predictor_hidden_dim must be positive.")
    _validate_probability("contrastive_feature_mask_prob", config.contrastive_feature_mask_prob)
    for name, value in (
        ("psd_channel_mask_prob", config.psd_channel_mask_prob),
        ("psd_frequency_mask_prob", config.psd_frequency_mask_prob),
        ("psd_element_mask_prob", config.psd_element_mask_prob),
        ("fc_node_mask_prob", config.fc_node_mask_prob),
        ("fc_edge_mask_prob", config.fc_edge_mask_prob),
        ("fc_band_mask_prob", config.fc_band_mask_prob),
    ):
        _validate_probability(name, value)


def _validate_probability(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")
