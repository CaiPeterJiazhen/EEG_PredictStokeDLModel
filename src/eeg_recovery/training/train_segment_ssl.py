from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from eeg_recovery.models.encoders import FC_SHAPE, PSD_SHAPE, SharedFCEncoder, SharedPSDEncoder
from eeg_recovery.models.multimodal_model import branches_for_feature_kind
from eeg_recovery.training.segment_ssl_dataset import SegmentSSLRecord, SegmentSSLScaler
from eeg_recovery.training.train_structured_ssl import build_fc_edge_index
from eeg_recovery.training.train_supervised import _plot_loss_history, resolve_device


SEGMENT_SSL_OBJECTIVES = {"barlow", "vicreg"}


@dataclass(frozen=True)
class SegmentSSLTrainingConfig:
    objective: str = "barlow"
    feature_kind: str = "psd"
    epochs: int = 20
    batch_size: int = 16
    embedding_dim: int = 32
    projection_dim: int = 32
    dropout: float = 0.0
    lr: float = 1e-3
    feature_mask_prob: float = 0.03
    noise_std: float = 0.02
    amplitude_scale_low: float = 0.9
    amplitude_scale_high: float = 1.1
    psd_channel_mask_prob: float = 0.05
    psd_frequency_mask_prob: float = 0.05
    fc_node_mask_prob: float = 0.02
    fc_edge_mask_prob: float = 0.05
    fc_band_mask_prob: float = 0.02
    vicreg_invariance_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    vicreg_variance_target: float = 1.0
    vicreg_eps: float = 1e-4
    barlow_offdiag_weight: float = 0.005
    barlow_eps: float = 1e-9
    lambda_latent: float = 1.0
    lambda_local: float = 0.1
    masked_latent_loss: str = "cosine"
    device: str = "auto"
    seed: int = 42


class _SegmentSSLModel(nn.Module):
    def __init__(
        self,
        branches: tuple[str, ...],
        *,
        embedding_dim: int,
        projection_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.branches = branches
        self.branch_encoders = nn.ModuleDict(
            {
                branch: (
                    SharedPSDEncoder(embedding_dim=embedding_dim, dropout=dropout, encoder_kind="cnn")
                    if branch == "psd"
                    else SharedFCEncoder(embedding_dim=embedding_dim, dropout=dropout, encoder_kind="cnn")
                )
                for branch in branches
            }
        )
        total_embedding_dim = embedding_dim * len(branches)
        self.projector = nn.Linear(total_embedding_dim, projection_dim)
        self.latent_predictor = nn.Sequential(
            nn.Linear(total_embedding_dim, max(total_embedding_dim, projection_dim)),
            nn.ReLU(),
            nn.Linear(max(total_embedding_dim, projection_dim), total_embedding_dim),
        )
        self.local_decoders = nn.ModuleDict(
            {
                branch: nn.Linear(embedding_dim, int(np.prod(PSD_SHAPE if branch == "psd" else FC_SHAPE)))
                for branch in branches
            }
        )

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        embeddings = [self.branch_encoders[branch](batch[branch]) for branch in self.branches]
        return torch.cat(embeddings, dim=1)

    def project(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.projector(self.encode(batch))

    def predict_latent(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.latent_predictor(self.encode(batch))

    def reconstruct_local(self, batch: Mapping[str, torch.Tensor], branch: str) -> torch.Tensor:
        embedding = self.branch_encoders[branch](batch[branch])
        flat = self.local_decoders[branch](embedding)
        shape = PSD_SHAPE if branch == "psd" else FC_SHAPE
        return flat.reshape(batch[branch].shape[0], *shape)

    def pretrained_state(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for branch in self.branches:
            for key, value in self.branch_encoders[branch].state_dict().items():
                state[f"branch_models.{branch}.encoder.{key}"] = value.detach().cpu().clone()
        return state


def augment_segment_batch(
    batch: Mapping[str, torch.Tensor],
    *,
    seed: int,
    psd_channel_mask_prob: float = 0.05,
    psd_frequency_mask_prob: float = 0.05,
    feature_mask_prob: float = 0.03,
    fc_node_mask_prob: float = 0.02,
    fc_edge_mask_prob: float = 0.05,
    fc_band_mask_prob: float = 0.02,
    noise_std: float = 0.02,
    amplitude_scale_range: tuple[float, float] = (0.9, 1.1),
) -> dict[str, torch.Tensor]:
    """Create one physiologically conservative augmented view for segment features."""

    augmented: dict[str, torch.Tensor] = {}
    for branch, value in batch.items():
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed + _stable_key_offset(branch))
        view = value.float().clone()
        scale_low, scale_high = amplitude_scale_range
        if scale_low <= 0 or scale_high <= 0 or scale_low > scale_high:
            raise ValueError("amplitude_scale_range must contain positive ordered values.")
        scales = torch.empty((view.shape[0],) + (1,) * (view.ndim - 1), device=view.device)
        scales.uniform_(scale_low, scale_high, generator=generator)
        view = view * scales
        if noise_std > 0:
            view = view + torch.randn(view.shape, generator=generator, device=view.device) * noise_std
        if branch == "psd":
            view = _augment_psd_tensor(
                view,
                generator=generator,
                channel_mask_prob=psd_channel_mask_prob,
                frequency_mask_prob=psd_frequency_mask_prob,
                element_mask_prob=feature_mask_prob,
            )
        else:
            view = _augment_fc_tensor(
                view,
                generator=generator,
                node_mask_prob=fc_node_mask_prob,
                edge_mask_prob=fc_edge_mask_prob,
                band_mask_prob=fc_band_mask_prob,
                element_mask_prob=feature_mask_prob,
            )
        augmented[branch] = view
    return augmented


def barlow_twins_loss(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    *,
    offdiag_weight: float = 0.005,
    eps: float = 1e-9,
) -> dict[str, torch.Tensor]:
    if projection_a.shape != projection_b.shape:
        raise ValueError("Projection shapes must match for Barlow Twins.")
    if projection_a.ndim != 2:
        raise ValueError("Projection tensors must be two-dimensional.")
    if projection_a.shape[0] < 2:
        zero = torch.zeros((), device=projection_a.device)
        return {"loss": zero, "on_diag": zero, "off_diag": zero}
    z_a = _standardize(projection_a.float(), eps)
    z_b = _standardize(projection_b.float(), eps)
    cross_correlation = z_a.T @ z_b / z_a.shape[0]
    on_diag = torch.diagonal(cross_correlation).add(-1.0).pow(2).sum()
    off_diag = (cross_correlation - torch.diag(torch.diagonal(cross_correlation))).pow(2).sum()
    return {"loss": on_diag + offdiag_weight * off_diag, "on_diag": on_diag, "off_diag": off_diag}


def vicreg_loss(
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
        raise ValueError("Projection shapes must match for VICReg.")
    if projection_a.ndim != 2:
        raise ValueError("Projection tensors must be two-dimensional.")
    if projection_a.shape[0] < 2:
        zero = torch.zeros((), device=projection_a.device)
        return {"loss": zero, "invariance": zero, "variance": zero, "covariance": zero}
    invariance = F.mse_loss(projection_a.float(), projection_b.float())
    variance = 0.5 * (
        F.relu(variance_target - torch.sqrt(projection_a.float().var(dim=0, unbiased=False) + eps)).mean()
        + F.relu(variance_target - torch.sqrt(projection_b.float().var(dim=0, unbiased=False) + eps)).mean()
    )
    covariance = 0.5 * (_off_diagonal_covariance_loss(projection_a.float()) + _off_diagonal_covariance_loss(projection_b.float()))
    loss = invariance_weight * invariance + variance_weight * variance + covariance_weight * covariance
    return {"loss": loss, "invariance": invariance, "variance": variance, "covariance": covariance}


def masked_latent_prediction_loss(
    *,
    encoder: nn.Module,
    predictor: nn.Module,
    clean: torch.Tensor,
    masked: torch.Tensor,
    loss_type: str = "cosine",
) -> torch.Tensor:
    """Predict the clean stop-gradient latent from a masked view."""

    with torch.no_grad():
        target = encoder(clean).detach()
    prediction = predictor(encoder(masked))
    if loss_type == "cosine":
        prediction = F.normalize(prediction.float(), dim=1)
        target = F.normalize(target.float(), dim=1)
        return 2.0 - 2.0 * (prediction * target).sum(dim=1).mean()
    if loss_type == "mse":
        return F.mse_loss(prediction.float(), target.float())
    raise ValueError("loss_type must be 'cosine' or 'mse'.")


def run_segment_ssl_pretraining(
    records: Iterable[SegmentSSLRecord],
    config: SegmentSSLTrainingConfig,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    resolved = _validate_config(config)
    records = list(records)
    if not records:
        raise ValueError("Segment SSL pretraining requires at least one segment record.")
    branches = tuple(branches_for_feature_kind(resolved.feature_kind))
    if any(branch not in {"psd", "wpli", "icoh"} for branch in branches):
        raise ValueError(f"Unsupported segment SSL feature_kind: {resolved.feature_kind}.")
    for record in records:
        for branch in branches:
            if branch not in record.features:
                raise ValueError(f"Record {record.subject_key} segment {record.segment_index} is missing branch {branch!r}.")

    _seed_everything(resolved.seed)
    device = resolve_device(resolved.device)
    scaler = SegmentSSLScaler.fit(records, branches=branches)
    model = _SegmentSSLModel(
        branches,
        embedding_dim=resolved.embedding_dim,
        projection_dim=resolved.projection_dim,
        dropout=resolved.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved.lr)
    rng = np.random.default_rng(resolved.seed)
    rows: list[dict[str, object]] = []

    for epoch in range(1, resolved.epochs + 1):
        order = rng.permutation(len(records))
        epoch_rows: list[dict[str, float]] = []
        model.train()
        for start in range(0, len(order), resolved.batch_size):
            batch_records = [records[int(index)] for index in order[start : start + resolved.batch_size]]
            clean_batch = _make_segment_batch(batch_records, branches=branches, scaler=scaler, device=device)
            view_a = augment_segment_batch(
                clean_batch,
                seed=resolved.seed + epoch * 100_000 + start,
                psd_channel_mask_prob=resolved.psd_channel_mask_prob,
                psd_frequency_mask_prob=resolved.psd_frequency_mask_prob,
                feature_mask_prob=resolved.feature_mask_prob,
                fc_node_mask_prob=resolved.fc_node_mask_prob,
                fc_edge_mask_prob=resolved.fc_edge_mask_prob,
                fc_band_mask_prob=resolved.fc_band_mask_prob,
                noise_std=resolved.noise_std,
                amplitude_scale_range=(resolved.amplitude_scale_low, resolved.amplitude_scale_high),
            )
            view_b = augment_segment_batch(
                clean_batch,
                seed=resolved.seed + epoch * 100_000 + start + 1,
                psd_channel_mask_prob=resolved.psd_channel_mask_prob,
                psd_frequency_mask_prob=resolved.psd_frequency_mask_prob,
                feature_mask_prob=resolved.feature_mask_prob,
                fc_node_mask_prob=resolved.fc_node_mask_prob,
                fc_edge_mask_prob=resolved.fc_edge_mask_prob,
                fc_band_mask_prob=resolved.fc_band_mask_prob,
                noise_std=resolved.noise_std,
                amplitude_scale_range=(resolved.amplitude_scale_low, resolved.amplitude_scale_high),
            )
            masked_for_latent = _masked_latent_view(clean_batch, resolved, seed=resolved.seed + epoch * 100_000 + start + 11)

            optimizer.zero_grad()
            projection_a = model.project(view_a)
            projection_b = model.project(view_b)
            alignment = _alignment_components(projection_a, projection_b, resolved)
            masked_latent = (
                _model_masked_latent_loss(model, clean_batch, masked_for_latent, resolved)
                if resolved.lambda_latent > 0
                else torch.zeros((), device=device)
            )
            local = (
                _local_reconstruction_loss(model, clean_batch, masked_for_latent, branches)
                if resolved.lambda_local > 0
                else torch.zeros((), device=device)
            )
            loss = alignment["loss"] + resolved.lambda_latent * masked_latent + resolved.lambda_local * local
            loss.backward()
            optimizer.step()

            epoch_rows.append(
                {
                    "loss": float(loss.detach().cpu().item()),
                    "alignment_loss": float(alignment["loss"].detach().cpu().item()),
                    "masked_latent_loss": float(masked_latent.detach().cpu().item()),
                    "local_reconstruction_loss": float(local.detach().cpu().item()),
                    "barlow_on_diag_loss": float(alignment["on_diag"].detach().cpu().item()),
                    "barlow_off_diag_loss": float(alignment["off_diag"].detach().cpu().item()),
                    "vicreg_invariance_loss": float(alignment["invariance"].detach().cpu().item()),
                    "vicreg_variance_loss": float(alignment["variance"].detach().cpu().item()),
                    "vicreg_covariance_loss": float(alignment["covariance"].detach().cpu().item()),
                }
            )

        rows.append(
            {
                "epoch": epoch,
                "objective": resolved.objective,
                "feature_kind": resolved.feature_kind,
                "n_segments": len(records),
                "batch_size": resolved.batch_size,
                "embedding_dim": resolved.embedding_dim,
                "projection_dim": resolved.projection_dim,
                "learning_rate": resolved.lr,
                "feature_mask_prob": resolved.feature_mask_prob,
                "noise_std": resolved.noise_std,
                "lambda_latent": resolved.lambda_latent,
                "lambda_local": resolved.lambda_local,
                **{
                    key: float(np.mean([row[key] for row in epoch_rows]))
                    for key in epoch_rows[0]
                },
            }
        )

    return model.pretrained_state(), pd.DataFrame(rows)


def extract_branch_encoder_state(
    pretrained_state: Mapping[str, torch.Tensor],
    branch: str,
) -> dict[str, torch.Tensor]:
    if branch not in {"psd", "wpli", "icoh"}:
        raise ValueError("branch must be 'psd', 'wpli', or 'icoh'.")
    prefix = f"branch_models.{branch}.encoder."
    extracted = {
        key.removeprefix(prefix): value.detach().cpu().clone()
        for key, value in pretrained_state.items()
        if key.startswith(prefix)
    }
    if not extracted:
        raise ValueError(f"Pretrained state does not contain encoder weights for branch {branch!r}.")
    return extracted


def segment_records_manifest_hash(records: Iterable[SegmentSSLRecord]) -> str:
    manifest_rows = [
        {
            "group": record.group,
            "subject_id": record.subject_id,
            "subject_key": record.subject_key,
            "stage": record.stage,
            "state": record.state,
            "segment_index": record.segment_index,
            "branches": sorted(record.features),
            "source_path": str(record.source_path),
        }
        for record in records
    ]
    payload = json.dumps(
        sorted(
            manifest_rows,
            key=lambda row: (
                row["group"],
                row["subject_key"],
                row["stage"],
                row["state"],
                row["segment_index"],
                row["source_path"],
            ),
        ),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def segment_ssl_transfer_run_name(
    *,
    objective: str,
    feature_kind: str,
    data_scope: str,
    transfer_mode: str,
    seed: str | int,
    pretrain_epochs: int,
    projection_dim: int,
    feature_mask_prob: float,
    noise_std: float,
    lambda_latent: float,
    supervised_epochs: int,
) -> str:
    return (
        f"segssl_{objective}_{data_scope}_{feature_kind}_{transfer_mode}"
        f"_seed{seed}_pre{pretrain_epochs}_proj{projection_dim}"
        f"_mask{_number_token(feature_mask_prob)}_noise{_number_token(noise_std)}"
        f"_mlp{_number_token(lambda_latent)}_sup{supervised_epochs}"
    )


def write_segment_ssl_transfer_outputs(
    *,
    output_root: str | Path,
    run_name: str,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    ssl_history: pd.DataFrame,
    supervised_loss_history: pd.DataFrame,
) -> dict[str, Path]:
    root = Path(output_root)
    safe = _safe_run_name(run_name)
    paths = {
        "predictions": root / "results" / "predictions" / f"dl_loso_predictions_{safe}.csv",
        "metrics": root / "results" / "metrics" / f"dl_model_comparison_{safe}.csv",
        "ssl_history": root / "results" / "ssl" / f"segment_ssl_history_{safe}.csv",
        "loss_history": root / "results" / "training_logs" / f"dl_loss_history_{safe}.csv",
        "loss_curve": root / "results" / "figures" / f"dl_loss_curve_{safe}.png",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    ssl_history.to_csv(paths["ssl_history"], index=False)
    supervised_loss_history.to_csv(paths["loss_history"], index=False)
    _plot_loss_history(supervised_loss_history, paths["loss_curve"])
    return paths


def summarize_seed_metrics(metrics_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame.copy() for frame in metrics_frames]
    if not frames:
        raise ValueError("At least one metrics frame is required.")
    metrics = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        column
        for column in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc")
        if column in metrics.columns
    ]
    rows = []
    for column in numeric_columns:
        values = metrics[column].dropna().astype(float)
        rows.append(
            {
                "metric": column,
                "seed_mean": float(values.mean()),
                "seed_std": float(values.std(ddof=0)),
                "seed_min": float(values.min()),
                "seed_max": float(values.max()),
                "n_seeds": int(values.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def per_subject_error_frequency(prediction_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame.copy() for frame in prediction_frames]
    if not frames:
        raise ValueError("At least one prediction frame is required.")
    predictions = pd.concat(frames, ignore_index=True)
    required = {"subject_id", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame missing required column(s): {', '.join(sorted(missing))}.")
    predictions["is_error"] = predictions["y_true"].astype(int) != predictions["y_pred"].astype(int)
    return (
        predictions.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            n_seeds=("is_error", "size"),
            n_errors=("is_error", "sum"),
        )
        .assign(error_frequency=lambda frame: frame["n_errors"] / frame["n_seeds"])
        .sort_values(["error_frequency", "subject_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _make_segment_batch(
    records: list[SegmentSSLRecord],
    *,
    branches: tuple[str, ...],
    scaler: SegmentSSLScaler,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = {}
    for branch in branches:
        arrays = [scaler.transform(record, branch) for record in records]
        batch[branch] = torch.as_tensor(np.stack(arrays).astype(np.float32), device=device)
    return batch


def _alignment_components(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    config: SegmentSSLTrainingConfig,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), device=projection_a.device)
    if config.objective == "barlow":
        components = barlow_twins_loss(
            projection_a,
            projection_b,
            offdiag_weight=config.barlow_offdiag_weight,
            eps=config.barlow_eps,
        )
        return {
            "loss": components["loss"],
            "on_diag": components["on_diag"],
            "off_diag": components["off_diag"],
            "invariance": zero,
            "variance": zero,
            "covariance": zero,
        }
    if config.objective == "vicreg":
        components = vicreg_loss(
            projection_a,
            projection_b,
            invariance_weight=config.vicreg_invariance_weight,
            variance_weight=config.vicreg_variance_weight,
            covariance_weight=config.vicreg_covariance_weight,
            variance_target=config.vicreg_variance_target,
            eps=config.vicreg_eps,
        )
        return {
            "loss": components["loss"],
            "on_diag": zero,
            "off_diag": zero,
            "invariance": components["invariance"],
            "variance": components["variance"],
            "covariance": components["covariance"],
        }
    raise ValueError(f"Unsupported objective: {config.objective}.")


def _model_masked_latent_loss(
    model: _SegmentSSLModel,
    clean: Mapping[str, torch.Tensor],
    masked: Mapping[str, torch.Tensor],
    config: SegmentSSLTrainingConfig,
) -> torch.Tensor:
    with torch.no_grad():
        target = model.encode(clean).detach()
    prediction = model.predict_latent(masked)
    if config.masked_latent_loss == "cosine":
        prediction = F.normalize(prediction.float(), dim=1)
        target = F.normalize(target.float(), dim=1)
        return 2.0 - 2.0 * (prediction * target).sum(dim=1).mean()
    if config.masked_latent_loss == "mse":
        return F.mse_loss(prediction.float(), target.float())
    raise ValueError("masked_latent_loss must be 'cosine' or 'mse'.")


def _local_reconstruction_loss(
    model: _SegmentSSLModel,
    clean: Mapping[str, torch.Tensor],
    masked: Mapping[str, torch.Tensor],
    branches: tuple[str, ...],
) -> torch.Tensor:
    losses = []
    for branch in branches:
        losses.append(F.mse_loss(model.reconstruct_local(masked, branch), clean[branch]))
    return torch.stack(losses).mean()


def _masked_latent_view(
    batch: Mapping[str, torch.Tensor],
    config: SegmentSSLTrainingConfig,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    return augment_segment_batch(
        batch,
        seed=seed,
        psd_channel_mask_prob=max(config.psd_channel_mask_prob, config.feature_mask_prob),
        psd_frequency_mask_prob=max(config.psd_frequency_mask_prob, config.feature_mask_prob),
        feature_mask_prob=config.feature_mask_prob,
        fc_node_mask_prob=max(config.fc_node_mask_prob, config.feature_mask_prob),
        fc_edge_mask_prob=max(config.fc_edge_mask_prob, config.feature_mask_prob),
        fc_band_mask_prob=max(config.fc_band_mask_prob, config.feature_mask_prob),
        noise_std=0.0,
        amplitude_scale_range=(1.0, 1.0),
    )


def _augment_psd_tensor(
    value: torch.Tensor,
    *,
    generator: torch.Generator,
    channel_mask_prob: float,
    frequency_mask_prob: float,
    element_mask_prob: float,
) -> torch.Tensor:
    _validate_feature_tensor_shape(value, PSD_SHAPE, "psd")
    view = value
    if channel_mask_prob > 0:
        mask = torch.rand((view.shape[0], view.shape[1]), generator=generator, device=view.device) < channel_mask_prob
        view = view.masked_fill(mask.unsqueeze(2), 0.0)
    if frequency_mask_prob > 0:
        mask = torch.rand((view.shape[0], view.shape[2]), generator=generator, device=view.device) < frequency_mask_prob
        view = view.masked_fill(mask.unsqueeze(1), 0.0)
    if element_mask_prob > 0:
        mask = torch.rand(view.shape, generator=generator, device=view.device) < element_mask_prob
        view = view.masked_fill(mask, 0.0)
    return view


def _augment_fc_tensor(
    value: torch.Tensor,
    *,
    generator: torch.Generator,
    node_mask_prob: float,
    edge_mask_prob: float,
    band_mask_prob: float,
    element_mask_prob: float,
) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError(f"FC segment tensors must be shaped (batch, edge, band), got {tuple(value.shape)}.")
    view = value
    if node_mask_prob > 0 and value.shape[1] == FC_SHAPE[0]:
        edge_index = torch.as_tensor(build_fc_edge_index(), dtype=torch.long, device=value.device)
        node_mask = torch.rand((value.shape[0], 62), generator=generator, device=value.device) < node_mask_prob
        edge_mask = node_mask[:, edge_index[:, 0]] | node_mask[:, edge_index[:, 1]]
        view = view.masked_fill(edge_mask.unsqueeze(2), 0.0)
    if edge_mask_prob > 0:
        mask = torch.rand((view.shape[0], view.shape[1]), generator=generator, device=view.device) < edge_mask_prob
        view = view.masked_fill(mask.unsqueeze(2), 0.0)
    if band_mask_prob > 0:
        mask = torch.rand((view.shape[0], view.shape[2]), generator=generator, device=view.device) < band_mask_prob
        view = view.masked_fill(mask.unsqueeze(1), 0.0)
    if element_mask_prob > 0:
        mask = torch.rand(view.shape, generator=generator, device=view.device) < element_mask_prob
        view = view.masked_fill(mask, 0.0)
    return view


def _validate_config(config: SegmentSSLTrainingConfig) -> SegmentSSLTrainingConfig:
    if config.objective not in SEGMENT_SSL_OBJECTIVES:
        raise ValueError("objective must be 'barlow' or 'vicreg'.")
    branches_for_feature_kind(config.feature_kind)
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if config.embedding_dim < 1 or config.projection_dim < 1:
        raise ValueError("embedding_dim and projection_dim must be positive.")
    if config.lr <= 0:
        raise ValueError("lr must be positive.")
    if config.dropout < 0:
        raise ValueError("dropout must be non-negative.")
    if config.lambda_latent < 0 or config.lambda_local < 0:
        raise ValueError("lambda_latent and lambda_local must be non-negative.")
    for name, value in (
        ("feature_mask_prob", config.feature_mask_prob),
        ("psd_channel_mask_prob", config.psd_channel_mask_prob),
        ("psd_frequency_mask_prob", config.psd_frequency_mask_prob),
        ("fc_node_mask_prob", config.fc_node_mask_prob),
        ("fc_edge_mask_prob", config.fc_edge_mask_prob),
        ("fc_band_mask_prob", config.fc_band_mask_prob),
    ):
        _validate_probability(name, value)
    if config.noise_std < 0:
        raise ValueError("noise_std must be non-negative.")
    if config.vicreg_variance_target <= 0 or config.vicreg_eps <= 0:
        raise ValueError("vicreg_variance_target and vicreg_eps must be positive.")
    if config.barlow_offdiag_weight < 0 or config.barlow_eps <= 0:
        raise ValueError("barlow_offdiag_weight must be non-negative and barlow_eps must be positive.")
    if config.masked_latent_loss not in {"cosine", "mse"}:
        raise ValueError("masked_latent_loss must be 'cosine' or 'mse'.")
    return config


def _validate_feature_tensor_shape(value: torch.Tensor, expected: tuple[int, ...], branch: str) -> None:
    if tuple(value.shape[1:]) != expected:
        raise ValueError(f"Expected {branch} segment tensors shaped (*, {expected}), got {tuple(value.shape)}.")


def _validate_probability(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _standardize(projection: torch.Tensor, eps: float) -> torch.Tensor:
    return (projection - projection.mean(dim=0)) / (projection.std(dim=0, unbiased=False) + eps)


def _off_diagonal_covariance_loss(projection: torch.Tensor) -> torch.Tensor:
    batch_size, feature_dim = projection.shape
    centered = projection - projection.mean(dim=0)
    covariance = centered.T @ centered / max(batch_size - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return off_diagonal.pow(2).sum() / feature_dim


def _stable_key_offset(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(value))


def _number_token(value: float) -> str:
    return str(value).replace(".", "_")


def _safe_run_name(run_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_-")
    if not safe:
        raise ValueError("run_name must contain at least one filename-safe character.")
    return safe


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
