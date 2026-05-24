from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch

from eeg_recovery.config import PathConfig
from eeg_recovery.io.eeglab import read_eeglab_fdt, read_eeglab_set_metadata
from eeg_recovery.io.index import EEGFileRecord, build_eeg_file_index
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.models.ssl_model import (
    MaskedReconstructionLoss,
    NTXentLoss,
    SSLTimeSeriesModel,
)
from eeg_recovery.training.train_supervised import resolve_device


SSLDataScope = Literal[
    "supervised-baseline",
    "all-patient-baseline",
    "all-patient",
    "all-patient-health",
    "health-only",
]

SSLObjective = Literal["contrastive", "reconstruction", "combined"]

SSL_DATA_SCOPES: tuple[str, ...] = (
    "supervised-baseline",
    "all-patient-baseline",
    "all-patient",
    "all-patient-health",
    "health-only",
)

SSL_SCOPE_ALIASES = {
    "supervised": "supervised-baseline",
    "supervised_baseline": "supervised-baseline",
    "all_patient_baseline": "all-patient-baseline",
    "patients-baseline": "all-patient-baseline",
    "all_patient": "all-patient",
    "patients": "all-patient",
    "supervised-unsupervised-patient": "all-patient",
    "all_patient_health": "all-patient-health",
    "patients-health": "all-patient-health",
    "health": "health-only",
    "health_only": "health-only",
}


@dataclass(frozen=True)
class SSLAugmentationConfig:
    crop_samples: int = 512
    noise_std: float = 0.01
    amplitude_scale_range: tuple[float, float] = (0.9, 1.1)
    channel_dropout_prob: float = 0.05
    time_mask_prob: float = 0.2
    time_mask_fraction: float = 0.1


@dataclass(frozen=True)
class SSLTrainingConfig:
    data_scope: str = "supervised-baseline"
    objective: SSLObjective = "combined"
    epochs: int = 20
    batch_size: int = 8
    crop_samples: int = 512
    embedding_dim: int = 32
    projection_dim: int = 16
    temperature: float = 0.2
    reconstruction_weight: float = 1.0
    reconstruction_mask_fraction: float = 0.15
    lr: float = 1e-3
    device: str = "auto"
    seed: int = 42


@dataclass(frozen=True)
class SSLViews:
    view_a: np.ndarray
    view_b: np.ndarray


def normalize_ssl_data_scope(data_scope: str) -> str:
    normalized = data_scope.strip().lower()
    normalized = SSL_SCOPE_ALIASES.get(normalized, normalized)
    if normalized not in SSL_DATA_SCOPES:
        joined = ", ".join(SSL_DATA_SCOPES)
        raise ValueError(f"data_scope must be one of {joined}; got {data_scope!r}.")
    return normalized


def build_ssl_file_records(
    config: PathConfig,
    supervised_subject_ids: Iterable[str],
    *,
    data_scope: str,
    strict_loso_test_subject_id: str | None = None,
) -> list[EEGFileRecord]:
    records = build_eeg_file_index(
        config.patient_eeg_root,
        config.health_eeg_root,
        supervised_subject_ids=supervised_subject_ids,
        validate_supervised_baseline=True,
    )
    return select_ssl_records(
        records,
        data_scope=data_scope,
        strict_loso_test_subject_id=strict_loso_test_subject_id,
    )


def select_ssl_records(
    records: Iterable[EEGFileRecord],
    *,
    data_scope: str,
    strict_loso_test_subject_id: str | None = None,
) -> list[EEGFileRecord]:
    """Select unlabeled EEG records for an SSL pretraining data pool."""

    scope = normalize_ssl_data_scope(data_scope)
    excluded_subject = (
        normalize_subject_id(strict_loso_test_subject_id)
        if strict_loso_test_subject_id is not None
        else None
    )

    selected: list[EEGFileRecord] = []
    for record in records:
        if (
            excluded_subject is not None
            and record.group == "patient"
            and record.subject_id == excluded_subject
        ):
            continue
        if _record_in_scope(record, scope):
            selected.append(record)

    return sorted(
        selected,
        key=lambda record: (
            record.group,
            record.subject_id,
            record.stage,
            record.state,
            str(record.set_path),
        ),
    )


def _record_in_scope(record: EEGFileRecord, scope: str) -> bool:
    if scope == "supervised-baseline":
        return record.group == "patient" and record.stage == "基线" and record.is_supervised_subject
    if scope == "all-patient-baseline":
        return record.group == "patient" and record.stage == "基线"
    if scope == "all-patient":
        return record.group == "patient"
    if scope == "all-patient-health":
        return record.group in {"patient", "health"}
    if scope == "health-only":
        return record.group == "health"
    raise ValueError(f"Unsupported SSL data scope: {scope}")


def make_ssl_views(
    array: np.ndarray,
    *,
    seed: int,
    config: SSLAugmentationConfig,
) -> SSLViews:
    if array.ndim != 2:
        raise ValueError(f"Expected array shape (channels, samples), got {array.shape}.")
    if config.crop_samples <= 0:
        raise ValueError("crop_samples must be positive.")
    rng = np.random.default_rng(seed)
    view_a = _augment_one_view(array, rng, config)
    view_b = _augment_one_view(array, rng, config)
    return SSLViews(view_a=view_a.astype(np.float32), view_b=view_b.astype(np.float32))


def _augment_one_view(
    array: np.ndarray,
    rng: np.random.Generator,
    config: SSLAugmentationConfig,
) -> np.ndarray:
    crop = _time_crop_or_pad(array.astype(np.float32, copy=False), config.crop_samples, rng)
    scale_low, scale_high = config.amplitude_scale_range
    if scale_low <= 0 or scale_high <= 0 or scale_low > scale_high:
        raise ValueError("amplitude_scale_range must contain positive values ordered as (low, high).")
    crop = crop * np.float32(rng.uniform(scale_low, scale_high))
    if config.noise_std > 0:
        crop = crop + rng.normal(0.0, config.noise_std, size=crop.shape).astype(np.float32)
    if config.channel_dropout_prob > 0:
        drop_mask = rng.random(crop.shape[0]) < config.channel_dropout_prob
        crop = crop.copy()
        crop[drop_mask, :] = 0.0
    if config.time_mask_prob > 0 and rng.random() < config.time_mask_prob:
        crop = crop.copy()
        mask_len = max(1, int(round(crop.shape[1] * config.time_mask_fraction)))
        mask_len = min(mask_len, crop.shape[1])
        start = int(rng.integers(0, crop.shape[1] - mask_len + 1))
        crop[:, start : start + mask_len] = 0.0
    return crop


def _time_crop_or_pad(
    array: np.ndarray,
    crop_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_samples = array.shape[1]
    if n_samples == crop_samples:
        return array.copy()
    if n_samples > crop_samples:
        start = int(rng.integers(0, n_samples - crop_samples + 1))
        return array[:, start : start + crop_samples].copy()
    padded = np.zeros((array.shape[0], crop_samples), dtype=np.float32)
    padded[:, :n_samples] = array
    return padded


def load_ssl_arrays(records: Iterable[EEGFileRecord]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for record in records:
        metadata = read_eeglab_set_metadata(record.set_path)
        arrays.append(read_eeglab_fdt(metadata, subject_id=record.subject_id, state=record.state).astype(np.float32))
    return arrays


def run_ssl_pretraining_on_arrays(
    arrays: Iterable[np.ndarray],
    training_config: SSLTrainingConfig,
) -> pd.DataFrame:
    config = _validate_training_config(training_config)
    arrays = [np.asarray(array, dtype=np.float32) for array in arrays]
    if not arrays:
        raise ValueError("SSL pretraining requires at least one EEG array.")
    n_channels = arrays[0].shape[0]
    if any(array.ndim != 2 or array.shape[0] != n_channels for array in arrays):
        raise ValueError("All SSL arrays must have shape (same_channels, samples).")

    device = resolve_device(config.device)
    model = SSLTimeSeriesModel(
        n_channels=n_channels,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    contrastive_loss_fn = NTXentLoss(temperature=config.temperature)
    reconstruction_loss_fn = MaskedReconstructionLoss()
    augmentation_config = SSLAugmentationConfig(crop_samples=config.crop_samples)
    rng = np.random.default_rng(config.seed)

    rows = []
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(arrays))
        epoch_losses: list[float] = []
        epoch_contrastive: list[float] = []
        epoch_reconstruction: list[float] = []
        model.train()
        for start in range(0, len(order), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            views = [
                make_ssl_views(
                    arrays[int(index)],
                    seed=config.seed + epoch * 100_000 + int(index),
                    config=augmentation_config,
                )
                for index in batch_indices
            ]
            view_a = torch.as_tensor(np.stack([view.view_a for view in views]), device=device)
            view_b = torch.as_tensor(np.stack([view.view_b for view in views]), device=device)

            optimizer.zero_grad()
            output_a = model(view_a)
            output_b = model(view_b)
            contrastive = contrastive_loss_fn(output_a.projection, output_b.projection)
            mask = _make_reconstruction_mask(
                view_a.shape,
                mask_fraction=config.reconstruction_mask_fraction,
                device=device,
                seed=config.seed + epoch * 10_000 + start,
            )
            reconstruction = reconstruction_loss_fn(output_a.reconstruction, view_a, mask)
            loss = _combine_ssl_losses(
                objective=config.objective,
                contrastive_loss=contrastive,
                reconstruction_loss=reconstruction,
                reconstruction_weight=config.reconstruction_weight,
            )
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_contrastive.append(float(contrastive.detach().cpu().item()))
            epoch_reconstruction.append(float(reconstruction.detach().cpu().item()))

        rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(epoch_losses)),
                "contrastive_loss": float(np.mean(epoch_contrastive)),
                "reconstruction_loss": float(np.mean(epoch_reconstruction)),
                "objective": config.objective,
                "data_scope": normalize_ssl_data_scope(config.data_scope),
                "n_arrays": len(arrays),
                "batch_size": config.batch_size,
                "crop_samples": config.crop_samples,
                "embedding_dim": config.embedding_dim,
                "projection_dim": config.projection_dim,
                "learning_rate": config.lr,
            }
        )
    return pd.DataFrame(rows)


def run_ssl_pretraining(
    records: Iterable[EEGFileRecord],
    training_config: SSLTrainingConfig,
) -> pd.DataFrame:
    return run_ssl_pretraining_on_arrays(load_ssl_arrays(records), training_config)


def write_ssl_outputs(
    history: pd.DataFrame,
    output_root: str | Path,
    *,
    run_name: str,
) -> Path:
    if history.empty:
        raise ValueError("history is empty.")
    output_path = Path(output_root) / "results" / "ssl" / f"ssl_pretraining_history_{_safe_run_name(run_name)}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_path, index=False)
    return output_path


def _validate_training_config(config: SSLTrainingConfig) -> SSLTrainingConfig:
    normalize_ssl_data_scope(config.data_scope)
    if config.objective not in {"contrastive", "reconstruction", "combined"}:
        raise ValueError("objective must be 'contrastive', 'reconstruction', or 'combined'.")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if config.crop_samples < 1:
        raise ValueError("crop_samples must be at least 1.")
    if config.lr <= 0:
        raise ValueError("lr must be positive.")
    if config.reconstruction_weight < 0:
        raise ValueError("reconstruction_weight must be non-negative.")
    if not 0 <= config.reconstruction_mask_fraction <= 1:
        raise ValueError("reconstruction_mask_fraction must be between 0 and 1.")
    return config


def _combine_ssl_losses(
    *,
    objective: str,
    contrastive_loss: torch.Tensor,
    reconstruction_loss: torch.Tensor,
    reconstruction_weight: float,
) -> torch.Tensor:
    if objective == "contrastive":
        return contrastive_loss
    if objective == "reconstruction":
        return reconstruction_loss
    if objective == "combined":
        return contrastive_loss + reconstruction_weight * reconstruction_loss
    raise ValueError(f"Unsupported SSL objective: {objective}")


def _make_reconstruction_mask(
    shape: torch.Size,
    *,
    mask_fraction: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    if mask_fraction <= 0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand(shape, generator=generator, device=device) < mask_fraction


def _safe_run_name(run_name: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in run_name)
    safe = safe.strip("_-")
    if not safe:
        raise ValueError("run_name must contain at least one filename-safe character.")
    return safe
