from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.signal import stft
from torch import nn

from eeg_recovery.channels.hemisphere_flip import flip_channels_for_affected_hand
from eeg_recovery.io.eeglab import read_eeglab_fdt, read_eeglab_set_metadata
from eeg_recovery.io.index import EEGFileRecord
from eeg_recovery.metadata.subjects import normalize_subject_id
from eeg_recovery.models.encoders import FCConv1DEncoder, PSDConv2DEncoder
from eeg_recovery.models.ssl_model import NTXentLoss
from eeg_recovery.training.train_feature_ssl import FeatureSSLPairRecord
from eeg_recovery.training.train_supervised import resolve_device


@dataclass(frozen=True)
class StructuredSSLConfig:
    psd_epochs: int = 10
    fc_epochs: int = 10
    batch_size: int = 8
    embedding_dim: int = 32
    projection_dim: int = 16
    lr: float = 1e-3
    temperature: float = 0.2
    tfr_noise_std: float = 0.02
    tfr_channel_mask_prob: float = 0.05
    tfr_frequency_mask_prob: float = 0.05
    fc_noise_std: float = 0.02
    fc_node_dropout_prob: float = 0.05
    fc_edge_dropout_prob: float = 0.05
    device: str = "auto"
    seed: int = 42


def build_time_frequency_images(
    data: np.ndarray,
    *,
    sampling_rate: float,
    max_windows: int = 4,
    nperseg: int = 256,
    noverlap: int = 128,
    channel_names: Sequence[str] | None = None,
    affected_hand: str | None = None,
) -> np.ndarray:
    """Build channel-by-frequency STFT power images shaped windows x 62 x 90."""

    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected raw EEG shape channels x samples, got {array.shape}.")
    if array.shape[0] != 62:
        raise ValueError(f"Expected 62 channels, got {array.shape[0]}.")
    if max_windows < 1:
        raise ValueError("max_windows must be positive.")
    if array.shape[1] < nperseg:
        raise ValueError(f"Need at least {nperseg} samples for STFT, got {array.shape[1]}.")
    if (channel_names is None) != (affected_hand is None):
        raise ValueError("channel_names and affected_hand must be provided together.")
    if channel_names is not None and affected_hand is not None:
        array = flip_channels_for_affected_hand(
            array,
            channel_names,
            affected_hand,
            channel_axis=0,
        ).astype(np.float32, copy=False)

    frequencies, _, spectra = stft(
        array,
        fs=float(sampling_rate),
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend="constant",
        boundary=None,
        padded=False,
        axis=1,
    )
    power = np.log1p(np.abs(spectra) ** 2).astype(np.float32)
    target_frequencies = np.linspace(0.5, 45.0, 90, dtype=np.float32)
    images = np.empty((power.shape[2], array.shape[0], target_frequencies.size), dtype=np.float32)
    for channel_index in range(array.shape[0]):
        images[:, channel_index, :] = np.stack(
            [
                np.interp(target_frequencies, frequencies, power[channel_index, :, window_index])
                for window_index in range(power.shape[2])
            ]
        )
    if images.shape[0] > max_windows:
        indices = np.linspace(0, images.shape[0] - 1, max_windows).round().astype(int)
        images = images[indices]
    return images.astype(np.float32, copy=False)


def load_time_frequency_images_for_records(
    records: Iterable[EEGFileRecord],
    *,
    max_windows_per_record: int,
    nperseg: int = 256,
    noverlap: int = 128,
    affected_hand_by_subject: Mapping[str, str] | None = None,
) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for record in records:
        metadata = read_eeglab_set_metadata(record.set_path)
        data = read_eeglab_fdt(metadata, subject_id=record.subject_id, state=record.state)
        affected_hand = (
            _affected_hand_for_tfr_record(record, affected_hand_by_subject)
            if affected_hand_by_subject is not None
            else None
        )
        record_images = build_time_frequency_images(
            data,
            sampling_rate=metadata.srate,
            max_windows=max_windows_per_record,
            nperseg=nperseg,
            noverlap=noverlap,
            channel_names=metadata.ch_names if affected_hand is not None else None,
            affected_hand=affected_hand,
        )
        images.extend(record_images)
    return images


def build_fc_edge_index(n_nodes: int = 62) -> np.ndarray:
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2.")
    return np.array(
        [(first, second) for first in range(n_nodes) for second in range(first + 1, n_nodes)],
        dtype=np.int64,
    )


def _affected_hand_for_tfr_record(
    record: EEGFileRecord,
    affected_hand_by_subject: Mapping[str, str],
) -> str:
    if record.group == "health":
        return "右"
    affected = affected_hand_by_subject.get(normalize_subject_id(record.subject_id))
    if affected not in {"右", "左"}:
        raise ValueError(f"Missing valid affected hand for patient {record.subject_id}.")
    return affected


def build_fc_graph_views(
    fc: np.ndarray,
    *,
    edge_index: np.ndarray,
    seed: int,
    node_dropout_prob: float = 0.05,
    edge_dropout_prob: float = 0.05,
    noise_std: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(fc, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected FC feature shape edges x bands, got {array.shape}.")
    if edge_index.shape != (array.shape[0], 2):
        raise ValueError(f"edge_index shape must be ({array.shape[0]}, 2), got {edge_index.shape}.")
    rng = np.random.default_rng(seed)
    return (
        _augment_fc_graph(array, edge_index, rng, node_dropout_prob, edge_dropout_prob, noise_std),
        _augment_fc_graph(array, edge_index, rng, node_dropout_prob, edge_dropout_prob, noise_std),
    )


def run_structured_ssl_pretraining(
    tfr_images: Iterable[np.ndarray],
    pairs: Iterable[FeatureSSLPairRecord],
    config: StructuredSSLConfig,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    _validate_structured_ssl_config(config)
    device = resolve_device(config.device)
    rows: list[dict[str, object]] = []

    psd_state, psd_history = _pretrain_psd_tfr_encoder(list(tfr_images), config, device)
    rows.extend(psd_history)
    fc_state, fc_history = _pretrain_fc_graph_encoder(list(pairs), config, device)
    rows.extend(fc_history)

    state: dict[str, torch.Tensor] = {}
    state.update({f"branch_models.psd.encoder.encoder.{key}": value for key, value in psd_state.items()})
    state.update({f"branch_models.wpli.encoder.encoder.{key}": value for key, value in fc_state.items()})
    return state, pd.DataFrame(rows)


def _pretrain_psd_tfr_encoder(
    tfr_images: list[np.ndarray],
    config: StructuredSSLConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    arrays = [np.asarray(image, dtype=np.float32) for image in tfr_images]
    if not arrays:
        raise ValueError("TFR SSL requires at least one time-frequency image.")
    if any(array.shape != (62, 90) for array in arrays):
        raise ValueError("All TFR images must have shape 62 x 90.")

    encoder = PSDConv2DEncoder(embedding_dim=config.embedding_dim, dropout=0.0).to(device)
    projection = nn.Linear(config.embedding_dim, config.projection_dim).to(device)
    optimizer = torch.optim.Adam([*encoder.parameters(), *projection.parameters()], lr=config.lr)
    loss_fn = NTXentLoss(temperature=config.temperature)
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []
    for epoch in range(1, config.psd_epochs + 1):
        order = rng.permutation(len(arrays))
        losses: list[float] = []
        encoder.train()
        projection.train()
        for start in range(0, len(order), config.batch_size):
            batch_arrays = [arrays[int(index)] for index in order[start : start + config.batch_size]]
            view_a = np.stack(
                [
                    _augment_tfr_image(
                        image,
                        seed=config.seed + epoch * 100_000 + start + index,
                        noise_std=config.tfr_noise_std,
                        channel_mask_prob=config.tfr_channel_mask_prob,
                        frequency_mask_prob=config.tfr_frequency_mask_prob,
                    )
                    for index, image in enumerate(batch_arrays)
                ]
            )
            view_b = np.stack(
                [
                    _augment_tfr_image(
                        image,
                        seed=config.seed + epoch * 200_000 + start + index,
                        noise_std=config.tfr_noise_std,
                        channel_mask_prob=config.tfr_channel_mask_prob,
                        frequency_mask_prob=config.tfr_frequency_mask_prob,
                    )
                    for index, image in enumerate(batch_arrays)
                ]
            )
            tensor_a = torch.as_tensor(view_a, device=device)
            tensor_b = torch.as_tensor(view_b, device=device)
            optimizer.zero_grad()
            loss = loss_fn(projection(encoder(tensor_a)), projection(encoder(tensor_b)))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        rows.append({"branch": "psd_tfr", "epoch": epoch, "loss": float(np.mean(losses)), "n_samples": len(arrays)})
    return {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}, rows


def _pretrain_fc_graph_encoder(
    pairs: list[FeatureSSLPairRecord],
    config: StructuredSSLConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    arrays = [
        state
        for pair in pairs
        for state in pair.modalities.get("wpli", ())
    ]
    arrays = [np.asarray(array, dtype=np.float32) for array in arrays]
    if not arrays:
        raise ValueError("Graph FC SSL requires at least one wPLI feature array.")
    if any(array.shape != (1891, 6) for array in arrays):
        raise ValueError("All wPLI arrays must have shape 1891 x 6.")

    edge_index = build_fc_edge_index()
    encoder = FCConv1DEncoder(embedding_dim=config.embedding_dim, dropout=0.0).to(device)
    projection = nn.Linear(config.embedding_dim, config.projection_dim).to(device)
    optimizer = torch.optim.Adam([*encoder.parameters(), *projection.parameters()], lr=config.lr)
    loss_fn = NTXentLoss(temperature=config.temperature)
    rng = np.random.default_rng(config.seed + 17)
    rows: list[dict[str, object]] = []
    for epoch in range(1, config.fc_epochs + 1):
        order = rng.permutation(len(arrays))
        losses: list[float] = []
        encoder.train()
        projection.train()
        for start in range(0, len(order), config.batch_size):
            batch_arrays = [arrays[int(index)] for index in order[start : start + config.batch_size]]
            views = [
                build_fc_graph_views(
                    array,
                    edge_index=edge_index,
                    seed=config.seed + epoch * 100_000 + start + index,
                    node_dropout_prob=config.fc_node_dropout_prob,
                    edge_dropout_prob=config.fc_edge_dropout_prob,
                    noise_std=config.fc_noise_std,
                )
                for index, array in enumerate(batch_arrays)
            ]
            tensor_a = torch.as_tensor(np.stack([view[0] for view in views]), device=device)
            tensor_b = torch.as_tensor(np.stack([view[1] for view in views]), device=device)
            optimizer.zero_grad()
            loss = loss_fn(projection(encoder(tensor_a)), projection(encoder(tensor_b)))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        rows.append({"branch": "wpli_graph", "epoch": epoch, "loss": float(np.mean(losses)), "n_samples": len(arrays)})
    return {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}, rows


def _augment_tfr_image(
    image: np.ndarray,
    *,
    seed: int,
    noise_std: float,
    channel_mask_prob: float,
    frequency_mask_prob: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    view = np.array(image, dtype=np.float32, copy=True)
    if noise_std > 0:
        view += rng.normal(0.0, noise_std, size=view.shape).astype(np.float32)
    if channel_mask_prob > 0:
        channel_mask = rng.random(view.shape[0]) < channel_mask_prob
        view[channel_mask, :] = 0.0
    if frequency_mask_prob > 0:
        frequency_mask = rng.random(view.shape[1]) < frequency_mask_prob
        view[:, frequency_mask] = 0.0
    return view


def _augment_fc_graph(
    fc: np.ndarray,
    edge_index: np.ndarray,
    rng: np.random.Generator,
    node_dropout_prob: float,
    edge_dropout_prob: float,
    noise_std: float,
) -> np.ndarray:
    view = np.array(fc, dtype=np.float32, copy=True)
    n_nodes = int(edge_index.max()) + 1
    if noise_std > 0:
        view += rng.normal(0.0, noise_std, size=view.shape).astype(np.float32)
    if node_dropout_prob > 0:
        dropped_nodes = rng.random(n_nodes) < node_dropout_prob
        incident = dropped_nodes[edge_index[:, 0]] | dropped_nodes[edge_index[:, 1]]
        view[incident, :] = 0.0
    if edge_dropout_prob > 0:
        dropped_edges = rng.random(view.shape[0]) < edge_dropout_prob
        view[dropped_edges, :] = 0.0
    return view


def _validate_structured_ssl_config(config: StructuredSSLConfig) -> None:
    if config.psd_epochs < 1 or config.fc_epochs < 1:
        raise ValueError("psd_epochs and fc_epochs must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if config.embedding_dim < 1 or config.projection_dim < 1:
        raise ValueError("embedding_dim and projection_dim must be positive.")
    if config.lr <= 0 or config.temperature <= 0:
        raise ValueError("lr and temperature must be positive.")
    for name, value in (
        ("tfr_channel_mask_prob", config.tfr_channel_mask_prob),
        ("tfr_frequency_mask_prob", config.tfr_frequency_mask_prob),
        ("fc_node_dropout_prob", config.fc_node_dropout_prob),
        ("fc_edge_dropout_prob", config.fc_edge_dropout_prob),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    if config.tfr_noise_std < 0 or config.fc_noise_std < 0:
        raise ValueError("noise std values must be non-negative.")
