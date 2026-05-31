from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
import re

import numpy as np

from eeg_recovery.channels.hemisphere_flip import flip_channels_for_affected_hand
from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62, ChannelMappingError
from eeg_recovery.features.eeg_summary import ROI_PREFIXES
from eeg_recovery.io.eeglab import read_eeglab_fdt, read_eeglab_set_metadata
from eeg_recovery.io.index import EEGFileRecord


class ComplexityFeatureError(ValueError):
    """Raised when EEG complexity features cannot be computed safely."""


@dataclass(frozen=True)
class ComplexityConfig:
    kmax: int = 8
    max_samples: int = 20000


@dataclass(frozen=True)
class ComplexityFeatureVector:
    values: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class ComplexityFeature:
    channel_higuchi_fd: np.ndarray
    subject_id: str
    state: str
    channel_names_after_alignment: tuple[str, ...]
    sampling_rate: float
    source_set_path: Path
    metadata: dict[str, object]


def higuchi_fractal_dimension(signal: np.ndarray, *, kmax: int = 8) -> float:
    """Compute Higuchi fractal dimension for one EEG channel."""

    series = np.asarray(signal, dtype=np.float64).reshape(-1)
    series = series[np.isfinite(series)]
    if series.size < max(16, kmax * 4):
        raise ComplexityFeatureError("Higuchi FD requires more samples than kmax * 4.")
    if kmax < 2:
        raise ComplexityFeatureError("kmax must be at least 2.")

    series = series - float(np.mean(series))
    scale = float(np.std(series))
    if scale <= 1e-12:
        return 1.0
    series = series / scale

    lengths: list[float] = []
    scales: list[float] = []
    n_samples = int(series.size)
    for k in range(1, kmax + 1):
        k_lengths: list[float] = []
        for m in range(k):
            n_max = int(np.floor((n_samples - m - 1) / k))
            if n_max < 2:
                continue
            indices = m + np.arange(n_max + 1) * k
            diffs = np.abs(np.diff(series[indices]))
            normalizer = (n_samples - 1) / (n_max * k)
            k_lengths.append(float(np.sum(diffs) * normalizer))
        if k_lengths:
            length = float(np.mean(k_lengths))
            if length > 0 and np.isfinite(length):
                lengths.append(length)
                scales.append(1.0 / float(k))
    if len(lengths) < 2:
        return 1.0
    slope, _ = np.polyfit(np.log(scales), np.log(lengths), 1)
    value = float(slope)
    if not np.isfinite(value):
        return 1.0
    return float(np.clip(value, 1.0, 2.5))


def compute_complexity_summary_from_aligned_data(
    data: np.ndarray,
    *,
    channel_names: Sequence[str],
    state: str,
    config: ComplexityConfig = ComplexityConfig(),
) -> ComplexityFeatureVector:
    _validate_state(state)
    channel_names_tuple = tuple(str(name) for name in channel_names)
    if channel_names_tuple != CANONICAL_CHANNELS_62:
        raise ChannelMappingError("Complexity input channel_names must match the fixed 62-channel order")
    data_array = _prepare_data(data, config)
    channel_fd = np.asarray(
        [higuchi_fractal_dimension(channel, kmax=config.kmax) for channel in data_array],
        dtype=np.float32,
    )
    return _summarize_channel_higuchi_fd(channel_fd, channel_names_tuple, state)


def compute_complexity_for_eeg_record(
    record: EEGFileRecord,
    affected_hand: str,
    output_root: str | Path | None = None,
    config: ComplexityConfig = ComplexityConfig(),
) -> ComplexityFeature:
    metadata = read_eeglab_set_metadata(record.set_path)
    data = read_eeglab_fdt(metadata, subject_id=record.subject_id, state=record.state)
    data = _prepare_data(data, config)
    aligned = flip_channels_for_affected_hand(
        data,
        metadata.ch_names,
        affected_hand.strip(),
        channel_axis=0,
    )
    channel_fd = np.asarray(
        [higuchi_fractal_dimension(channel, kmax=config.kmax) for channel in aligned],
        dtype=np.float32,
    )
    feature = ComplexityFeature(
        channel_higuchi_fd=channel_fd,
        subject_id=record.subject_id,
        state=record.state,
        channel_names_after_alignment=tuple(CANONICAL_CHANNELS_62),
        sampling_rate=float(metadata.srate),
        source_set_path=metadata.set_path,
        metadata={
            "kmax": int(config.kmax),
            "max_samples": int(config.max_samples),
            "affected_hand": affected_hand.strip(),
            "hemisphere_aligned": True,
        },
    )
    if output_root is not None:
        write_complexity_feature(feature, output_root)
    return feature


def write_complexity_feature(feature: ComplexityFeature, output_root: str | Path) -> Path:
    _validate_feature_for_write(feature)
    feature_dir = Path(output_root) / "data" / "features" / "complexity"
    feature_dir.mkdir(parents=True, exist_ok=True)
    output_path = feature_dir / f"{feature.subject_id}_{feature.state}_complexity.npz"
    np.savez_compressed(
        output_path,
        channel_higuchi_fd=feature.channel_higuchi_fd,
        subject_id=np.array(feature.subject_id),
        state=np.array(feature.state),
        channel_names_after_alignment=np.array(feature.channel_names_after_alignment),
        sampling_rate=np.array(feature.sampling_rate),
        source_set_path=np.array(feature.source_set_path.as_posix()),
        **{key: np.array(value) for key, value in feature.metadata.items()},
    )
    return output_path


def compute_subject_complexity_summary_features(output_root: str | Path, subject_id: str) -> ComplexityFeatureVector:
    root = Path(output_root) / "data" / "features" / "complexity"
    payloads = {
        state: _load_npz(root / f"{subject_id}_{state}_complexity.npz")
        for state in ("EO", "EC")
    }
    vectors = [
        _summarize_channel_higuchi_fd(
            np.asarray(payloads[state]["channel_higuchi_fd"], dtype=np.float32),
            tuple(np.asarray(payloads[state]["channel_names_after_alignment"]).astype(str)),
            state,
        )
        for state in ("EO", "EC")
    ]
    delta_vector = _summarize_channel_higuchi_fd(
        np.abs(
            np.asarray(payloads["EC"]["channel_higuchi_fd"], dtype=np.float32)
            - np.asarray(payloads["EO"]["channel_higuchi_fd"], dtype=np.float32)
        ),
        tuple(np.asarray(payloads["EO"]["channel_names_after_alignment"]).astype(str)),
        "abs_delta",
    )
    values = np.concatenate([vectors[0].values, vectors[1].values, delta_vector.values]).astype(np.float32)
    names = (*vectors[0].feature_names, *vectors[1].feature_names, *delta_vector.feature_names)
    if not np.isfinite(values).all():
        raise ComplexityFeatureError("Subject complexity summary contains non-finite values.")
    return ComplexityFeatureVector(values=values, feature_names=names)


def _summarize_channel_higuchi_fd(
    channel_fd: np.ndarray,
    channel_names: Sequence[str],
    state: str,
) -> ComplexityFeatureVector:
    _validate_channel_vector(channel_fd, channel_names)
    state_token = _safe_feature_token(state)
    values: list[float] = []
    names: list[str] = []

    def add(name: str, value: float) -> None:
        names.append(f"complexity_{state_token}_{name}")
        values.append(float(value))

    add("higuchi_fd_mean", _finite_mean(channel_fd))
    add("higuchi_fd_std", _finite_std(channel_fd))
    ipsi_indices, contra_indices = _hemisphere_channel_indices(channel_names)
    ipsi_mean = _finite_mean(channel_fd[ipsi_indices]) if ipsi_indices.size else 0.0
    contra_mean = _finite_mean(channel_fd[contra_indices]) if contra_indices.size else 0.0
    add("higuchi_fd_ipsilesional_mean", ipsi_mean)
    add("higuchi_fd_contralesional_mean", contra_mean)
    add("higuchi_fd_ipsi_contra_signed_asymmetry", _signed_asymmetry(ipsi_mean, contra_mean))
    for roi_name, roi_indices in _roi_channel_indices(channel_names):
        roi_values = channel_fd[roi_indices]
        add(f"{roi_name}_higuchi_fd_mean", _finite_mean(roi_values))
        add(f"{roi_name}_higuchi_fd_std", _finite_std(roi_values))
    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ComplexityFeatureError("Complexity summary contains non-finite values.")
    return ComplexityFeatureVector(values=vector, feature_names=tuple(names))


def _prepare_data(data: np.ndarray, config: ComplexityConfig) -> np.ndarray:
    if config.max_samples < 128:
        raise ComplexityFeatureError("max_samples must be at least 128.")
    data_array = np.asarray(data, dtype=np.float32)
    if data_array.ndim != 2:
        raise ComplexityFeatureError(f"data must be channels x samples, got shape {data_array.shape}")
    if data_array.shape[0] != len(CANONICAL_CHANNELS_62):
        raise ComplexityFeatureError(f"Expected 62 channels, got {data_array.shape[0]}")
    if not np.isfinite(data_array).all():
        raise ComplexityFeatureError("data contains NaN or infinite values")
    if data_array.shape[1] > config.max_samples:
        data_array = data_array[:, : config.max_samples]
    return np.array(data_array, copy=True)


def _validate_channel_vector(channel_fd: np.ndarray, channel_names: Sequence[str]) -> None:
    if np.asarray(channel_fd).shape != (len(CANONICAL_CHANNELS_62),):
        raise ComplexityFeatureError(f"Expected 62 channel FD values, got {np.asarray(channel_fd).shape}")
    if tuple(channel_names) != CANONICAL_CHANNELS_62:
        raise ChannelMappingError("channel_names must match the fixed 62-channel order")
    if not np.isfinite(channel_fd).all():
        raise ComplexityFeatureError("channel FD values contain NaN or infinite values")


def _validate_feature_for_write(feature: ComplexityFeature) -> None:
    _validate_state(feature.state)
    _validate_channel_vector(feature.channel_higuchi_fd, feature.channel_names_after_alignment)


def _validate_state(state: str) -> None:
    if state not in {"EO", "EC", "abs_delta"}:
        raise ComplexityFeatureError(f"state must be EO, EC, or abs_delta, got {state!r}")


def _roi_channel_indices(channel_names: Sequence[str]) -> list[tuple[str, np.ndarray]]:
    labels = tuple(str(name) for name in channel_names)
    result: list[tuple[str, np.ndarray]] = []
    for roi_name, prefixes in ROI_PREFIXES:
        indices = [
            index
            for index, label in enumerate(labels)
            if any(label.upper().startswith(prefix) for prefix in prefixes)
        ]
        if indices:
            result.append((roi_name, np.asarray(indices, dtype=np.int64)))
    return result


def _hemisphere_channel_indices(channel_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    ipsi_indices = []
    contra_indices = []
    for index, label in enumerate(channel_names):
        hemisphere = _channel_hemisphere(label)
        if hemisphere == "left":
            ipsi_indices.append(index)
        elif hemisphere == "right":
            contra_indices.append(index)
    return np.asarray(ipsi_indices, dtype=np.int64), np.asarray(contra_indices, dtype=np.int64)


def _channel_hemisphere(label: str) -> str | None:
    match = re.search(r"(\d+)$", str(label).upper())
    if not match:
        return None
    return "left" if int(match.group(1)) % 2 == 1 else "right"


def _signed_asymmetry(ipsilesional_value: float, contralesional_value: float, eps: float = 1e-8) -> float:
    return float(
        (float(ipsilesional_value) - float(contralesional_value))
        / (abs(float(ipsilesional_value)) + abs(float(contralesional_value)) + eps)
    )


def _finite_mean(values: np.ndarray) -> float:
    mean = float(np.nanmean(np.asarray(values, dtype=np.float32)))
    return mean if np.isfinite(mean) else 0.0


def _finite_std(values: np.ndarray) -> float:
    std = float(np.nanstd(np.asarray(values, dtype=np.float32)))
    return std if np.isfinite(std) else 0.0


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Required complexity feature file is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _safe_feature_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token or "state"
