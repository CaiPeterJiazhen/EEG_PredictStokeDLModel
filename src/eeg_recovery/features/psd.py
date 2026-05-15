from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import welch

from eeg_recovery.channels.hemisphere_flip import flip_channels_for_affected_hand
from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62, ChannelMappingError
from eeg_recovery.io.eeglab import read_eeglab_fdt, read_eeglab_set_metadata
from eeg_recovery.io.index import EEGFileRecord


class PSDFeatureError(ValueError):
    """Raised when PSD feature inputs or outputs violate the project contract."""


@dataclass(frozen=True)
class PSDConfig:
    frequency_resolution_hz: float = 0.5
    min_frequency_hz: float = 0.5
    max_frequency_hz: float = 45.0
    nperseg: int | None = None
    overlap_fraction: float = 0.5
    window: str = "hann"
    scaling: str = "density"
    detrend: str = "constant"

    def welch_nperseg(self, sampling_rate: float) -> int:
        if self.nperseg is not None:
            return int(self.nperseg)
        return int(round(float(sampling_rate) / self.frequency_resolution_hz))

    def welch_noverlap(self, sampling_rate: float) -> int:
        return int(round(self.welch_nperseg(sampling_rate) * self.overlap_fraction))


@dataclass(frozen=True)
class PSDFeature:
    psd: np.ndarray
    frequency_bins: np.ndarray
    subject_id: str
    state: str
    channel_names_after_alignment: tuple[str, ...]
    sampling_rate: float
    source_set_path: Path
    metadata: dict[str, Any]


def target_frequency_bins(config: PSDConfig | None = None) -> np.ndarray:
    """Return the fixed 0.5-45 Hz bins used by downstream PSD models."""

    resolved = config if config is not None else PSDConfig()
    start = int(round(resolved.min_frequency_hz / resolved.frequency_resolution_hz))
    stop = int(round(resolved.max_frequency_hz / resolved.frequency_resolution_hz))
    bins = np.arange(start, stop + 1, dtype=float) * resolved.frequency_resolution_hz
    return bins


def compute_single_state_psd(
    data: np.ndarray,
    sampling_rate: float,
    channel_names: Sequence[str],
    affected_hand: str,
    subject_id: str,
    state: str,
    source_set_path: str | Path,
    config: PSDConfig | None = None,
) -> PSDFeature:
    """Compute one EO or EC PSD matrix after affected-hand channel alignment."""

    resolved = config if config is not None else PSDConfig()
    data_array = np.asarray(data, dtype=float)
    _validate_state(state)
    _validate_data_contract(data_array, sampling_rate, channel_names)
    normalized_hand = affected_hand.strip()

    aligned = flip_channels_for_affected_hand(
        data_array,
        channel_names,
        normalized_hand,
        channel_axis=0,
    )
    nperseg = resolved.welch_nperseg(sampling_rate)
    noverlap = resolved.welch_noverlap(sampling_rate)
    _validate_welch_settings(aligned, nperseg, noverlap)

    frequencies, power = welch(
        aligned,
        fs=float(sampling_rate),
        window=resolved.window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=resolved.detrend,
        scaling=resolved.scaling,
        axis=1,
    )
    expected_bins = target_frequency_bins(resolved)
    frequency_indices = _target_frequency_indices(frequencies, expected_bins)
    psd = power[:, frequency_indices]

    if psd.shape != (62, 90):
        raise PSDFeatureError(f"Expected PSD shape (62, 90), got {psd.shape}")

    metadata = {
        "frequency_resolution_hz": resolved.frequency_resolution_hz,
        "min_frequency_hz": resolved.min_frequency_hz,
        "max_frequency_hz": resolved.max_frequency_hz,
        "welch_nperseg": nperseg,
        "welch_noverlap": noverlap,
        "welch_window": resolved.window,
        "welch_scaling": resolved.scaling,
        "welch_detrend": resolved.detrend,
        "affected_hand": normalized_hand,
        "hemisphere_aligned": True,
    }
    return PSDFeature(
        psd=psd,
        frequency_bins=expected_bins,
        subject_id=subject_id,
        state=state,
        channel_names_after_alignment=tuple(CANONICAL_CHANNELS_62),
        sampling_rate=float(sampling_rate),
        source_set_path=Path(source_set_path),
        metadata=metadata,
    )


def write_psd_feature(feature: PSDFeature, output_root: str | Path) -> Path:
    """Write a single-state PSD feature under output_root/data/features/psd."""

    _validate_feature_for_write(feature)
    if any(separator in feature.subject_id for separator in ("/", "\\")):
        raise PSDFeatureError(f"subject_id must not contain path separators: {feature.subject_id!r}")

    feature_dir = Path(output_root) / "data" / "features" / "psd"
    feature_dir.mkdir(parents=True, exist_ok=True)
    output_path = feature_dir / f"{feature.subject_id}_{feature.state}_psd.npz"
    np.savez_compressed(
        output_path,
        psd=feature.psd,
        frequency_bins=feature.frequency_bins,
        subject_id=np.array(feature.subject_id),
        state=np.array(feature.state),
        channel_names_after_alignment=np.array(feature.channel_names_after_alignment),
        sampling_rate=np.array(feature.sampling_rate),
        source_set_path=np.array(feature.source_set_path.as_posix()),
        **{key: np.array(value) for key, value in feature.metadata.items()},
    )
    return output_path


def _validate_feature_for_write(feature: PSDFeature) -> None:
    _validate_state(feature.state)
    if feature.psd.shape != (62, 90):
        raise PSDFeatureError(f"PSD shape must be (62, 90), got {feature.psd.shape}")
    if feature.frequency_bins.shape != (90,) or not np.allclose(
        feature.frequency_bins,
        target_frequency_bins(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise PSDFeatureError("frequency_bins must match the fixed 0.5-45 Hz target bins")
    if tuple(feature.channel_names_after_alignment) != CANONICAL_CHANNELS_62:
        raise PSDFeatureError("channel_names_after_alignment must match the fixed 62-channel order")


def compute_psd_for_eeg_record(
    record: EEGFileRecord,
    affected_hand: str,
    output_root: str | Path | None = None,
    config: PSDConfig | None = None,
) -> PSDFeature:
    """Read one indexed EEGLAB record, compute PSD, and optionally write it."""

    metadata = read_eeglab_set_metadata(record.set_path)
    data = read_eeglab_fdt(metadata, subject_id=record.subject_id, state=record.state)
    feature = compute_single_state_psd(
        data,
        metadata.srate,
        metadata.ch_names,
        affected_hand,
        record.subject_id,
        record.state,
        metadata.set_path,
        config=config,
    )
    if output_root is not None:
        write_psd_feature(feature, output_root)
    return feature


def _validate_data_contract(
    data: np.ndarray,
    sampling_rate: float,
    channel_names: Sequence[str],
) -> None:
    if data.ndim != 2:
        raise PSDFeatureError(f"data must be channels x samples, got shape {data.shape}")
    if tuple(channel_names) != CANONICAL_CHANNELS_62:
        raise ChannelMappingError("PSD input channel_names must match the fixed 62-channel order")
    if data.shape[0] != len(CANONICAL_CHANNELS_62):
        raise PSDFeatureError(f"Expected 62 channels, got {data.shape[0]}")
    if not np.isfinite(data).all():
        raise PSDFeatureError("data contains NaN or infinite values")
    if float(sampling_rate) <= 0:
        raise PSDFeatureError(f"sampling_rate must be positive, got {sampling_rate}")


def _validate_state(state: str) -> None:
    if state not in {"EO", "EC"}:
        raise PSDFeatureError(f"state must be 'EO' or 'EC', got {state!r}")


def _validate_welch_settings(data: np.ndarray, nperseg: int, noverlap: int) -> None:
    if nperseg <= 0:
        raise PSDFeatureError(f"Welch nperseg must be positive, got {nperseg}")
    if data.shape[1] < nperseg:
        raise PSDFeatureError(
            f"Need at least {nperseg} samples for fixed-resolution Welch PSD, got {data.shape[1]}"
        )
    if noverlap < 0 or noverlap >= nperseg:
        raise PSDFeatureError(
            f"Welch noverlap must be in [0, nperseg), got noverlap={noverlap}, nperseg={nperseg}"
        )


def _target_frequency_indices(frequencies: np.ndarray, expected_bins: np.ndarray) -> np.ndarray:
    indices: list[int] = []
    for target in expected_bins:
        matches = np.flatnonzero(np.isclose(frequencies, target, rtol=0.0, atol=1e-9))
        if matches.size != 1:
            raise PSDFeatureError(f"Welch output does not contain target frequency bin {target} Hz")
        indices.append(int(matches[0]))
    return np.array(indices, dtype=int)
