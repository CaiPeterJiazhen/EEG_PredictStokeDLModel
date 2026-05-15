from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import stft

from eeg_recovery.channels.hemisphere_flip import flip_channels_for_affected_hand
from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62, ChannelMappingError
from eeg_recovery.io.eeglab import read_eeglab_fdt, read_eeglab_set_metadata
from eeg_recovery.io.index import EEGFileRecord


CONNECTIVITY_BANDS = (
    ("Delta", (1.0, 3.0)),
    ("Theta", (4.0, 7.0)),
    ("Alpha", (8.0, 13.0)),
    ("Beta Low", (13.0, 18.0)),
    ("Beta Medium", (18.0, 21.0)),
    ("Beta High", (21.0, 30.0)),
)


class ConnectivityFeatureError(ValueError):
    """Raised when FC feature inputs or outputs violate the project contract."""


@dataclass(frozen=True)
class ConnectivityConfig:
    nperseg: int | None = None
    noverlap: int | None = None
    window: str = "hann"
    detrend: str | bool = "constant"

    def stft_nperseg(self, sampling_rate: float) -> int:
        if self.nperseg is not None:
            return int(self.nperseg)
        return int(round(float(sampling_rate) * 2.0))

    def stft_noverlap(self, sampling_rate: float) -> int:
        if self.noverlap is not None:
            return int(self.noverlap)
        return self.stft_nperseg(sampling_rate) // 2


@dataclass(frozen=True)
class ConnectivityFeature:
    wpli: np.ndarray
    imaginary_coherence: np.ndarray
    edge_list: tuple[tuple[str, str], ...]
    band_names: tuple[str, ...]
    band_ranges_hz: tuple[tuple[float, float], ...]
    subject_id: str
    state: str
    channel_names_after_alignment: tuple[str, ...]
    sampling_rate: float
    source_set_path: Path
    metadata: dict[str, Any]


def connectivity_bands() -> tuple[tuple[str, tuple[float, float]], ...]:
    """Return the fixed six band definitions used for FC features."""

    return CONNECTIVITY_BANDS


def build_edge_list(
    channel_names: Sequence[str] = CANONICAL_CHANNELS_62,
) -> tuple[tuple[str, str], ...]:
    """Build deterministic upper-triangle channel pairs from canonical order."""

    names = tuple(channel_names)
    if names != CANONICAL_CHANNELS_62:
        raise ChannelMappingError("FC edge list channel_names must match the fixed 62-channel order")

    edges: list[tuple[str, str]] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            edges.append((first, second))
    return tuple(edges)


def compute_single_state_connectivity(
    data: np.ndarray,
    sampling_rate: float,
    channel_names: Sequence[str],
    affected_hand: str,
    subject_id: str,
    state: str,
    source_set_path: str | Path,
    config: ConnectivityConfig | None = None,
) -> ConnectivityFeature:
    """Compute one EO or EC FC matrix after affected-hand channel alignment."""

    resolved = config if config is not None else ConnectivityConfig()
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
    nperseg = resolved.stft_nperseg(sampling_rate)
    noverlap = resolved.stft_noverlap(sampling_rate)
    _validate_stft_settings(aligned, nperseg, noverlap)

    frequencies, _, spectra = stft(
        aligned,
        fs=float(sampling_rate),
        window=resolved.window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=resolved.detrend,
        boundary=None,
        padded=False,
        axis=1,
    )
    wpli, imaginary_coherence = _compute_band_connectivity(spectra, frequencies)

    edge_list = build_edge_list(CANONICAL_CHANNELS_62)
    band_names = tuple(name for name, _ in CONNECTIVITY_BANDS)
    band_ranges = tuple(ranges for _, ranges in CONNECTIVITY_BANDS)
    _validate_feature_arrays(wpli, imaginary_coherence, edge_list, band_names, band_ranges)

    metadata = {
        "stft_nperseg": nperseg,
        "stft_noverlap": noverlap,
        "stft_window": resolved.window,
        "stft_detrend": resolved.detrend,
        "affected_hand": normalized_hand,
        "hemisphere_aligned": True,
    }
    return ConnectivityFeature(
        wpli=wpli,
        imaginary_coherence=imaginary_coherence,
        edge_list=edge_list,
        band_names=band_names,
        band_ranges_hz=band_ranges,
        subject_id=subject_id,
        state=state,
        channel_names_after_alignment=tuple(CANONICAL_CHANNELS_62),
        sampling_rate=float(sampling_rate),
        source_set_path=Path(source_set_path),
        metadata=metadata,
    )


def write_connectivity_feature(feature: ConnectivityFeature, output_root: str | Path) -> Path:
    """Write a single-state FC feature under output_root/data/features/fc."""

    _validate_feature_for_write(feature)
    if any(separator in feature.subject_id for separator in ("/", "\\")):
        raise ConnectivityFeatureError(
            f"subject_id must not contain path separators: {feature.subject_id!r}"
        )

    feature_dir = Path(output_root) / "data" / "features" / "fc"
    feature_dir.mkdir(parents=True, exist_ok=True)
    output_path = feature_dir / f"{feature.subject_id}_{feature.state}_fc.npz"
    np.savez_compressed(
        output_path,
        wpli=feature.wpli,
        imaginary_coherence=feature.imaginary_coherence,
        edge_list=np.array(feature.edge_list),
        band_names=np.array(feature.band_names),
        band_ranges_hz=np.array(feature.band_ranges_hz, dtype=float),
        subject_id=np.array(feature.subject_id),
        state=np.array(feature.state),
        channel_names_after_alignment=np.array(feature.channel_names_after_alignment),
        sampling_rate=np.array(feature.sampling_rate),
        source_set_path=np.array(feature.source_set_path.as_posix()),
        **{key: np.array(value) for key, value in feature.metadata.items()},
    )
    return output_path


def compute_connectivity_for_eeg_record(
    record: EEGFileRecord,
    affected_hand: str,
    output_root: str | Path | None = None,
    config: ConnectivityConfig | None = None,
) -> ConnectivityFeature:
    """Read one indexed EEGLAB record, compute FC, and optionally write it."""

    metadata = read_eeglab_set_metadata(record.set_path)
    data = read_eeglab_fdt(metadata, subject_id=record.subject_id, state=record.state)
    feature = compute_single_state_connectivity(
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
        write_connectivity_feature(feature, output_root)
    return feature


def _compute_band_connectivity(
    spectra: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edges = _edge_index_pairs()
    wpli = np.zeros((len(edges), len(CONNECTIVITY_BANDS)), dtype=float)
    imaginary_coherence = np.zeros_like(wpli)

    for band_index, (_, (low_hz, high_hz)) in enumerate(CONNECTIVITY_BANDS):
        frequency_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
        if not np.any(frequency_mask):
            raise ConnectivityFeatureError(
                f"STFT output contains no frequency bins for band {low_hz}-{high_hz} Hz"
            )

        band_spectra = spectra[:, frequency_mask, :].reshape(spectra.shape[0], -1)
        powers = np.mean(np.abs(band_spectra) ** 2, axis=1)
        for edge_index, (first, second) in enumerate(edges):
            cross = band_spectra[first] * np.conj(band_spectra[second])
            imag_cross = np.imag(cross)
            wpli_denominator = np.mean(np.abs(imag_cross))
            if wpli_denominator > 0:
                wpli[edge_index, band_index] = abs(np.mean(imag_cross)) / wpli_denominator

            mean_cross = np.mean(cross)
            icoh_denominator = np.sqrt(powers[first] * powers[second])
            if icoh_denominator > 0:
                imaginary_coherence[edge_index, band_index] = (
                    np.imag(mean_cross) / icoh_denominator
                )

    return np.nan_to_num(wpli, copy=False), np.nan_to_num(imaginary_coherence, copy=False)


def _edge_index_pairs() -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for first in range(len(CANONICAL_CHANNELS_62)):
        for second in range(first + 1, len(CANONICAL_CHANNELS_62)):
            pairs.append((first, second))
    return tuple(pairs)


def _validate_feature_for_write(feature: ConnectivityFeature) -> None:
    _validate_state(feature.state)
    expected_edges = build_edge_list()
    expected_band_names = tuple(name for name, _ in CONNECTIVITY_BANDS)
    expected_band_ranges = tuple(ranges for _, ranges in CONNECTIVITY_BANDS)
    _validate_feature_arrays(
        feature.wpli,
        feature.imaginary_coherence,
        feature.edge_list,
        feature.band_names,
        feature.band_ranges_hz,
    )
    if feature.edge_list != expected_edges:
        raise ConnectivityFeatureError("edge_list must match the fixed upper-triangle order")
    if feature.band_names != expected_band_names:
        raise ConnectivityFeatureError("band_names must match the fixed FC band order")
    if feature.band_ranges_hz != expected_band_ranges:
        raise ConnectivityFeatureError("band_ranges_hz must match the fixed FC band ranges")
    if tuple(feature.channel_names_after_alignment) != CANONICAL_CHANNELS_62:
        raise ConnectivityFeatureError(
            "channel_names_after_alignment must match the fixed 62-channel order"
        )


def _validate_feature_arrays(
    wpli: np.ndarray,
    imaginary_coherence: np.ndarray,
    edge_list: Sequence[tuple[str, str]],
    band_names: Sequence[str],
    band_ranges_hz: Sequence[tuple[float, float]],
) -> None:
    if len(edge_list) != 1891:
        raise ConnectivityFeatureError(f"Expected 1891 FC edges, got {len(edge_list)}")
    if tuple(band_names) != tuple(name for name, _ in CONNECTIVITY_BANDS):
        raise ConnectivityFeatureError("band_names must match the fixed FC band order")
    if tuple(band_ranges_hz) != tuple(ranges for _, ranges in CONNECTIVITY_BANDS):
        raise ConnectivityFeatureError("band_ranges_hz must match the fixed FC band ranges")
    if wpli.shape != (1891, 6):
        raise ConnectivityFeatureError(f"wPLI shape must be (1891, 6), got {wpli.shape}")
    if imaginary_coherence.shape != (1891, 6):
        raise ConnectivityFeatureError(
            "imaginary_coherence shape must be (1891, 6), "
            f"got {imaginary_coherence.shape}"
        )
    if not np.isfinite(wpli).all():
        raise ConnectivityFeatureError("wPLI contains NaN or infinite values")
    if not np.isfinite(imaginary_coherence).all():
        raise ConnectivityFeatureError("imaginary_coherence contains NaN or infinite values")


def _validate_data_contract(
    data: np.ndarray,
    sampling_rate: float,
    channel_names: Sequence[str],
) -> None:
    if data.ndim != 2:
        raise ConnectivityFeatureError(f"data must be channels x samples, got shape {data.shape}")
    if tuple(channel_names) != CANONICAL_CHANNELS_62:
        raise ChannelMappingError("FC input channel_names must match the fixed 62-channel order")
    if data.shape[0] != len(CANONICAL_CHANNELS_62):
        raise ConnectivityFeatureError(f"Expected 62 channels, got {data.shape[0]}")
    if not np.isfinite(data).all():
        raise ConnectivityFeatureError("data contains NaN or infinite values")
    if float(sampling_rate) <= 0:
        raise ConnectivityFeatureError(f"sampling_rate must be positive, got {sampling_rate}")


def _validate_state(state: str) -> None:
    if state not in {"EO", "EC"}:
        raise ConnectivityFeatureError(f"state must be 'EO' or 'EC', got {state!r}")


def _validate_stft_settings(data: np.ndarray, nperseg: int, noverlap: int) -> None:
    if nperseg <= 0:
        raise ConnectivityFeatureError(f"STFT nperseg must be positive, got {nperseg}")
    if data.shape[1] < nperseg:
        raise ConnectivityFeatureError(
            f"Need at least {nperseg} samples for fixed-window STFT FC, got {data.shape[1]}"
        )
    if noverlap < 0 or noverlap >= nperseg:
        raise ConnectivityFeatureError(
            f"STFT noverlap must be in [0, nperseg), got noverlap={noverlap}, nperseg={nperseg}"
        )
