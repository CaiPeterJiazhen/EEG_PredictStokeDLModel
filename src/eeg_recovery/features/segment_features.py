from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
from typing import Any, Iterable

import numpy as np

from eeg_recovery.features.connectivity import (
    ConnectivityConfig,
    build_edge_list,
    compute_single_state_connectivity,
    connectivity_bands,
)
from eeg_recovery.features.psd import PSDConfig, compute_single_state_psd, target_frequency_bins
from eeg_recovery.io.eeglab import read_eeglab_fdt, read_eeglab_set_metadata
from eeg_recovery.io.index import EEGFileRecord


class SegmentFeatureError(ValueError):
    """Raised when segment-level feature inputs or cache files are invalid."""


@dataclass(frozen=True)
class SegmentFeatureConfig:
    psd_window_seconds: float = 4.0
    psd_overlap_fraction: float = 0.5
    fc_window_seconds: float = 8.0
    fc_overlap_fraction: float = 0.5
    compute_fc: bool = False
    psd_config: PSDConfig = PSDConfig()
    fc_config: ConnectivityConfig = ConnectivityConfig()


@dataclass(frozen=True)
class SegmentWindow:
    segment_index: int
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class SegmentPSDFeature:
    psd: np.ndarray
    frequency_bins: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SegmentFCFeature:
    wpli: np.ndarray
    imaginary_coherence: np.ndarray
    edge_list: np.ndarray
    band_names: np.ndarray
    band_ranges_hz: np.ndarray
    metadata: dict[str, Any]


def slice_fixed_windows(
    *,
    n_samples: int,
    sampling_rate: float,
    window_seconds: float,
    overlap_fraction: float,
) -> list[SegmentWindow]:
    """Return deterministic fixed-length sample windows for one continuous EEG trace."""

    if n_samples < 1:
        raise SegmentFeatureError("n_samples must be positive.")
    if sampling_rate <= 0:
        raise SegmentFeatureError("sampling_rate must be positive.")
    if window_seconds <= 0:
        raise SegmentFeatureError("window_seconds must be positive.")
    if not 0 <= overlap_fraction < 1:
        raise SegmentFeatureError("overlap_fraction must be in [0, 1).")

    window_samples = int(round(float(window_seconds) * float(sampling_rate)))
    if window_samples <= 0:
        raise SegmentFeatureError("window_seconds resolves to fewer than one sample.")
    if n_samples < window_samples:
        raise SegmentFeatureError(
            f"Need at least {window_samples} samples for {window_seconds:g}s windows, got {n_samples}."
        )
    step_samples = max(1, int(round(window_samples * (1.0 - overlap_fraction))))

    windows: list[SegmentWindow] = []
    start = 0
    while start + window_samples <= n_samples:
        windows.append(
            SegmentWindow(
                segment_index=len(windows),
                start_sample=start,
                end_sample=start + window_samples,
            )
        )
        start += step_samples
    return windows


def compute_segment_psd_features(
    *,
    data: np.ndarray,
    sampling_rate: float,
    channel_names: Iterable[str],
    affected_hand: str,
    subject_id: str,
    subject_key: str,
    group: str,
    stage: str,
    state: str,
    source_set_path: str | Path,
    source_fdt_path: str | Path | None = None,
    config: SegmentFeatureConfig | None = None,
) -> list[SegmentPSDFeature]:
    """Split one continuous EO/EC recording into segments and compute PSD features."""

    resolved = config if config is not None else SegmentFeatureConfig()
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise SegmentFeatureError(f"data must be channels x samples, got {array.shape}.")

    windows = slice_fixed_windows(
        n_samples=array.shape[1],
        sampling_rate=sampling_rate,
        window_seconds=resolved.psd_window_seconds,
        overlap_fraction=resolved.psd_overlap_fraction,
    )
    source_set = Path(source_set_path)
    source_fdt = Path(source_fdt_path) if source_fdt_path is not None else None
    source_stats = _source_stats(source_set, source_fdt)
    normalized_hand = _normalize_affected_hand(affected_hand)

    features: list[SegmentPSDFeature] = []
    for window in windows:
        segment = array[:, window.start_sample : window.end_sample]
        psd_feature = compute_single_state_psd(
            segment,
            sampling_rate,
            tuple(channel_names),
            normalized_hand,
            subject_id,
            state,
            source_set,
            config=resolved.psd_config,
        )
        metadata = {
            "subject_id": subject_id,
            "subject_key": subject_key,
            "group": group,
            "stage": stage,
            "state": state,
            "segment_index": window.segment_index,
            "start_sample": window.start_sample,
            "end_sample": window.end_sample,
            "affected_hand": normalized_hand,
            "affected_hand_aligned": bool(psd_feature.metadata["hemisphere_aligned"]),
            "source_set_path": str(source_set),
            "source_fdt_path": "" if source_fdt is None else str(source_fdt),
            "sampling_rate": float(sampling_rate),
            "psd_window_seconds": float(resolved.psd_window_seconds),
            "psd_overlap_fraction": float(resolved.psd_overlap_fraction),
            "frequency_resolution_hz": float(resolved.psd_config.frequency_resolution_hz),
            "cache_fingerprint": _cache_fingerprint(source_set, source_fdt, window),
            **source_stats,
        }
        features.append(
            SegmentPSDFeature(
                psd=psd_feature.psd.astype(np.float32, copy=False),
                frequency_bins=psd_feature.frequency_bins.astype(np.float32, copy=False),
                metadata=metadata,
            )
        )
    return features


def compute_segment_fc_features(
    *,
    data: np.ndarray,
    sampling_rate: float,
    channel_names: Iterable[str],
    affected_hand: str,
    subject_id: str,
    subject_key: str,
    group: str,
    stage: str,
    state: str,
    source_set_path: str | Path,
    source_fdt_path: str | Path | None = None,
    config: SegmentFeatureConfig | None = None,
) -> list[SegmentFCFeature]:
    """Split one continuous EO/EC recording into segments and compute wPLI/iCoh features."""

    resolved = config if config is not None else SegmentFeatureConfig()
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise SegmentFeatureError(f"data must be channels x samples, got {array.shape}.")

    windows = slice_fixed_windows(
        n_samples=array.shape[1],
        sampling_rate=sampling_rate,
        window_seconds=resolved.fc_window_seconds,
        overlap_fraction=resolved.fc_overlap_fraction,
    )
    source_set = Path(source_set_path)
    source_fdt = Path(source_fdt_path) if source_fdt_path is not None else None
    source_stats = _source_stats(source_set, source_fdt)
    normalized_hand = _normalize_affected_hand(affected_hand)

    features: list[SegmentFCFeature] = []
    for window in windows:
        segment = array[:, window.start_sample : window.end_sample]
        fc_feature = compute_single_state_connectivity(
            segment,
            sampling_rate,
            tuple(channel_names),
            normalized_hand,
            subject_id,
            state,
            source_set,
            config=resolved.fc_config,
        )
        metadata = {
            "subject_id": subject_id,
            "subject_key": subject_key,
            "group": group,
            "stage": stage,
            "state": state,
            "segment_index": window.segment_index,
            "start_sample": window.start_sample,
            "end_sample": window.end_sample,
            "affected_hand": normalized_hand,
            "affected_hand_aligned": bool(fc_feature.metadata["hemisphere_aligned"]),
            "source_set_path": str(source_set),
            "source_fdt_path": "" if source_fdt is None else str(source_fdt),
            "sampling_rate": float(sampling_rate),
            "fc_window_seconds": float(resolved.fc_window_seconds),
            "fc_overlap_fraction": float(resolved.fc_overlap_fraction),
            "stft_nperseg": int(fc_feature.metadata["stft_nperseg"]),
            "stft_noverlap": int(fc_feature.metadata["stft_noverlap"]),
            "stft_window": str(fc_feature.metadata["stft_window"]),
            "stft_detrend": str(fc_feature.metadata["stft_detrend"]),
            "cache_fingerprint": _cache_fingerprint(source_set, source_fdt, window),
            **source_stats,
        }
        features.append(
            SegmentFCFeature(
                wpli=fc_feature.wpli.astype(np.float32, copy=False),
                imaginary_coherence=fc_feature.imaginary_coherence.astype(np.float32, copy=False),
                edge_list=np.asarray(fc_feature.edge_list),
                band_names=np.asarray(fc_feature.band_names),
                band_ranges_hz=np.asarray(fc_feature.band_ranges_hz, dtype=np.float32),
                metadata=metadata,
            )
        )
    return features


def compute_segment_psd_for_eeg_record(
    record: EEGFileRecord,
    affected_hand: str,
    *,
    output_root: str | Path | None = None,
    config: SegmentFeatureConfig | None = None,
    force: bool = False,
) -> list[SegmentPSDFeature]:
    """Read one indexed EEGLAB record, compute PSD segments, and optionally cache them."""

    metadata = read_eeglab_set_metadata(record.set_path)
    data = read_eeglab_fdt(metadata, subject_id=record.subject_id, state=record.state)
    features = compute_segment_psd_features(
        data=data,
        sampling_rate=metadata.srate,
        channel_names=metadata.ch_names,
        affected_hand=affected_hand,
        subject_id=record.subject_id,
        subject_key=record.subject_key,
        group=record.group,
        stage=record.stage,
        state=record.state,
        source_set_path=metadata.set_path,
        source_fdt_path=metadata.fdt_path,
        config=config,
    )
    if output_root is not None:
        for feature in features:
            output_path = segment_cache_path(output_root, "psd", feature.metadata)
            if force or not is_segment_cache_fresh(output_path):
                write_segment_psd_feature(feature, output_root)
    return features


def compute_segment_fc_for_eeg_record(
    record: EEGFileRecord,
    affected_hand: str,
    *,
    output_root: str | Path | None = None,
    config: SegmentFeatureConfig | None = None,
    force: bool = False,
) -> list[SegmentFCFeature]:
    """Read one indexed EEGLAB record, compute FC/wPLI segments, and optionally cache them."""

    metadata = read_eeglab_set_metadata(record.set_path)
    data = read_eeglab_fdt(metadata, subject_id=record.subject_id, state=record.state)
    features = compute_segment_fc_features(
        data=data,
        sampling_rate=metadata.srate,
        channel_names=metadata.ch_names,
        affected_hand=affected_hand,
        subject_id=record.subject_id,
        subject_key=record.subject_key,
        group=record.group,
        stage=record.stage,
        state=record.state,
        source_set_path=metadata.set_path,
        source_fdt_path=metadata.fdt_path,
        config=config,
    )
    if output_root is not None:
        for feature in features:
            output_path = segment_cache_path(output_root, "fc", feature.metadata)
            if force or not is_segment_cache_fresh(output_path):
                write_segment_fc_feature(feature, output_root)
    return features


def write_segment_psd_feature(feature: SegmentPSDFeature, output_root: str | Path) -> Path:
    """Write a single PSD segment under data/features/segment_level/psd."""

    if feature.psd.shape != (62, 90):
        raise SegmentFeatureError(f"PSD segment shape must be (62, 90), got {feature.psd.shape}.")
    if feature.frequency_bins.shape != (90,) or not np.allclose(
        feature.frequency_bins,
        target_frequency_bins(),
        rtol=0.0,
        atol=1e-6,
    ):
        raise SegmentFeatureError("frequency_bins must match the fixed 0.5-45 Hz target bins.")

    output_path = segment_cache_path(output_root, "psd", feature.metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "psd": feature.psd.astype(np.float32, copy=False),
        "frequency_bins": feature.frequency_bins.astype(np.float32, copy=False),
    }
    for key, value in feature.metadata.items():
        payload[key] = np.array(value)
    np.savez_compressed(output_path, **payload)
    return output_path


def write_segment_fc_feature(feature: SegmentFCFeature, output_root: str | Path) -> Path:
    """Write a single FC segment under data/features/segment_level/fc."""

    if feature.wpli.shape != (1891, 6):
        raise SegmentFeatureError(f"wPLI segment shape must be (1891, 6), got {feature.wpli.shape}.")
    if feature.imaginary_coherence.shape != (1891, 6):
        raise SegmentFeatureError(
            "imaginary_coherence segment shape must be (1891, 6), "
            f"got {feature.imaginary_coherence.shape}."
        )
    expected_edges = np.asarray(build_edge_list())
    if feature.edge_list.shape != expected_edges.shape or not np.array_equal(feature.edge_list, expected_edges):
        raise SegmentFeatureError("edge_list must match the fixed 62-channel upper-triangle order.")
    expected_band_names = np.asarray([name for name, _ in connectivity_bands()])
    expected_band_ranges = np.asarray([ranges for _, ranges in connectivity_bands()], dtype=np.float32)
    if feature.band_names.shape != expected_band_names.shape or not np.array_equal(feature.band_names, expected_band_names):
        raise SegmentFeatureError("band_names must match the fixed FC band order.")
    if feature.band_ranges_hz.shape != expected_band_ranges.shape or not np.allclose(
        feature.band_ranges_hz,
        expected_band_ranges,
        rtol=0.0,
        atol=1e-6,
    ):
        raise SegmentFeatureError("band_ranges_hz must match the fixed FC band ranges.")

    output_path = segment_cache_path(output_root, "fc", feature.metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wpli": feature.wpli.astype(np.float32, copy=False),
        "imaginary_coherence": feature.imaginary_coherence.astype(np.float32, copy=False),
        "edge_list": feature.edge_list,
        "band_names": feature.band_names,
        "band_ranges_hz": feature.band_ranges_hz.astype(np.float32, copy=False),
    }
    for key, value in feature.metadata.items():
        payload[key] = np.array(value)
    np.savez_compressed(output_path, **payload)
    return output_path


def segment_cache_path(
    output_root: str | Path,
    modality: str,
    metadata: dict[str, Any],
) -> Path:
    if modality not in {"psd", "fc"}:
        raise SegmentFeatureError("modality must be 'psd' or 'fc'.")
    subject_key = _safe_token(str(metadata["subject_key"]))
    stage = _safe_token(str(metadata["stage"]))
    state = _safe_token(str(metadata["state"]))
    segment_index = int(metadata["segment_index"])
    suffix = "psd" if modality == "psd" else "fc"
    filename = f"{subject_key}_{stage}_{state}_seg{segment_index:04d}_{suffix}.npz"
    return Path(output_root) / "data" / "features" / "segment_level" / modality / filename


def is_segment_cache_fresh(path: str | Path) -> bool:
    """Return True only when a cache file exists and its recorded source mtimes match."""

    cache_path = Path(path)
    if not cache_path.exists():
        return False
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            source_set = Path(str(payload["source_set_path"].item()))
            source_fdt_raw = str(payload["source_fdt_path"].item()) if "source_fdt_path" in payload else ""
            source_fdt = Path(source_fdt_raw) if source_fdt_raw else None
            stored_set_mtime = int(payload["source_set_mtime_ns"].item())
            stored_set_size = int(payload["source_set_size"].item())
            stored_fdt_mtime = int(payload["source_fdt_mtime_ns"].item()) if "source_fdt_mtime_ns" in payload else -1
            stored_fdt_size = int(payload["source_fdt_size"].item()) if "source_fdt_size" in payload else -1
    except Exception:
        return False

    if not source_set.exists():
        return False
    set_stat = source_set.stat()
    if int(set_stat.st_mtime_ns) != stored_set_mtime or int(set_stat.st_size) != stored_set_size:
        return False
    if source_fdt is not None:
        if not source_fdt.exists():
            return False
        fdt_stat = source_fdt.stat()
        if int(fdt_stat.st_mtime_ns) != stored_fdt_mtime or int(fdt_stat.st_size) != stored_fdt_size:
            return False
    return True


def _source_stats(source_set: Path, source_fdt: Path | None) -> dict[str, int]:
    if not source_set.exists():
        raise SegmentFeatureError(f"source_set_path does not exist: {source_set}")
    set_stat = source_set.stat()
    stats = {
        "source_set_mtime_ns": int(set_stat.st_mtime_ns),
        "source_set_size": int(set_stat.st_size),
    }
    if source_fdt is not None:
        if not source_fdt.exists():
            raise SegmentFeatureError(f"source_fdt_path does not exist: {source_fdt}")
        fdt_stat = source_fdt.stat()
        stats["source_fdt_mtime_ns"] = int(fdt_stat.st_mtime_ns)
        stats["source_fdt_size"] = int(fdt_stat.st_size)
    else:
        stats["source_fdt_mtime_ns"] = -1
        stats["source_fdt_size"] = -1
    return stats


def _cache_fingerprint(source_set: Path, source_fdt: Path | None, window: SegmentWindow) -> str:
    digest = hashlib.sha256()
    for path in (source_set, source_fdt):
        if path is None:
            continue
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(int(stat.st_size)).encode("ascii"))
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    digest.update(str(window.start_sample).encode("ascii"))
    digest.update(str(window.end_sample).encode("ascii"))
    return digest.hexdigest()


def _normalize_affected_hand(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"right", "r", "healthy", "none", "na"}:
        return "\u53f3"
    if normalized in {"left", "l"}:
        return "\u5de6"
    return value.strip()


def _safe_token(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if not safe:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return f"u{digest}"
    return safe
