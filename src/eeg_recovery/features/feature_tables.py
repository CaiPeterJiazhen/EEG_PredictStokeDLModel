from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.features.connectivity import build_edge_list, connectivity_bands
from eeg_recovery.metadata.subjects import normalize_subject_id


STATES = ("EO", "EC")


class FeatureTableError(ValueError):
    """Raised when saved feature files cannot be converted to tabular features."""


def band_definitions() -> tuple[tuple[str, tuple[float, float]], ...]:
    """Return the fixed six EEG bands used by PSD and FC tabular baselines."""

    return connectivity_bands()


def load_psd_band_power_table(
    psd_dir: str | Path,
    subject_ids: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Load EO/EC PSD files and average frequency bins into band-power features."""

    root = Path(psd_dir)
    subjects = _resolve_subject_ids(root, "_psd.npz", subject_ids)
    rows: list[dict[str, float | str]] = []
    for subject_id in subjects:
        row: dict[str, float | str] = {"subject_id": subject_id}
        for state in STATES:
            payload = _load_required_npz(root / f"{subject_id}_{state}_psd.npz")
            if "psd" not in payload:
                raise FeatureTableError(f"PSD file is missing 'psd': {subject_id}_{state}_psd.npz")
            psd = np.asarray(payload["psd"], dtype=float)
            if psd.ndim != 2:
                raise FeatureTableError(f"Expected 2D PSD array for {subject_id} {state}, got {psd.shape}")
            if psd.shape[0] != len(CANONICAL_CHANNELS_62):
                raise FeatureTableError(f"Expected 62 PSD channels for {subject_id} {state}, got {psd.shape}")
            if not np.isfinite(psd).all():
                raise FeatureTableError(f"PSD contains NaN or infinite values for {subject_id} {state}")
            frequencies = _frequency_bins_from_payload(payload, psd.shape[1])
            for band_name, (low_hz, high_hz) in band_definitions():
                mask = (frequencies >= low_hz) & (frequencies <= high_hz)
                if not np.any(mask):
                    raise FeatureTableError(
                        f"PSD file for {subject_id} {state} has no bins in {band_name} band"
                    )
                band_power = psd[:, mask].mean(axis=1)
                for channel_name, value in zip(CANONICAL_CHANNELS_62, band_power, strict=True):
                    row[f"psd_{state}_{channel_name}_{_slug(band_name)}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def load_fc_feature_table(
    fc_dir: str | Path,
    metric: str = "wpli",
    subject_ids: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Load EO/EC FC files and flatten one FC metric into patient-level rows."""

    if metric not in {"wpli", "imaginary_coherence"}:
        raise FeatureTableError("metric must be 'wpli' or 'imaginary_coherence'")

    root = Path(fc_dir)
    subjects = _resolve_subject_ids(root, "_fc.npz", subject_ids)
    band_names = tuple(name for name, _ in band_definitions())
    edge_list = build_edge_list()
    rows: list[dict[str, float | str]] = []
    for subject_id in subjects:
        row: dict[str, float | str] = {"subject_id": subject_id}
        for state in STATES:
            payload = _load_required_npz(root / f"{subject_id}_{state}_fc.npz")
            if metric not in payload:
                raise FeatureTableError(f"FC file is missing {metric!r}: {subject_id}_{state}_fc.npz")
            values = np.asarray(payload[metric], dtype=float)
            if values.shape != (1891, 6):
                raise FeatureTableError(
                    f"Expected {metric} shape (1891, 6) for {subject_id} {state}, got {values.shape}"
                )
            for edge_index, (first, second) in enumerate(edge_list):
                for band_index, band_name in enumerate(band_names):
                    column = (
                        f"fc_{metric}_{state}_edge{edge_index:04d}_"
                        f"{first}-{second}_{_slug(band_name)}"
                    )
                    row[column] = float(values[edge_index, band_index])
        rows.append(row)
    return pd.DataFrame(rows)


def merge_feature_tables(*tables: pd.DataFrame) -> pd.DataFrame:
    """Inner-join one or more feature tables on subject_id."""

    if not tables:
        raise FeatureTableError("At least one feature table is required.")
    merged = tables[0].copy()
    if "subject_id" not in merged.columns:
        raise FeatureTableError("Feature table is missing required subject_id column.")
    for table in tables[1:]:
        if "subject_id" not in table.columns:
            raise FeatureTableError("Feature table is missing required subject_id column.")
        merged = merged.merge(table, on="subject_id", how="inner", validate="one_to_one")
    return merged


join_feature_tables = merge_feature_tables


def _resolve_subject_ids(
    root: Path,
    suffix: str,
    subject_ids: Iterable[object] | None,
) -> list[str]:
    if subject_ids is not None:
        subjects = [normalize_subject_id(subject_id) for subject_id in subject_ids]
    else:
        subjects = sorted(
            {
                normalize_subject_id(path.name.split("_", 1)[0])
                for path in root.glob(f"*{suffix}")
            }
        )
    if not subjects:
        raise FeatureTableError(f"No feature files found in {root}")
    for subject_id in subjects:
        for state in STATES:
            state_suffix = suffix.replace(".npz", "")
            expected = root / f"{subject_id}_{state}{state_suffix}.npz"
            if not expected.exists():
                raise FeatureTableError(f"Missing required {state} feature file for {subject_id}: {expected}")
    return subjects


def _load_required_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FeatureTableError(f"Feature file does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.array(payload[key], copy=True) for key in payload.files}


def _frequency_bins_from_payload(payload: dict[str, Any], n_bins: int) -> np.ndarray:
    if "frequency_bins" in payload:
        frequencies = np.asarray(payload["frequency_bins"], dtype=float)
    else:
        frequencies = np.arange(1, n_bins + 1, dtype=float) * 0.5
    if frequencies.ndim != 1:
        raise FeatureTableError(f"frequency_bins must be 1D, got shape {frequencies.shape}")
    if frequencies.shape != (n_bins,):
        raise FeatureTableError(f"frequency_bins shape must be ({n_bins},), got {frequencies.shape}")
    if not np.isfinite(frequencies).all():
        raise FeatureTableError("frequency_bins contains NaN or infinite values")
    return frequencies


def _slug(value: str) -> str:
    return "_".join(value.split())
