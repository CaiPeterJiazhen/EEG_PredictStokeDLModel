from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

_SCOPES: tuple[tuple[str, tuple[str, ...] | None], ...] = (
    ("global", None),
    ("frontal", ("FP", "AF", "F", "FC", "FT")),
    ("posterior", ("P", "PO", "O", "CB")),
)


@dataclass(frozen=True)
class QEEGSlowingFeatureVector:
    values: np.ndarray
    feature_names: tuple[str, ...]


def compute_subject_qeeg_slowing_features(output_root: str | Path, subject_id: str) -> QEEGSlowingFeatureVector:
    root = Path(output_root)
    psd_payloads = {
        state: _load_npz(root / "data" / "features" / "psd" / f"{subject_id}_{state}_psd.npz")
        for state in ("EO", "EC")
    }
    return compute_qeeg_slowing_from_psd_payloads(
        psd_eo=np.asarray(psd_payloads["EO"]["psd"], dtype=np.float32),
        psd_ec=np.asarray(psd_payloads["EC"]["psd"], dtype=np.float32),
        frequency_bins=np.asarray(psd_payloads["EO"]["frequency_bins"], dtype=np.float32),
        channel_names=np.asarray(psd_payloads["EO"]["channel_names_after_alignment"]).astype(str),
    )


def compute_qeeg_slowing_from_psd_payloads(
    *,
    psd_eo: np.ndarray,
    psd_ec: np.ndarray,
    frequency_bins: np.ndarray,
    channel_names: np.ndarray,
) -> QEEGSlowingFeatureVector:
    _validate_psd_inputs(psd_eo, psd_ec, frequency_bins, channel_names)
    values: list[float] = []
    names: list[str] = []

    state_summaries: dict[str, dict[str, dict[str, float]]] = {}
    for state_name, psd in (("eo", psd_eo), ("ec", psd_ec)):
        state_summaries[state_name] = {}
        for scope_name, prefixes in _SCOPES:
            indices = _scope_channel_indices(channel_names, prefixes)
            if indices.size == 0:
                continue
            summary = _scope_summary(psd[indices], frequency_bins, channel_names[indices])
            state_summaries[state_name][scope_name] = summary
            for feature_name in (
                "dar",
                "dtabr",
                "slow_burden",
                "alpha_beta_preserved",
                "alpha_peak_hz",
                "alpha_peak_prominence",
                "alpha_centroid_hz",
                "slow_fast_bsi",
            ):
                names.append(f"qeeg_{state_name}_{scope_name}_{feature_name}")
                values.append(summary[feature_name])

    for scope_name in sorted(set(state_summaries["eo"]) & set(state_summaries["ec"])):
        eo_summary = state_summaries["eo"][scope_name]
        ec_summary = state_summaries["ec"][scope_name]
        for feature_name in ("alpha_mean", "dar", "dtabr", "alpha_peak_hz"):
            output_name = "alpha" if feature_name == "alpha_mean" else feature_name
            names.append(f"qeeg_reactivity_{scope_name}_{output_name}_ec_minus_eo")
            values.append(ec_summary[feature_name] - eo_summary[feature_name])

    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("qEEG slowing features contain non-finite values.")
    return QEEGSlowingFeatureVector(values=vector, feature_names=tuple(names))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Required qEEG slowing source file is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _validate_psd_inputs(
    psd_eo: np.ndarray,
    psd_ec: np.ndarray,
    frequency_bins: np.ndarray,
    channel_names: np.ndarray,
) -> None:
    if psd_eo.shape != psd_ec.shape:
        raise ValueError("EO and EC PSD arrays must have the same shape.")
    if psd_eo.ndim != 2:
        raise ValueError("PSD arrays must have shape (channels, frequencies).")
    if psd_eo.shape[0] != len(channel_names):
        raise ValueError("channel_names length must match PSD channel dimension.")
    if psd_eo.shape[1] != len(frequency_bins):
        raise ValueError("frequency_bins length must match PSD frequency dimension.")
    for band_name, (low, high) in _BANDS.items():
        if not _frequency_mask(frequency_bins, low, high).any():
            raise ValueError(f"Missing PSD frequency bins for {band_name} band.")


def _scope_channel_indices(channel_names: np.ndarray, prefixes: tuple[str, ...] | None) -> np.ndarray:
    if prefixes is None:
        return np.arange(len(channel_names), dtype=int)
    selected = [
        index
        for index, channel_name in enumerate(channel_names.astype(str))
        if _channel_token(channel_name).startswith(prefixes)
    ]
    return np.asarray(selected, dtype=int)


def _channel_token(channel_name: str) -> str:
    return "".join(character for character in str(channel_name).upper() if character.isalpha())


def _scope_summary(psd: np.ndarray, frequency_bins: np.ndarray, channel_names: np.ndarray) -> dict[str, float]:
    band_means = {band_name: _band_mean(psd, frequency_bins, *limits) for band_name, limits in _BANDS.items()}
    slow = band_means["delta"] + band_means["theta"]
    fast = band_means["alpha"] + band_means["beta"]
    total = sum(band_means.values())
    alpha_peak_hz, alpha_peak_power, alpha_mean = _alpha_peak(psd, frequency_bins)
    return {
        "alpha_mean": band_means["alpha"],
        "dar": _safe_ratio(band_means["delta"], band_means["alpha"]),
        "dtabr": _safe_ratio(slow, fast),
        "slow_burden": _safe_ratio(slow, total),
        "alpha_beta_preserved": _safe_ratio(fast, total),
        "alpha_peak_hz": alpha_peak_hz,
        "alpha_peak_prominence": _safe_ratio(alpha_peak_power, alpha_mean),
        "alpha_centroid_hz": _alpha_centroid(psd, frequency_bins),
        "slow_fast_bsi": _slow_fast_bsi(psd, frequency_bins, channel_names),
    }


def _band_mean(psd: np.ndarray, frequency_bins: np.ndarray, low: float, high: float) -> float:
    return float(np.nanmean(np.asarray(psd[:, _frequency_mask(frequency_bins, low, high)], dtype=np.float64)))


def _alpha_peak(psd: np.ndarray, frequency_bins: np.ndarray) -> tuple[float, float, float]:
    mask = _frequency_mask(frequency_bins, *_BANDS["alpha"])
    frequencies = np.asarray(frequency_bins[mask], dtype=np.float64)
    spectrum = np.nanmean(np.asarray(psd[:, mask], dtype=np.float64), axis=0)
    peak_index = int(np.nanargmax(spectrum))
    return float(frequencies[peak_index]), float(spectrum[peak_index]), float(np.nanmean(spectrum))


def _alpha_centroid(psd: np.ndarray, frequency_bins: np.ndarray, eps: float = 1e-12) -> float:
    mask = _frequency_mask(frequency_bins, *_BANDS["alpha"])
    frequencies = np.asarray(frequency_bins[mask], dtype=np.float64)
    spectrum = np.nanmean(np.asarray(psd[:, mask], dtype=np.float64), axis=0)
    spectrum = np.clip(spectrum, 0.0, None)
    total = float(np.sum(spectrum))
    if total <= eps:
        return 0.0
    return float(np.sum(frequencies * spectrum) / total)


def _slow_fast_bsi(psd: np.ndarray, frequency_bins: np.ndarray, channel_names: np.ndarray) -> float:
    left_indices, right_indices = _hemisphere_indices(channel_names)
    if left_indices.size == 0 or right_indices.size == 0:
        return 0.0
    slow = np.nanmean(psd[:, _frequency_mask(frequency_bins, 1.0, 8.0)], axis=1)
    fast = np.nanmean(psd[:, _frequency_mask(frequency_bins, 8.0, 30.0)], axis=1)
    ratio = slow / (fast + 1e-8)
    left = float(np.nanmean(ratio[left_indices]))
    right = float(np.nanmean(ratio[right_indices]))
    return abs(left - right) / (abs(left) + abs(right) + 1e-8)


def _hemisphere_indices(channel_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    for index, channel_name in enumerate(channel_names.astype(str)):
        digits = "".join(character for character in channel_name if character.isdigit())
        if not digits:
            continue
        if int(digits[-1]) % 2 == 1:
            left.append(index)
        else:
            right.append(index)
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def _frequency_mask(frequency_bins: np.ndarray, low: float, high: float) -> np.ndarray:
    return (frequency_bins >= low) & (frequency_bins < high)


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-8) -> float:
    value = float(numerator) / (float(denominator) + eps)
    if not np.isfinite(value):
        return 0.0
    return value
