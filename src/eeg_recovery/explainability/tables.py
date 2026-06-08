from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


PSD_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Delta", 0.5, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta Low", 13.0, 18.0),
    ("Beta Medium", 18.0, 21.0),
    ("Beta High", 21.0, 30.0),
    ("Gamma", 30.0, 45.5),
)


def psd_band_name(frequency_hz: float) -> str:
    for name, low, high in PSD_BANDS:
        if low <= float(frequency_hz) < high:
            return name
    return "Other"


def make_psd_attribution_long_table(
    *,
    subject_id: str,
    fold_index: int,
    seed: int,
    state: str,
    attribution: np.ndarray,
    channel_names: Sequence[str],
    frequency_bins: Sequence[float],
    y_true: int,
    y_score: float,
    y_pred: int,
    residual: float,
    signed_distance: float,
) -> pd.DataFrame:
    values = np.asarray(attribution, dtype=float)
    frequencies = np.asarray(frequency_bins, dtype=float)
    if values.shape != (len(channel_names), len(frequencies)):
        raise ValueError("PSD attribution shape must be channels x frequencies.")
    rows = []
    for channel_index, channel in enumerate(channel_names):
        for frequency_bin, frequency_hz in enumerate(frequencies):
            signed = float(values[channel_index, frequency_bin])
            rows.append(
                {
                    "subject_id": subject_id,
                    "fold_index": int(fold_index),
                    "seed": int(seed),
                    "state": state,
                    "channel": str(channel),
                    "channel_index": int(channel_index),
                    "frequency_hz": float(frequency_hz),
                    "frequency_bin": int(frequency_bin),
                    "band": psd_band_name(float(frequency_hz)),
                    "signed_attribution": signed,
                    "abs_attribution": abs(signed),
                    "y_true": int(y_true),
                    "y_score": float(y_score),
                    "y_pred": int(y_pred),
                    "correct": int(int(y_true) == int(y_pred)),
                    "residual": float(residual),
                    "signed_distance": float(signed_distance),
                }
            )
    return pd.DataFrame(rows)


def make_wpli_edge_attribution_long_table(
    *,
    subject_id: str,
    fold_index: int,
    seed: int,
    state: str,
    attribution: np.ndarray,
    edge_list: Sequence[tuple[str, str]],
    band_names: Sequence[str],
    y_true: int,
    y_score: float,
    y_pred: int,
    residual: float,
    signed_distance: float,
) -> pd.DataFrame:
    values = np.asarray(attribution, dtype=float)
    if values.shape != (len(edge_list), len(band_names)):
        raise ValueError("WPLI attribution shape must be edges x bands.")
    channel_index = _channel_index_map(edge_list)
    rows = []
    for edge_index, (channel_i, channel_j) in enumerate(edge_list):
        for band_index, band in enumerate(band_names):
            signed = float(values[edge_index, band_index])
            rows.append(
                {
                    "subject_id": subject_id,
                    "fold_index": int(fold_index),
                    "seed": int(seed),
                    "state": state,
                    "edge_index": int(edge_index),
                    "channel_i": str(channel_i),
                    "channel_j": str(channel_j),
                    "channel_i_index": channel_index[str(channel_i)],
                    "channel_j_index": channel_index[str(channel_j)],
                    "band": str(band),
                    "band_index": int(band_index),
                    "signed_attribution": signed,
                    "abs_attribution": abs(signed),
                    "y_true": int(y_true),
                    "y_score": float(y_score),
                    "y_pred": int(y_pred),
                    "correct": int(int(y_true) == int(y_pred)),
                    "residual": float(residual),
                    "signed_distance": float(signed_distance),
                }
            )
    return pd.DataFrame(rows)


def _channel_index_map(edge_list: Sequence[tuple[str, str]]) -> dict[str, int]:
    ordered: list[str] = []
    for first, second in edge_list:
        for item in (first, second):
            if item not in ordered:
                ordered.append(item)
    return {channel: index for index, channel in enumerate(ordered)}
