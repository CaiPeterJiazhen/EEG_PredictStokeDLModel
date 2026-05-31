from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from eeg_recovery.metadata.labels import model_input_metadata


NUMERIC_BASELINE_COLUMNS = ("age", "duration", "FMA_pre", "MBI_pre")
CATEGORICAL_BASELINE_COLUMNS = ("sex", "affected_hand")


def baseline_clinical_feature_matrix(metadata: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode baseline-only clinical covariates as a finite low-capacity feature matrix."""

    baseline = model_input_metadata(metadata)
    numeric = _numeric_baseline_features(baseline, NUMERIC_BASELINE_COLUMNS)
    categorical, categorical_names = _categorical_baseline_features(baseline, CATEGORICAL_BASELINE_COLUMNS)
    feature_matrix = np.column_stack([numeric, categorical]).astype(np.float32)
    feature_names = (*NUMERIC_BASELINE_COLUMNS, *categorical_names)
    if not np.isfinite(feature_matrix).all():
        raise ValueError("Baseline clinical feature matrix contains non-finite values.")
    return feature_matrix, feature_names


def _numeric_baseline_features(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    values = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        fill_value = float(series.median()) if series.notna().any() else 0.0
        values.append(series.fillna(fill_value).to_numpy(dtype=np.float32))
    return np.column_stack(values)


def _categorical_baseline_features(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    encoded_columns: list[np.ndarray] = []
    encoded_names: list[str] = []
    for column in columns:
        series = frame[column].astype("string").fillna("missing")
        categories = tuple(sorted(series.unique().tolist()))
        for category in categories:
            encoded_columns.append((series == category).to_numpy(dtype=np.float32))
            encoded_names.append(f"{column}={category}")
    if not encoded_columns:
        return np.zeros((len(frame), 0), dtype=np.float32), ()
    return np.column_stack(encoded_columns).astype(np.float32), tuple(encoded_names)
