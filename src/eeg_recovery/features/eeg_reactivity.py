from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import numpy as np

from eeg_recovery.features.eeg_summary import compute_subject_eeg_summary_features


@dataclass(frozen=True)
class EEGReactivityFeatureVector:
    values: np.ndarray
    feature_names: tuple[str, ...]


def compute_subject_eeg_reactivity_features(output_root: str | Path, subject_id: str) -> EEGReactivityFeatureVector:
    summary = compute_subject_eeg_summary_features(output_root, subject_id)
    return reactivity_features_from_named_values(summary.values, summary.feature_names)


def reactivity_features_from_named_values(
    values: np.ndarray,
    feature_names: Iterable[str],
    *,
    eps: float = 1e-8,
) -> EEGReactivityFeatureVector:
    names = tuple(feature_names)
    value_array = np.asarray(values, dtype=np.float32)
    if value_array.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")
    if value_array.shape[0] != len(names):
        raise ValueError("feature_names length must match values length.")
    lookup = {name: float(value_array[index]) for index, name in enumerate(names)}
    output_values: list[float] = []
    output_names: list[str] = []

    for eo_prefix, ec_prefix, output_prefix in (
        ("psd_eo_", "psd_ec_", "reactivity_psd_"),
        ("wpli_eo_", "wpli_ec_", "reactivity_wpli_"),
    ):
        for eo_name in names:
            if not eo_name.startswith(eo_prefix):
                continue
            suffix = eo_name[len(eo_prefix) :]
            ec_name = f"{ec_prefix}{suffix}"
            if ec_name not in lookup:
                continue
            eo_value = lookup[eo_name]
            ec_value = lookup[ec_name]
            delta = ec_value - eo_value
            fractional_change = delta / (abs(ec_value) + abs(eo_value) + eps)
            output_names.append(f"{output_prefix}{suffix}_ec_minus_eo")
            output_values.append(float(delta))
            output_names.append(f"{output_prefix}{suffix}_ec_minus_eo_fractional_change")
            output_values.append(float(fractional_change))

    vector = np.asarray(output_values, dtype=np.float32)
    if vector.size == 0:
        raise ValueError("No paired EO/EC PSD or WPLI summary features were found.")
    if not np.isfinite(vector).all():
        raise ValueError("EEG reactivity features contain non-finite values.")
    return EEGReactivityFeatureVector(values=vector, feature_names=tuple(output_names))
