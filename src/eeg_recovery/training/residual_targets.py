from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from eeg_recovery.metadata.subjects import normalize_subject_id


@dataclass(frozen=True)
class FoldResidualTargets:
    frame: pd.DataFrame
    mean: float
    std: float
    tau: float
    fit_subject_ids: tuple[str, ...]


def compute_signed_distance_from_label_table(
    label_table: pd.DataFrame,
    *,
    threshold: float = 1.5,
) -> pd.DataFrame:
    required = {"subject_id", "Residual", "label"}
    missing = required - set(label_table.columns)
    if missing:
        raise ValueError(f"label table missing required column(s): {', '.join(sorted(missing))}")
    frame = label_table.loc[:, ["subject_id", "Residual", "label"]].copy()
    frame["subject_id"] = frame["subject_id"].map(normalize_subject_id)
    frame["Residual"] = pd.to_numeric(frame["Residual"], errors="raise")
    frame["label"] = frame["label"].astype(int)
    frame["signed_distance"] = float(threshold) - frame["Residual"].astype(float)
    expected_label = (frame["signed_distance"] >= 0.0).astype(int)
    if not (expected_label.to_numpy() == frame["label"].to_numpy()).all():
        raise ValueError("signed_distance sign is inconsistent with binary labels.")
    return frame.set_index("subject_id", drop=False)


def fold_local_standardize_signed_distance(
    targets: pd.DataFrame,
    *,
    fit_subject_ids: Iterable[str],
    transform_subject_ids: Iterable[str],
    eps: float = 1e-6,
) -> FoldResidualTargets:
    if "signed_distance" not in targets.columns:
        raise ValueError("targets must include signed_distance.")
    fit_ids = tuple(normalize_subject_id(subject_id) for subject_id in fit_subject_ids)
    transform_ids = tuple(normalize_subject_id(subject_id) for subject_id in transform_subject_ids)
    if not fit_ids:
        raise ValueError("fit_subject_ids must not be empty.")
    missing_fit = [subject_id for subject_id in fit_ids if subject_id not in targets.index]
    missing_transform = [subject_id for subject_id in transform_ids if subject_id not in targets.index]
    if missing_fit or missing_transform:
        missing = sorted(set(missing_fit + missing_transform))
        raise ValueError(f"residual targets missing subject(s): {', '.join(missing)}")
    fit_values = targets.loc[list(fit_ids), "signed_distance"].to_numpy(dtype=float)
    mean = float(fit_values.mean())
    std = float(fit_values.std(ddof=0))
    if not np.isfinite(std) or std < eps:
        std = 1.0
    transformed = targets.loc[list(transform_ids)].copy()
    transformed["signed_distance_z"] = (transformed["signed_distance"].astype(float) - mean) / std
    transformed["soft_y"] = make_soft_labels_from_signed_distance(
        transformed["signed_distance"].astype(float),
        tau=std,
    ).to_numpy(dtype=float)
    return FoldResidualTargets(
        frame=transformed,
        mean=mean,
        std=std,
        tau=std,
        fit_subject_ids=fit_ids,
    )


def make_soft_labels_from_signed_distance(
    signed_distance: pd.Series | np.ndarray,
    *,
    tau: float,
) -> pd.Series:
    if tau <= 0:
        raise ValueError("tau must be positive.")
    values = np.asarray(signed_distance, dtype=float)
    clipped = np.clip(values / float(tau), -40.0, 40.0)
    soft = 1.0 / (1.0 + np.exp(-clipped))
    index = signed_distance.index if isinstance(signed_distance, pd.Series) else None
    return pd.Series(soft, index=index)
