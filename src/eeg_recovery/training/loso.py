from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from eeg_recovery.metadata.subjects import normalize_subject_id


@dataclass(frozen=True)
class LOSOFold:
    fold_index: int
    test_subject_id: str
    train_subject_ids: tuple[str, ...]


def make_loso_folds(subject_ids: Iterable[Any]) -> list[LOSOFold]:
    """Build patient-level leave-one-subject-out folds."""

    normalized_subject_ids = _normalize_subject_ids(subject_ids)
    if len(normalized_subject_ids) < 2:
        raise ValueError("LOSO requires at least 2 unique subject IDs.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for subject_id in normalized_subject_ids:
        if subject_id in seen and subject_id not in duplicates:
            duplicates.append(subject_id)
        seen.add(subject_id)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate subject ID(s) after normalization: {joined}")

    normalized_subject_ids = sorted(normalized_subject_ids)
    folds: list[LOSOFold] = []
    for fold_index, test_subject_id in enumerate(normalized_subject_ids):
        train_subject_ids = tuple(
            subject_id
            for subject_id in normalized_subject_ids
            if subject_id != test_subject_id
        )
        folds.append(
            LOSOFold(
                fold_index=fold_index,
                test_subject_id=test_subject_id,
                train_subject_ids=train_subject_ids,
            )
        )
    return folds


def fit_transformer_on_train(
    transformer: Any,
    X: Any,
    subject_ids: Iterable[Any],
    fold: LOSOFold,
    y: Any | None = None,
    **fit_params: Any,
) -> Any:
    """Fit a transformer using only samples from the fold's training subjects."""

    X_array = np.asarray(X)
    sample_subject_ids = _normalize_subject_ids(subject_ids)
    if X_array.shape[0] != len(sample_subject_ids):
        raise ValueError(
            "X and subject_ids must have the same length along the sample axis: "
            f"{X_array.shape[0]} != {len(sample_subject_ids)}"
        )

    y_train = None
    if y is not None:
        y_array = np.asarray(y)
        if y_array.shape[0] != X_array.shape[0]:
            raise ValueError(
                "X and y must have the same length along the sample axis: "
                f"{X_array.shape[0]} != {y_array.shape[0]}"
            )
    else:
        y_array = None

    test_subject_id, train_subject_ids = _normalized_fold_subject_ids(fold)
    fold_subject_ids = {test_subject_id, *train_subject_ids}
    sample_subject_set = set(sample_subject_ids)
    unexpected_subjects = sorted(sample_subject_set - fold_subject_ids)
    if unexpected_subjects:
        joined = ", ".join(unexpected_subjects)
        raise ValueError(f"Sample subject_ids outside the LOSO fold: {joined}")

    missing_fold_subjects = sorted(fold_subject_ids - sample_subject_set)
    if missing_fold_subjects:
        joined = ", ".join(missing_fold_subjects)
        raise ValueError(f"No samples found for LOSO fold subject(s): {joined}")

    train_subject_set = set(train_subject_ids)
    train_mask = np.array(
        [subject_id in train_subject_set for subject_id in sample_subject_ids],
        dtype=bool,
    )
    if not train_mask.any():
        raise ValueError("No training samples found for this LOSO fold.")

    if y_array is not None:
        y_train = y_array[train_mask]

    masked_fit_params = {
        name: _mask_fit_param_if_sample_aligned(value, train_mask, n_samples=X_array.shape[0])
        for name, value in fit_params.items()
    }
    X_train = X_array[train_mask]
    if y_array is None:
        return transformer.fit(X_train, **masked_fit_params)
    return transformer.fit(X_train, y_train, **masked_fit_params)


def transform_with_fitted(transformer: Any, X: Any) -> Any:
    """Apply a previously fitted transformer without changing its fit state."""

    return transformer.transform(X)


def _normalize_subject_ids(subject_ids: Iterable[Any]) -> list[str]:
    if isinstance(subject_ids, (str, bytes)):
        raise ValueError("subject_ids must be an iterable of subject ID values, not a string.")
    return [normalize_subject_id(subject_id) for subject_id in subject_ids]


def _normalized_fold_subject_ids(fold: LOSOFold) -> tuple[str, tuple[str, ...]]:
    train_subject_ids = tuple(
        normalize_subject_id(subject_id)
        for subject_id in fold.train_subject_ids
    )
    test_subject_id = normalize_subject_id(fold.test_subject_id)
    if test_subject_id in train_subject_ids:
        raise ValueError(f"LOSO fold leaks test subject into train subjects: {test_subject_id}")
    return test_subject_id, train_subject_ids


def _fold_subject_set(fold: LOSOFold) -> set[str]:
    test_subject_id, train_subject_ids = _normalized_fold_subject_ids(fold)
    return {test_subject_id, *train_subject_ids}


def _mask_fit_param_if_sample_aligned(value: Any, train_mask: np.ndarray, *, n_samples: int) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] == n_samples:
        return value[train_mask]
    if isinstance(value, list) and len(value) == n_samples:
        return np.asarray(value)[train_mask]
    return value
