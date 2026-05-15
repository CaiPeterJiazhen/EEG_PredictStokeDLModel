from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from eeg_recovery.training.loso import (
    LOSOFold,
    fit_transformer_on_train,
    make_loso_folds,
    transform_with_fitted,
)


SUPERVISED_SUBJECT_IDS = [
    "sub01",
    "sub05",
    "sub07",
    "sub08",
    "sub09",
    "sub10",
    "sub011",
    "sub013",
    "sub14",
    "sub15",
    "sub16",
    "sub17",
    "sub18",
    "sub20",
    "sub22",
    "sub24",
    "sub27",
    "sub28",
    "sub29",
]

EXPECTED_NORMALIZED_SUBJECT_IDS = [
    "sub01",
    "sub05",
    "sub07",
    "sub08",
    "sub09",
    "sub10",
    "sub11",
    "sub13",
    "sub14",
    "sub15",
    "sub16",
    "sub17",
    "sub18",
    "sub20",
    "sub22",
    "sub24",
    "sub27",
    "sub28",
    "sub29",
]


class RecordingTransformer:
    def __init__(self) -> None:
        self.fit_X: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RecordingTransformer":
        self.fit_X = np.asarray(X).copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.fit_X is None:
            raise RuntimeError("transform called before fit")
        return np.asarray(X) + 100


class SupervisedRecordingTransformer:
    def __init__(self) -> None:
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.fit_params: dict[str, object] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params: object,
    ) -> "SupervisedRecordingTransformer":
        self.fit_X = np.asarray(X).copy()
        self.fit_y = None if y is None else np.asarray(y).copy()
        self.fit_params = {
            key: np.asarray(value).copy()
            if isinstance(value, (list, tuple, np.ndarray))
            else value
            for key, value in fit_params.items()
        }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.fit_X is None:
            raise RuntimeError("transform called before fit")
        return np.asarray(X)


def test_make_loso_folds_builds_one_fold_per_supervised_subject() -> None:
    folds = make_loso_folds(SUPERVISED_SUBJECT_IDS)

    assert len(folds) == 19
    assert [fold.fold_index for fold in folds] == list(range(19))
    assert [fold.test_subject_id for fold in folds] == EXPECTED_NORMALIZED_SUBJECT_IDS
    assert {fold.test_subject_id for fold in folds} == set(EXPECTED_NORMALIZED_SUBJECT_IDS)


def test_make_loso_folds_sorts_unordered_inputs_by_canonical_subject_id() -> None:
    folds = make_loso_folds(set(SUPERVISED_SUBJECT_IDS))

    assert [fold.test_subject_id for fold in folds] == EXPECTED_NORMALIZED_SUBJECT_IDS


def test_each_loso_fold_has_one_test_subject_and_eighteen_train_subjects() -> None:
    folds = make_loso_folds(SUPERVISED_SUBJECT_IDS)

    for fold in folds:
        assert isinstance(fold.test_subject_id, str)
        assert len(fold.train_subject_ids) == 18
        assert fold.test_subject_id not in fold.train_subject_ids
        assert set(fold.train_subject_ids) == set(EXPECTED_NORMALIZED_SUBJECT_IDS) - {
            fold.test_subject_id
        }


def test_make_loso_folds_rejects_duplicate_normalized_subject_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate subject ID.*sub11"):
        make_loso_folds(["sub011", "sub11", "sub13"])


def test_make_loso_folds_requires_at_least_two_subjects() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        make_loso_folds(["sub01"])


def test_transformer_guard_fits_only_training_subject_samples() -> None:
    fold = make_loso_folds(["sub01", "sub05", "sub07"])[0]
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [5.0, 50.0],
            [7.0, 70.0],
            [8.0, 80.0],
        ]
    )
    sample_subject_ids = ["sub01", "sub05", "sub05", "sub07", "sub07"]

    transformer = fit_transformer_on_train(
        RecordingTransformer(),
        X,
        sample_subject_ids,
        fold,
    )

    np.testing.assert_array_equal(transformer.fit_X, X[[1, 2, 3, 4]])
    assert not np.any(np.all(transformer.fit_X == X[0], axis=1))


def test_transformer_guard_fits_supervised_transformer_with_training_y_only() -> None:
    fold = make_loso_folds(["sub01", "sub05", "sub07"])[0]
    X = np.array(
        [
            [100.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )
    y = np.array([99, 0, 1, 1])
    sample_subject_ids = ["sub01", "sub05", "sub05", "sub07"]

    transformer = fit_transformer_on_train(
        SupervisedRecordingTransformer(),
        X,
        sample_subject_ids,
        fold,
        y=y,
    )

    np.testing.assert_array_equal(transformer.fit_X, X[[1, 2, 3]])
    np.testing.assert_array_equal(transformer.fit_y, y[[1, 2, 3]])


def test_transformer_guard_masks_sample_aligned_fit_params() -> None:
    fold = make_loso_folds(["sub01", "sub05", "sub07"])[0]
    X = np.array([[100.0], [1.0], [2.0], [3.0]])
    y = np.array([99, 0, 1, 1])
    sample_weight = np.array([999.0, 1.0, 2.0, 3.0])
    sample_groups = ["test-group", "train-a", "train-b", "train-c"]
    fit_note = "keep me"
    sample_subject_ids = ["sub01", "sub05", "sub05", "sub07"]

    transformer = fit_transformer_on_train(
        SupervisedRecordingTransformer(),
        X,
        sample_subject_ids,
        fold,
        y=y,
        sample_weight=sample_weight,
        groups=sample_groups,
        note=fit_note,
    )

    np.testing.assert_array_equal(transformer.fit_params["sample_weight"], sample_weight[[1, 2, 3]])
    np.testing.assert_array_equal(transformer.fit_params["groups"], np.array(sample_groups)[[1, 2, 3]])
    assert transformer.fit_params["note"] == fit_note


def test_transformer_guard_normalizes_manual_fold_train_subject_ids() -> None:
    fold = LOSOFold(
        fold_index=0,
        test_subject_id="sub01",
        train_subject_ids=("sub011", "sub13"),
    )
    X = np.array([[100.0], [11.0], [13.0]])
    sample_subject_ids = ["sub01", "sub11", "sub13"]

    transformer = fit_transformer_on_train(
        RecordingTransformer(),
        X,
        sample_subject_ids,
        fold,
    )

    np.testing.assert_array_equal(transformer.fit_X, X[[1, 2]])


def test_fold_local_standard_scaler_state_comes_from_train_samples_only() -> None:
    fold = make_loso_folds(["sub01", "sub05", "sub07"])[0]
    X = np.array(
        [
            [1000.0, 2000.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    sample_subject_ids = ["sub01", "sub05", "sub05", "sub07"]

    scaler = fit_transformer_on_train(StandardScaler(), X, sample_subject_ids, fold)

    np.testing.assert_allclose(scaler.mean_, X[[1, 2, 3]].mean(axis=0))
    train_transformed = transform_with_fitted(scaler, X[[1, 2, 3]])
    test_transformed = transform_with_fitted(scaler, X[[0]])
    all_transformed = transform_with_fitted(scaler, X)

    assert train_transformed.shape == (3, 2)
    assert test_transformed.shape == (1, 2)
    assert all_transformed.shape == X.shape


def test_transformer_guard_rejects_sample_count_mismatch() -> None:
    fold = make_loso_folds(["sub01", "sub05"])[0]

    with pytest.raises(ValueError, match="same length"):
        fit_transformer_on_train(StandardScaler(), np.zeros((3, 2)), ["sub01", "sub05"], fold)


def test_transformer_guard_rejects_missing_fold_subject_samples() -> None:
    fold = make_loso_folds(["sub01", "sub05", "sub07"])[0]

    with pytest.raises(ValueError, match="No samples found.*sub07"):
        fit_transformer_on_train(
            StandardScaler(),
            np.zeros((2, 2)),
            ["sub01", "sub05"],
            fold,
        )


def test_transformer_guard_rejects_samples_outside_fold_subjects() -> None:
    fold = make_loso_folds(["sub01", "sub05"])[0]

    with pytest.raises(ValueError, match="outside the LOSO fold.*sub07"):
        fit_transformer_on_train(
            StandardScaler(),
            np.zeros((3, 2)),
            ["sub01", "sub05", "sub07"],
            fold,
        )
