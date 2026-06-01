from __future__ import annotations

import pandas as pd

from eeg_recovery.training.residual_targets import (
    compute_signed_distance_from_label_table,
    fold_local_standardize_signed_distance,
    make_soft_labels_from_signed_distance,
)


def _label_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "Residual": [0.0, 1.5, 3.0, -1.5],
            "label": [1, 1, 0, 1],
        }
    )


def test_signed_distance_is_threshold_minus_residual() -> None:
    targets = compute_signed_distance_from_label_table(_label_table(), threshold=1.5)

    assert targets.loc["sub01"].signed_distance == 1.5
    assert targets.loc["sub03"].signed_distance == -1.5


def test_positive_label_corresponds_to_nonnegative_signed_distance() -> None:
    targets = compute_signed_distance_from_label_table(_label_table(), threshold=1.5)

    assert ((targets["signed_distance"] >= 0).astype(int) == targets["label"]).all()


def test_fold_local_residual_scaler_excludes_test_subject() -> None:
    targets = compute_signed_distance_from_label_table(_label_table(), threshold=1.5)

    scaled = fold_local_standardize_signed_distance(
        targets,
        fit_subject_ids=["sub01", "sub02", "sub04"],
        transform_subject_ids=["sub01", "sub02", "sub03", "sub04"],
    )

    assert "sub03" not in scaled.fit_subject_ids
    fit_values = scaled.frame.loc[["sub01", "sub02", "sub04"], "signed_distance_z"]
    assert abs(float(fit_values.mean())) < 1e-7


def test_soft_labels_are_probabilities() -> None:
    targets = compute_signed_distance_from_label_table(_label_table(), threshold=1.5)
    soft = make_soft_labels_from_signed_distance(targets["signed_distance"], tau=1.0)

    assert ((soft >= 0.0) & (soft <= 1.0)).all()
    assert soft.loc["sub01"] > soft.loc["sub03"]
