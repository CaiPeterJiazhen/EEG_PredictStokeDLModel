from __future__ import annotations

import numpy as np
import torch

from eeg_recovery.training.train_supervised import (
    SupervisedFeatureRecord,
    _fit_state_scaler,
    _make_batch,
    _train_validation_subjects,
)


def _record(subject_id: str, label: int, qeeg_value: float) -> SupervisedFeatureRecord:
    psd = np.full((62, 90), qeeg_value, dtype=np.float32)
    wpli = np.full((1891, 6), qeeg_value + 1.0, dtype=np.float32)
    return SupervisedFeatureRecord(
        subject_id=subject_id,
        label=label,
        eo=psd,
        ec=psd,
        modalities={"psd": (psd, psd), "wpli": (wpli, wpli)},
        qeeg_features=np.asarray([qeeg_value], dtype=np.float32),
        qeeg_feature_names=("qeeg_ec_global_slow_fast_bsi",),
    )


def test_qeeg_scaler_is_fit_without_outer_test_subject():
    records = [_record("sub01", 0, 1.0), _record("sub02", 1, 3.0), _record("sub03", 0, 100.0)]
    fit_records = [records[0], records[1]]

    scaler = _fit_state_scaler(fit_records)
    mean, std = scaler["__qeeg__"]
    batch = _make_batch(records, scaler, torch.device("cpu"), "multimodal")

    np.testing.assert_allclose(mean, np.asarray([2.0], dtype=np.float32))
    np.testing.assert_allclose(std, np.asarray([1.0], dtype=np.float32))
    assert batch["qeeg_features"].shape == (3, 1)
    assert float(batch["qeeg_features"][2, 0]) == 98.0


def test_fold_internal_validation_selection_keeps_test_subject_out_of_qeeg_fit():
    train_subjects = ["sub01", "sub02", "sub03"]
    fit_subjects, val_subjects = _train_validation_subjects(train_subjects, fold_index=1)

    assert "sub04" not in fit_subjects
    assert "sub04" not in val_subjects
    assert set(fit_subjects + val_subjects) == set(train_subjects)


def test_patient_level_loso_qeeg_batch_keeps_one_prediction_per_subject():
    records = [_record(f"sub0{index}", index % 2, float(index)) for index in range(1, 4)]
    scaler = _fit_state_scaler(records[:2])
    batch = _make_batch(records, scaler, torch.device("cpu"), "multimodal")

    assert batch["y"].shape == (3, 1)
    assert batch["qeeg_features"].shape == (3, 1)
