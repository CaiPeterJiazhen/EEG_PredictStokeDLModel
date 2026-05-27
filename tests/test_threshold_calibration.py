from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_recovery.training.segment_barlow_model_selection import (
    apply_leave_one_seed_thresholds,
    choose_test_label_threshold,
    leave_one_seed_thresholds,
)


def _frame(seed: int, scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "fold_index": [0, 1, 2, 3],
            "y_true": [1, 0, 1, 0],
            "y_score": scores,
            "seed": seed,
        }
    )


def test_leave_one_seed_threshold_calibration_excludes_current_seed() -> None:
    seed0 = _frame(0, [0.95, 0.85, 0.40, 0.20])
    seed1 = _frame(1, [0.90, 0.10, 0.80, 0.30])
    seed2 = _frame(2, [0.60, 0.20, 0.55, 0.45])

    thresholds = leave_one_seed_thresholds(
        {0: seed0, 1: seed1, 2: seed2},
        metric="balanced_accuracy",
    )

    assert set(thresholds) == {0, 1, 2}
    assert thresholds[0].calibration_seed_count == 2
    assert thresholds[0].held_out_seed == 0
    assert thresholds[0].threshold != choose_test_label_threshold(seed0).threshold


def test_apply_leave_one_seed_thresholds_uses_seed_specific_thresholds() -> None:
    seed0 = _frame(0, [0.95, 0.85, 0.40, 0.20])
    seed1 = _frame(1, [0.90, 0.10, 0.80, 0.30])
    calibrated = apply_leave_one_seed_thresholds({0: seed0, 1: seed1})

    assert set(calibrated["threshold_method"]) == {"leave_one_seed_out_oof"}
    assert calibrated.loc[calibrated["seed"] == 0, "threshold"].nunique() == 1
    assert calibrated.loc[calibrated["seed"] == 1, "threshold"].nunique() == 1
    assert np.issubdtype(calibrated["y_pred"].dtype, np.integer)


def test_test_label_threshold_is_marked_exploratory() -> None:
    frame = _frame(0, [0.95, 0.85, 0.40, 0.20])

    selection = choose_test_label_threshold(frame)

    assert selection.threshold_method == "test_label_optimized"
    assert selection.exploratory is True
    with pytest.raises(ValueError, match="exploratory"):
        choose_test_label_threshold(frame, allow_as_primary=True)
