from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from eeg_recovery.training.ensemble_calibration import (
    apply_oof_ensemble_weight_threshold_calibration,
    apply_oof_threshold_calibration,
    best_threshold_from_scores,
    ensemble_prediction_frames,
)


def test_ensemble_prediction_frames_aligns_by_subject_and_uses_weights():
    ntxent = pd.DataFrame(
        {
            "subject_id": ["sub02", "sub01"],
            "fold_index": [1, 0],
            "y_true": [0, 1],
            "y_score": [0.9, 0.2],
        }
    )
    vicreg = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02"],
            "fold_index": [0, 1],
            "y_true": [1, 0],
            "y_score": [0.6, 0.1],
        }
    )

    ensemble = ensemble_prediction_frames(
        {"ntxent": ntxent, "vicreg": vicreg},
        weights={"ntxent": 0.25, "vicreg": 0.75},
    )

    assert ensemble["subject_id"].tolist() == ["sub01", "sub02"]
    np.testing.assert_allclose(ensemble["score_ntxent"].to_numpy(), np.array([0.2, 0.9]))
    np.testing.assert_allclose(ensemble["score_vicreg"].to_numpy(), np.array([0.6, 0.1]))
    np.testing.assert_allclose(ensemble["y_score"].to_numpy(), np.array([0.5, 0.3]))


def test_oof_threshold_calibration_excludes_current_subject():
    predictions = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "fold_index": [0, 1, 2, 3],
            "y_true": [1, 0, 1, 0],
            "y_score": [0.9, 0.8, 0.4, 0.1],
        }
    )

    calibrated = apply_oof_threshold_calibration(predictions, metric="balanced_accuracy")
    full_threshold = best_threshold_from_scores(
        predictions["y_true"].to_numpy(),
        predictions["y_score"].to_numpy(),
        metric="balanced_accuracy",
    ).threshold
    sub03_threshold = calibrated.loc[
        calibrated["subject_id"] == "sub03", "calibrated_threshold"
    ].iloc[0]
    expected_sub03 = best_threshold_from_scores(
        predictions.loc[predictions["subject_id"] != "sub03", "y_true"].to_numpy(),
        predictions.loc[predictions["subject_id"] != "sub03", "y_score"].to_numpy(),
        metric="balanced_accuracy",
    ).threshold

    assert sub03_threshold != full_threshold
    assert sub03_threshold == expected_sub03
    assert calibrated.loc[calibrated["subject_id"] == "sub03", "y_pred"].iloc[0] == 0


def test_oof_ensemble_weight_threshold_calibration_excludes_current_subject():
    first = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "fold_index": [0, 1, 2, 3],
            "y_true": [1, 0, 1, 0],
            "y_score": [0.9, 0.8, 0.4, 0.1],
        }
    )
    second = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub02", "sub03", "sub04"],
            "fold_index": [0, 1, 2, 3],
            "y_true": [1, 0, 1, 0],
            "y_score": [0.1, 0.9, 0.9, 0.8],
        }
    )

    calibrated = apply_oof_ensemble_weight_threshold_calibration(
        {"first": first, "second": second},
        weight_grid=[0.0, 0.5, 1.0],
        metric="balanced_accuracy",
    )
    weight_columns = ["calibrated_weight_first", "calibrated_weight_second"]
    sub03 = calibrated.loc[calibrated["subject_id"] == "sub03"].iloc[0]

    assert np.isclose(sub03[weight_columns].sum(), 1.0)
    assert sub03["calibrated_weight_first"] == 1.0
    assert sub03["calibrated_threshold"] == 0.8500000000000001
    assert sub03["y_pred"] == 0


def test_ensemble_calibration_script_help_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-B", "scripts/10_ensemble_ssl_predictions.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--prediction" in result.stdout
    assert "--calibration-metric" in result.stdout
