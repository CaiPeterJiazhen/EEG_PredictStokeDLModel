from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_recovery.training.segment_barlow_model_selection import (
    compute_per_subject_error_frequency,
    equal_weight_model_ensemble,
    load_required_prediction_csv,
)


def _predictions(scores: list[float], *, subjects: list[str] | None = None) -> pd.DataFrame:
    resolved_subjects = subjects or ["sub01", "sub02", "sub03"]
    return pd.DataFrame(
        {
            "subject_id": resolved_subjects,
            "fold_index": list(range(len(resolved_subjects))),
            "y_true": [1, 0, 1][: len(resolved_subjects)],
            "y_score": scores,
        }
    )


def test_equal_weight_ensemble_requires_identical_subjects_and_labels() -> None:
    psd = _predictions([0.8, 0.2, 0.7], subjects=["sub01", "sub02", "sub03"])
    wpli = _predictions([0.4, 0.6, 0.5], subjects=["sub01", "sub02", "sub04"])

    with pytest.raises(ValueError, match="identical subject_id"):
        equal_weight_model_ensemble({"psd": psd, "wpli": wpli}, model_group="psd_wpli")


def test_load_required_prediction_csv_rejects_missing_score(tmp_path: Path) -> None:
    path = tmp_path / "bad_predictions.csv"
    pd.DataFrame(
        {
            "subject_id": ["sub01"],
            "fold_index": [0],
            "y_true": [1],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="y_score"):
        load_required_prediction_csv(path)


def test_equal_weight_ensemble_outputs_expected_scores() -> None:
    psd = _predictions([0.8, 0.2, 0.7])
    wpli = _predictions([0.4, 0.6, 0.5])

    ensemble = equal_weight_model_ensemble(
        {"psd": psd, "wpli": wpli},
        model_group="psd_wpli",
        seed=0,
    )

    np.testing.assert_allclose(ensemble["y_score"].to_numpy(), np.array([0.6, 0.4, 0.6]))
    assert ensemble["ensemble_members"].iloc[0] == "psd+wpli"
    assert ensemble["ensemble_weighting"].iloc[0] == "equal"
    assert ensemble["seed"].iloc[0] == 0


def test_compute_per_subject_error_frequency_reports_score_spread() -> None:
    seed0 = _predictions([0.8, 0.7, 0.6])
    seed0["y_pred"] = (seed0["y_score"] >= 0.5).astype(int)
    seed1 = _predictions([0.9, 0.1, 0.4])
    seed1["y_pred"] = (seed1["y_score"] >= 0.5).astype(int)

    errors = compute_per_subject_error_frequency(
        [seed0, seed1],
        model_group="toy",
    )

    sub02 = errors.loc[errors["subject_id"] == "sub02"].iloc[0]
    assert sub02["model_group"] == "toy"
    assert sub02["n_runs"] == 2
    assert sub02["n_errors"] == 1
    assert sub02["error_rate"] == 0.5
    assert sub02["repeatedly_wrong_flag"] is True
    assert "borderline_label_flag" in errors.columns
