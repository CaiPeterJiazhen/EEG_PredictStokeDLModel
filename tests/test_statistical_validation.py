from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_recovery.evaluation.statistical_validation import (
    bootstrap_metric_ci,
    calibration_metrics,
    exact_binomial_accuracy_pvalue,
    label_permutation_test,
    mcnemar_test,
    paired_bootstrap_difference,
    subject_level_predictions,
)


def test_subject_level_predictions_collapses_seed_rows() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["sub01", "sub01", "sub02", "sub02"],
            "y_true": [1, 1, 0, 0],
            "y_score": [0.8, 0.6, 0.4, 0.2],
            "seed": [0, 1, 0, 1],
        }
    )

    collapsed = subject_level_predictions(frame)

    assert collapsed["subject_id"].tolist() == ["sub01", "sub02"]
    assert collapsed["y_score"].tolist() == [0.7, 0.30000000000000004]
    assert collapsed["n_source_rows"].tolist() == [2, 2]


def test_bootstrap_ci_uses_subject_count_not_source_row_count() -> None:
    frame = subject_level_predictions(
        pd.DataFrame(
            {
                "subject_id": ["sub01", "sub01", "sub02", "sub02", "sub03", "sub03"],
                "y_true": [1, 1, 0, 0, 1, 1],
                "y_score": [0.9, 0.8, 0.2, 0.3, 0.4, 0.6],
            }
        )
    )

    ci = bootstrap_metric_ci(
        frame["y_true"].to_numpy(int),
        frame["y_score"].to_numpy(float),
        n_bootstrap=50,
        random_state=1,
    )

    assert ci["n_subjects"] == 3
    assert {"accuracy_low", "accuracy_high", "roc_auc_low", "roc_auc_high"} <= set(ci)


def test_inference_statistics_return_finite_small_sample_values() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 0])
    score_a = np.array([0.9, 0.2, 0.8, 0.4, 0.7, 0.3])
    score_b = np.array([0.8, 0.6, 0.7, 0.5, 0.6, 0.4])

    permutation = label_permutation_test(
        y_true,
        score_a,
        metric="roc_auc",
        n_permutations=25,
        random_state=3,
    )
    paired = paired_bootstrap_difference(
        y_true,
        score_a,
        score_b,
        metric="accuracy",
        n_bootstrap=25,
        random_state=4,
    )
    cal = calibration_metrics(y_true, score_a, n_bins=3)

    assert 0.0 <= exact_binomial_accuracy_pvalue(y_true, score_a) <= 1.0
    assert permutation["n_permutations"] == 25
    assert "difference" in paired and "ci_low" in paired and "ci_high" in paired
    assert {"brier_score", "ece", "calibration_intercept", "calibration_slope"} <= set(cal)
    assert mcnemar_test(y_true, score_a, score_b)["n_discordant"] >= 0
