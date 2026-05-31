from __future__ import annotations

import pandas as pd

from eeg_recovery.training.patient_level_comparison import (
    paired_correctness_significance_ceiling,
    paired_patient_prediction_comparison,
)


def test_paired_patient_prediction_comparison_reports_ci_and_permutation_p_values():
    reference = pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, 9)],
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
            "y_score": [0.25, 0.35, 0.45, 0.55, 0.52, 0.65, 0.75, 0.85],
            "y_pred": [0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    candidate = reference.copy()
    candidate["y_score"] = [0.10, 0.20, 0.35, 0.45, 0.60, 0.70, 0.82, 0.90]
    candidate["y_pred"] = [0, 0, 0, 0, 1, 1, 1, 1]

    comparison = paired_patient_prediction_comparison(
        reference,
        candidate,
        n_bootstrap=50,
        n_permutations=50,
        random_state=7,
    )

    assert set(comparison["metric"]) >= {"accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"}
    assert comparison["delta"].notna().all()
    assert comparison["bootstrap_ci_low"].notna().all()
    assert comparison["bootstrap_ci_high"].notna().all()
    assert comparison["permutation_p"].between(0.0, 1.0).all()
    brier_row = comparison.loc[comparison["metric"] == "brier_score"].iloc[0]
    assert brier_row["delta"] < 0.0


def test_paired_correctness_significance_ceiling_reports_exact_discordant_limit():
    reference = pd.DataFrame(
        {
            "subject_id": [f"sub{index:02d}" for index in range(1, 20)],
            "y_true": [0] * 9 + [1] * 10,
            "y_score": [0.7, 0.8, 0.9] + [0.1] * 6 + [0.9] * 10,
        }
    )
    candidate = reference.copy()
    candidate["y_score"] = [0.1] * 9 + [0.9] * 10

    ceiling = paired_correctness_significance_ceiling(reference, candidate, threshold=0.5, alpha=0.05)

    assert ceiling["reference_errors"] == 3
    assert ceiling["candidate_errors"] == 0
    assert ceiling["favorable_discordant"] == 3
    assert ceiling["adverse_discordant"] == 0
    assert ceiling["observed_exact_two_sided_p"] == 0.25
    assert ceiling["best_possible_two_sided_p_given_reference"] == 0.25
    assert ceiling["two_sided_significance_possible"] is False
    assert ceiling["min_reference_errors_for_two_sided_alpha"] == 6
