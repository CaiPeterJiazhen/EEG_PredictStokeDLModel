from __future__ import annotations

from collections.abc import Callable
from math import comb

import numpy as np
import pandas as pd

from eeg_recovery.training.metrics import binary_classification_metrics


METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
)


def paired_patient_prediction_comparison(
    reference_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    *,
    threshold: float = 0.5,
    n_bootstrap: int = 5000,
    n_permutations: int = 5000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compare two patient-level prediction tables without treating seeds as samples."""

    reference, candidate = _aligned_prediction_frames(reference_predictions, candidate_predictions)
    y_true = reference["y_true"].to_numpy(dtype=int)
    reference_score = reference["y_score"].to_numpy(dtype=float)
    candidate_score = candidate["y_score"].to_numpy(dtype=float)

    reference_metrics = binary_classification_metrics(y_true, reference_score, threshold=threshold)
    candidate_metrics = binary_classification_metrics(y_true, candidate_score, threshold=threshold)
    observed = {
        metric: candidate_metrics[metric] - reference_metrics[metric]
        for metric in METRIC_NAMES
    }

    rng = np.random.default_rng(random_state)
    bootstrap_deltas = _bootstrap_metric_deltas(
        y_true,
        reference_score,
        candidate_score,
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
    permutation_deltas = _paired_random_swap_deltas(
        y_true,
        reference_score,
        candidate_score,
        threshold=threshold,
        n_permutations=n_permutations,
        rng=rng,
    )

    rows = []
    for metric in METRIC_NAMES:
        boot = bootstrap_deltas[metric]
        perm = permutation_deltas[metric]
        finite_boot = boot[np.isfinite(boot)]
        finite_perm = perm[np.isfinite(perm)]
        observed_delta = observed[metric]
        if finite_boot.size:
            ci_low, ci_high = np.quantile(finite_boot, [0.025, 0.975])
        else:
            ci_low, ci_high = np.nan, np.nan
        if finite_perm.size and np.isfinite(observed_delta):
            p_value = (np.sum(np.abs(finite_perm) >= abs(observed_delta)) + 1.0) / (finite_perm.size + 1.0)
        else:
            p_value = np.nan
        rows.append(
            {
                "metric": metric,
                "reference": reference_metrics[metric],
                "candidate": candidate_metrics[metric],
                "delta": observed_delta,
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
                "permutation_p": float(p_value),
                "n_patients": int(len(y_true)),
                "n_bootstrap": int(n_bootstrap),
                "n_permutations": int(n_permutations),
            }
        )
    return pd.DataFrame(rows)


def paired_correctness_significance_ceiling(
    reference_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    *,
    threshold: float = 0.5,
    alpha: float = 0.05,
) -> dict[str, float | int | bool]:
    """Exact paired-correctness significance audit for a fixed patient set.

    This intentionally ignores score magnitudes and only asks what is possible for
    accuracy-like paired correctness tests on the observed subjects. If the
    reference model has only a few errors, even a perfect candidate may not be
    able to reach a two-sided p-value below alpha.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be within (0, 1).")
    reference, candidate = _aligned_prediction_frames(reference_predictions, candidate_predictions)
    y_true = reference["y_true"].to_numpy(dtype=int)
    reference_correct = (reference["y_score"].to_numpy(dtype=float) >= threshold).astype(int) == y_true
    candidate_correct = (candidate["y_score"].to_numpy(dtype=float) >= threshold).astype(int) == y_true

    favorable = int(np.sum(candidate_correct & ~reference_correct))
    adverse = int(np.sum(reference_correct & ~candidate_correct))
    discordant = favorable + adverse
    reference_errors = int(np.sum(~reference_correct))
    candidate_errors = int(np.sum(~candidate_correct))
    observed_two_sided = _exact_two_sided_random_swap_p(favorable, adverse)
    observed_one_sided = _exact_one_sided_random_swap_p(favorable, adverse)
    best_two_sided = _exact_two_sided_random_swap_p(reference_errors, 0)
    best_one_sided = _exact_one_sided_random_swap_p(reference_errors, 0)
    return {
        "n_patients": int(len(y_true)),
        "threshold": float(threshold),
        "alpha": float(alpha),
        "reference_errors": reference_errors,
        "candidate_errors": candidate_errors,
        "favorable_discordant": favorable,
        "adverse_discordant": adverse,
        "discordant_total": discordant,
        "observed_exact_two_sided_p": float(observed_two_sided),
        "observed_exact_one_sided_p": float(observed_one_sided),
        "best_possible_two_sided_p_given_reference": float(best_two_sided),
        "best_possible_one_sided_p_given_reference": float(best_one_sided),
        "two_sided_significance_possible": bool(best_two_sided < alpha),
        "one_sided_significance_possible": bool(best_one_sided < alpha),
        "min_reference_errors_for_two_sided_alpha": _min_discordant_for_alpha(alpha, two_sided=True),
        "min_reference_errors_for_one_sided_alpha": _min_discordant_for_alpha(alpha, two_sided=False),
    }


def _aligned_prediction_frames(
    reference_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"subject_id", "y_true", "y_score"}
    for name, frame in (("reference", reference_predictions), ("candidate", candidate_predictions)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions missing required column(s): {', '.join(sorted(missing))}")
    reference = reference_predictions.sort_values("subject_id").reset_index(drop=True)
    candidate = candidate_predictions.sort_values("subject_id").reset_index(drop=True)
    if reference["subject_id"].tolist() != candidate["subject_id"].tolist():
        raise ValueError("Reference and candidate predictions must contain the same subject_id values.")
    if not np.array_equal(reference["y_true"].to_numpy(dtype=int), candidate["y_true"].to_numpy(dtype=int)):
        raise ValueError("Reference and candidate y_true values must match after subject_id alignment.")
    return reference, candidate


def _exact_two_sided_random_swap_p(favorable: int, adverse: int) -> float:
    discordant = int(favorable) + int(adverse)
    if discordant == 0:
        return 1.0
    observed = abs(int(favorable) - int(adverse))
    tail = sum(
        comb(discordant, k)
        for k in range(discordant + 1)
        if abs((2 * k) - discordant) >= observed
    )
    return float(tail / (2**discordant))


def _exact_one_sided_random_swap_p(favorable: int, adverse: int) -> float:
    discordant = int(favorable) + int(adverse)
    if discordant == 0:
        return 1.0
    return float(sum(comb(discordant, k) for k in range(int(favorable), discordant + 1)) / (2**discordant))


def _min_discordant_for_alpha(alpha: float, *, two_sided: bool) -> int:
    discordant = 1
    while True:
        p_value = _exact_two_sided_random_swap_p(discordant, 0) if two_sided else _exact_one_sided_random_swap_p(discordant, 0)
        if p_value < alpha:
            return discordant
        discordant += 1


def _bootstrap_metric_deltas(
    y_true: np.ndarray,
    reference_score: np.ndarray,
    candidate_score: np.ndarray,
    *,
    threshold: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    return _resampled_metric_deltas(
        y_true,
        reference_score,
        candidate_score,
        n_iterations=n_bootstrap,
        sampler=lambda n: rng.integers(0, n, size=n),
        threshold=threshold,
    )


def _paired_random_swap_deltas(
    y_true: np.ndarray,
    reference_score: np.ndarray,
    candidate_score: np.ndarray,
    *,
    threshold: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    deltas = {metric: np.full(n_permutations, np.nan, dtype=float) for metric in METRIC_NAMES}
    for iteration in range(n_permutations):
        swap = rng.random(len(y_true)) < 0.5
        permuted_reference = reference_score.copy()
        permuted_candidate = candidate_score.copy()
        permuted_reference[swap] = candidate_score[swap]
        permuted_candidate[swap] = reference_score[swap]
        ref_metrics = binary_classification_metrics(y_true, permuted_reference, threshold=threshold)
        cand_metrics = binary_classification_metrics(y_true, permuted_candidate, threshold=threshold)
        for metric in METRIC_NAMES:
            deltas[metric][iteration] = cand_metrics[metric] - ref_metrics[metric]
    return deltas


def _resampled_metric_deltas(
    y_true: np.ndarray,
    reference_score: np.ndarray,
    candidate_score: np.ndarray,
    *,
    n_iterations: int,
    sampler: Callable[[int], np.ndarray],
    threshold: float,
) -> dict[str, np.ndarray]:
    deltas = {metric: np.full(n_iterations, np.nan, dtype=float) for metric in METRIC_NAMES}
    n_patients = len(y_true)
    for iteration in range(n_iterations):
        indices = sampler(n_patients)
        ref_metrics = binary_classification_metrics(y_true[indices], reference_score[indices], threshold=threshold)
        cand_metrics = binary_classification_metrics(y_true[indices], candidate_score[indices], threshold=threshold)
        for metric in METRIC_NAMES:
            deltas[metric][iteration] = cand_metrics[metric] - ref_metrics[metric]
    return deltas
