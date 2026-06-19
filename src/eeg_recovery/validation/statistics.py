from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def adjust_pvalues_bh(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction preserving NaN positions."""

    p = np.asarray(list(p_values), dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    finite_p = p[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[finite] = restored
    return q


def spearman_association(x: Sequence[float], y: Sequence[float]) -> dict[str, float | int]:
    """Compute Spearman rho and p-value after removing non-finite pairs."""

    x_values, y_values = _finite_pair(x, y)
    if len(x_values) < 3 or _constant(x_values) or _constant(y_values):
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(len(x_values))}
    rho, p_value = stats.spearmanr(x_values, y_values)
    return {"rho": float(rho), "p_value": float(p_value), "n": int(len(x_values))}


def partial_spearman_association(
    x: Sequence[float],
    y: Sequence[float],
    covariates: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Compute partial Spearman by residualizing ranked x/y on ranked covariates."""

    frame = pd.DataFrame({"x": x, "y": y}).reset_index(drop=True)
    cov = covariates.reset_index(drop=True).copy()
    columns = [str(column) for column in cov.columns]
    cov.columns = columns
    joined = pd.concat([frame, cov], axis=1)
    for column in joined.columns:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < len(columns) + 3:
        return {
            "partial_rho": float("nan"),
            "p_value": float("nan"),
            "n": int(len(joined)),
            "covariates": ";".join(columns),
        }
    ranked = joined.rank(method="average")
    design = ranked.loc[:, columns].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(design)), design])
    x_resid = _linear_residuals(ranked["x"].to_numpy(dtype=float), design)
    y_resid = _linear_residuals(ranked["y"].to_numpy(dtype=float), design)
    if _constant(x_resid) or _constant(y_resid):
        rho = float("nan")
        p_value = float("nan")
    else:
        rho, p_value = stats.pearsonr(x_resid, y_resid)
    return {
        "partial_rho": float(rho),
        "p_value": float(p_value),
        "n": int(len(joined)),
        "covariates": ";".join(columns),
    }


def two_sample_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    method: str = "mannwhitney",
) -> dict[str, float | int | str]:
    """Run a two-sample group comparison with effect size."""

    a = _finite_1d(group_a)
    b = _finite_1d(group_b)
    if len(a) < 1 or len(b) < 1:
        return _empty_group_result(method, len(a), len(b))
    if method == "welch":
        statistic, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        effect = cohens_d(a, b)
        effect_name = "cohens_d"
    elif method == "mannwhitney":
        statistic, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        effect = rank_biserial_from_u(float(statistic), len(a), len(b))
        effect_name = "rank_biserial"
    else:
        raise ValueError("method must be 'welch' or 'mannwhitney'.")
    return {
        "test": method,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(effect),
        "effect_size_name": effect_name,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
    }


def paired_wilcoxon_test(
    before: Sequence[float],
    after: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> dict[str, float | int | str]:
    """Run a paired pre-post Wilcoxon test and bootstrap the mean paired change."""

    pre, post = _finite_pair(before, after)
    if len(pre) < 2:
        return {
            "test": "wilcoxon",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "paired_effect_size": float("nan"),
            "mean_change": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": int(len(pre)),
        }
    diff = pre - post
    try:
        statistic, p_value = stats.wilcoxon(pre, post, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        statistic, p_value = float("nan"), float("nan")
    ci_low, ci_high = bootstrap_ci(diff, n_bootstrap=n_bootstrap, random_state=random_state)
    return {
        "test": "wilcoxon",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "paired_effect_size": float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)),
        "mean_change": float(np.mean(diff)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": int(len(pre)),
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    random_state: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for the mean of finite values."""

    data = _finite_1d(values)
    if len(data) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    n = max(1, int(n_bootstrap))
    means = np.empty(n, dtype=float)
    for index in range(n):
        sample = rng.choice(data, size=len(data), replace=True)
        means[index] = np.mean(sample)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def permutation_group_p_value(
    values: Sequence[float],
    labels: Sequence[int],
    *,
    n_permutation: int = 1000,
    random_state: int = 0,
) -> float:
    """Two-sided permutation p-value for difference in group means."""

    frame = pd.DataFrame({"value": values, "label": labels}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame["label"].nunique() != 2 or len(frame) < 3:
        return float("nan")
    observed = _mean_difference(frame["value"].to_numpy(dtype=float), frame["label"].to_numpy(dtype=int))
    rng = np.random.default_rng(random_state)
    count = 0
    n = max(1, int(n_permutation))
    labels_array = frame["label"].to_numpy(dtype=int)
    values_array = frame["value"].to_numpy(dtype=float)
    for _ in range(n):
        permuted = rng.permutation(labels_array)
        if abs(_mean_difference(values_array, permuted)) >= abs(observed) - 1e-12:
            count += 1
    return float((count + 1) / (n + 1))


def add_fdr_column(frame: pd.DataFrame, p_column: str = "p_value", q_column: str = "q_value") -> pd.DataFrame:
    """Return a copy of frame with a BH-FDR q-value column."""

    result = frame.copy()
    if p_column in result.columns:
        result[q_column] = adjust_pvalues_bh(result[p_column].to_numpy(dtype=float))
    else:
        result[q_column] = np.nan
    return result


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Return Cohen's d using pooled sample variance."""

    first = _finite_1d(a)
    second = _finite_1d(b)
    if len(first) < 2 or len(second) < 2:
        return float("nan")
    pooled_num = (len(first) - 1) * np.var(first, ddof=1) + (len(second) - 1) * np.var(second, ddof=1)
    pooled_den = len(first) + len(second) - 2
    pooled = np.sqrt(pooled_num / pooled_den) if pooled_den > 0 else float("nan")
    return float((np.mean(first) - np.mean(second)) / (pooled + 1e-12))


def rank_biserial_from_u(u_statistic: float, n_a: int, n_b: int) -> float:
    """Return rank-biserial correlation from Mann-Whitney U."""

    if n_a <= 0 or n_b <= 0:
        return float("nan")
    return float(2.0 * u_statistic / (n_a * n_b) - 1.0)


def _finite_pair(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same length.")
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[mask], y_values[mask]


def _finite_1d(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _constant(values: Sequence[float]) -> bool:
    array = np.asarray(values, dtype=float).reshape(-1)
    return len(array) == 0 or np.nanmax(array) - np.nanmin(array) < 1e-12


def _linear_residuals(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def _empty_group_result(method: str, n_a: int, n_b: int) -> dict[str, Any]:
    return {
        "test": method,
        "statistic": float("nan"),
        "p_value": float("nan"),
        "effect_size": float("nan"),
        "effect_size_name": "",
        "n_a": int(n_a),
        "n_b": int(n_b),
        "mean_a": float("nan"),
        "mean_b": float("nan"),
        "median_a": float("nan"),
        "median_b": float("nan"),
    }


def _mean_difference(values: np.ndarray, labels: np.ndarray) -> float:
    unique = sorted(np.unique(labels))
    if len(unique) != 2:
        return float("nan")
    first = values[labels == unique[0]]
    second = values[labels == unique[1]]
    return float(np.mean(first) - np.mean(second))
