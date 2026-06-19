from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from eeg_recovery.validation.statistics import add_fdr_column


def compute_feature_overlap(
    *,
    model_attribution_features: Iterable[str],
    evidence_sets: Mapping[str, Iterable[str]],
    universe_features: Iterable[str] | None = None,
    n_permutation: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Test whether model-attributed features overlap statistical evidence sets."""

    model = {_clean_feature_id(feature) for feature in model_attribution_features if str(feature)}
    evidence = {
        name: {_clean_feature_id(feature) for feature in features if str(feature)}
        for name, features in evidence_sets.items()
    }
    if universe_features is None:
        universe = set(model)
        for features in evidence.values():
            universe.update(features)
    else:
        universe = {_clean_feature_id(feature) for feature in universe_features if str(feature)}
        universe.update(model)
        for features in evidence.values():
            universe.update(features)

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    universe_list = sorted(universe)
    for name, features in evidence.items():
        observed_overlap = model & features
        model_only = model - features
        evidence_only = features - model
        neither = universe - model - features
        table = [
            [len(observed_overlap), len(model_only)],
            [len(evidence_only), len(neither)],
        ]
        try:
            odds_ratio, fisher_p = fisher_exact(table, alternative="greater")
        except ValueError:
            odds_ratio, fisher_p = float("nan"), float("nan")
        permutation_p = _permutation_overlap_p_value(
            observed=len(observed_overlap),
            model_size=len(model),
            evidence_features=features,
            universe=universe_list,
            n_permutation=n_permutation,
            rng=rng,
        )
        rows.append(
            {
                "evidence_set": name,
                "universe_size": int(len(universe)),
                "model_feature_count": int(len(model)),
                "evidence_feature_count": int(len(features)),
                "overlap_count": int(len(observed_overlap)),
                "overlap_features": ";".join(sorted(observed_overlap)),
                "odds_ratio": float(odds_ratio),
                "fisher_p_value": float(fisher_p),
                "permutation_p_value": float(permutation_p),
                "p_value": float(fisher_p),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "evidence_set",
                "universe_size",
                "model_feature_count",
                "evidence_feature_count",
                "overlap_count",
                "overlap_features",
                "odds_ratio",
                "fisher_p_value",
                "permutation_p_value",
                "p_value",
                "q_value",
            ]
        )
    return add_fdr_column(pd.DataFrame(rows), p_column="p_value", q_column="q_value")


def load_model_attribution_features(
    explainability_dir: str | Path,
    *,
    top_k: int = 100,
) -> set[str]:
    """Load model high-attribution PSD/wPLI feature IDs from known explainability outputs."""

    root = Path(explainability_dir)
    features: set[str] = set()
    features.update(_load_feature_id_csv(root / "model_attribution_features.csv", top_k=top_k))
    features.update(_load_psd_top_features(root / "psd_channel_frequency_top_features.csv", top_k=top_k))
    features.update(_load_psd_top_features(root / "psd_channel_band_importance.csv", top_k=top_k))
    features.update(_load_wpli_top_features(root / "wpli_top_edges.csv", top_k=top_k))
    features.update(_load_wpli_top_features(root / "wpli_edge_band_importance.csv", top_k=top_k))
    return features


def significant_feature_set(
    frame: pd.DataFrame,
    *,
    q_threshold: float = 0.05,
    feature_column: str = "feature_id",
    q_column: str = "q_value",
) -> set[str]:
    """Extract significant feature IDs from a stats table with q-values."""

    if frame.empty or feature_column not in frame.columns or q_column not in frame.columns:
        return set()
    subset = frame[pd.to_numeric(frame[q_column], errors="coerce") <= float(q_threshold)]
    return {_clean_feature_id(value) for value in subset[feature_column].dropna().astype(str)}


def _load_feature_id_csv(path: Path, *, top_k: int) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if "feature_id" not in frame.columns:
        return set()
    frame = _sort_by_importance(frame).head(top_k)
    return {_clean_feature_id(value) for value in frame["feature_id"].dropna().astype(str)}


def _load_psd_top_features(path: Path, *, top_k: int) -> set[str]:
    if not path.exists():
        return set()
    frame = _sort_by_importance(pd.read_csv(path)).head(top_k)
    required = {"state", "channel", "band"}
    if not required.issubset(frame.columns):
        return _load_feature_id_csv(path, top_k=top_k)
    return {
        f"psd|{row.state}|{row.channel}|{row.band}"
        for row in frame.loc[:, ["state", "channel", "band"]].itertuples(index=False)
    }


def _load_wpli_top_features(path: Path, *, top_k: int) -> set[str]:
    if not path.exists():
        return set()
    frame = _sort_by_importance(pd.read_csv(path)).head(top_k)
    required = {"state", "channel_i", "channel_j", "band"}
    if not required.issubset(frame.columns):
        return _load_feature_id_csv(path, top_k=top_k)
    return {
        f"wpli|{row.state}|{row.channel_i}-{row.channel_j}|{row.band}"
        for row in frame.loc[:, ["state", "channel_i", "channel_j", "band"]].itertuples(index=False)
    }


def _sort_by_importance(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("mean_abs_attribution", "abs_attribution", "importance", "selection_frequency"):
        if column in frame.columns:
            return frame.sort_values(column, ascending=False)
    return frame


def _permutation_overlap_p_value(
    *,
    observed: int,
    model_size: int,
    evidence_features: set[str],
    universe: list[str],
    n_permutation: int,
    rng: np.random.Generator,
) -> float:
    if not universe or model_size <= 0 or not evidence_features:
        return float("nan")
    size = min(model_size, len(universe))
    count = 0
    n = max(1, int(n_permutation))
    universe_array = np.asarray(universe, dtype=object)
    for _ in range(n):
        sampled = set(rng.choice(universe_array, size=size, replace=False).tolist())
        if len(sampled & evidence_features) >= observed:
            count += 1
    return float((count + 1) / (n + 1))


def _clean_feature_id(feature: object) -> str:
    return str(feature).strip()
