from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def compute_topk_selection_frequency(
    feature_table: pd.DataFrame,
    *,
    k_values: Sequence[int] = (10, 20, 50),
    score_column: str = "mean_abs_attribution",
) -> pd.DataFrame:
    required = {"seed", "feature_id", score_column}
    missing = required - set(feature_table.columns)
    if missing:
        raise ValueError(f"feature_table is missing required columns: {sorted(missing)}")
    seeds = sorted(feature_table["seed"].drop_duplicates().tolist())
    rows = []
    for k in k_values:
        selected: dict[str, int] = {}
        for seed in seeds:
            seed_frame = feature_table[feature_table["seed"] == seed]
            top_features = (
                seed_frame.sort_values(score_column, ascending=False)
                .head(int(k))["feature_id"]
                .astype(str)
                .tolist()
            )
            for feature_id in top_features:
                selected[feature_id] = selected.get(feature_id, 0) + 1
        for feature_id, count in selected.items():
            rows.append(
                {
                    "feature_id": feature_id,
                    "k": int(k),
                    "n_selected": int(count),
                    "n_seeds": int(len(seeds)),
                    "selection_frequency": float(count) / float(len(seeds)) if seeds else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["k", "selection_frequency", "feature_id"], ascending=[True, False, True])
