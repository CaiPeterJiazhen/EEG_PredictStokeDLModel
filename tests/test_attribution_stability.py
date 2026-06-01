from __future__ import annotations

import pandas as pd

from eeg_recovery.explainability.stability import compute_topk_selection_frequency


def test_seed_stability_summary_computes_topk_frequency() -> None:
    frame = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1, 2, 2],
            "feature_id": ["A", "B", "A", "C", "A", "B"],
            "mean_abs_attribution": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        }
    )

    summary = compute_topk_selection_frequency(frame, k_values=(1, 2))

    top1_a = summary[(summary["feature_id"] == "A") & (summary["k"] == 1)].iloc[0]
    assert top1_a["n_selected"] == 3
    assert top1_a["selection_frequency"] == 1.0
