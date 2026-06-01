from __future__ import annotations

import numpy as np

from eeg_recovery.explainability.tables import make_wpli_edge_attribution_long_table


def test_wpli_long_table_maps_edge_index_to_channel_pair() -> None:
    frame = make_wpli_edge_attribution_long_table(
        subject_id="sub01",
        fold_index=0,
        seed=0,
        state="EC",
        attribution=np.ones((2, 2), dtype=float),
        edge_list=(("C3", "C4"), ("F3", "F4")),
        band_names=("Alpha", "Beta Medium"),
        y_true=0,
        y_score=0.2,
        y_pred=0,
        residual=2.0,
        signed_distance=-0.5,
    )

    assert {"edge_index", "channel_i", "channel_j", "band", "band_index"} <= set(frame.columns)
    assert frame.loc[0, "channel_i"] == "C3"
    assert frame.loc[0, "channel_j"] == "C4"
    assert frame.loc[3, "band"] == "Beta Medium"
