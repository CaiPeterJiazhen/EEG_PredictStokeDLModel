from __future__ import annotations

import numpy as np

from eeg_recovery.explainability.tables import make_psd_attribution_long_table


def test_psd_long_table_has_channel_frequency_and_band_columns() -> None:
    frame = make_psd_attribution_long_table(
        subject_id="sub01",
        fold_index=0,
        seed=0,
        state="EO",
        attribution=np.ones((2, 4), dtype=float),
        channel_names=("C3", "C4"),
        frequency_bins=np.array([2.0, 6.0, 10.0, 20.0]),
        y_true=1,
        y_score=0.7,
        y_pred=1,
        residual=-1.0,
        signed_distance=2.5,
    )

    expected = {
        "subject_id",
        "fold_index",
        "seed",
        "state",
        "channel",
        "channel_index",
        "frequency_hz",
        "frequency_bin",
        "band",
        "signed_attribution",
        "abs_attribution",
    }
    assert expected <= set(frame.columns)
    assert set(frame["band"]) == {"Delta", "Theta", "Alpha", "Beta Medium"}
