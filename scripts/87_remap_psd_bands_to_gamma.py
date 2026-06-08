from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.explainability.tables import psd_band_name


EXPLAIN_ROOT = PROJECT_ROOT / "results" / "explainability"
PSD_LONG_PATH = EXPLAIN_ROOT / "psd_attribution_long.csv"


def main() -> None:
    usecols = [
        "state",
        "channel",
        "channel_index",
        "frequency_hz",
        "frequency_bin",
        "signed_attribution",
        "abs_attribution",
        "y_true",
        "correct",
    ]
    psd_long = pd.read_csv(PSD_LONG_PATH, usecols=usecols)
    frequency_to_band = {
        float(frequency): psd_band_name(float(frequency))
        for frequency in sorted(psd_long["frequency_hz"].dropna().unique())
    }
    psd_long["band"] = psd_long["frequency_hz"].astype(float).map(frequency_to_band)

    audit = (
        psd_long.groupby(["band"], as_index=False)
        .agg(
            min_frequency_hz=("frequency_hz", "min"),
            max_frequency_hz=("frequency_hz", "max"),
            n_rows=("band", "size"),
        )
        .sort_values("min_frequency_hz")
    )
    audit.to_csv(EXPLAIN_ROOT / "psd_band_remap_gamma_audit.csv", index=False)

    channel_band = _aggregate_attribution(psd_long, ["state", "channel", "channel_index", "band"])
    channel_band.to_csv(EXPLAIN_ROOT / "psd_channel_band_importance.csv", index=False)

    frequency = _aggregate_attribution(psd_long, ["state", "frequency_hz", "frequency_bin", "band"])
    frequency.to_csv(EXPLAIN_ROOT / "psd_frequency_importance.csv", index=False)

    channel_frequency = _aggregate_attribution(
        psd_long,
        ["state", "channel", "channel_index", "frequency_hz", "frequency_bin", "band"],
    ).sort_values("mean_abs_attribution", ascending=False)
    channel_frequency.to_csv(EXPLAIN_ROOT / "psd_channel_frequency_top_features.csv", index=False)

    other_rows = int((psd_long["band"] == "Other").sum())
    print(f"Wrote PSD aggregates with Gamma mapping to {EXPLAIN_ROOT}")
    print(audit.to_string(index=False))
    print(f"Other rows: {other_rows}")


def _aggregate_attribution(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = frame.groupby(group_cols, as_index=False).agg(
        mean_signed_attribution=("signed_attribution", "mean"),
        mean_abs_attribution=("abs_attribution", "mean"),
        std_abs_attribution=("abs_attribution", "std"),
        n_samples=("abs_attribution", "size"),
    )
    positive = frame[frame["y_true"] == 1].groupby(group_cols, as_index=False)["abs_attribution"].mean()
    positive = positive.rename(columns={"abs_attribution": "positive_mean_abs_attribution"})
    negative = frame[frame["y_true"] == 0].groupby(group_cols, as_index=False)["abs_attribution"].mean()
    negative = negative.rename(columns={"abs_attribution": "negative_mean_abs_attribution"})
    correct = frame[frame["correct"] == 1].groupby(group_cols, as_index=False)["abs_attribution"].mean()
    correct = correct.rename(columns={"abs_attribution": "correct_only_mean_abs_attribution"})
    return (
        rows.merge(positive, on=group_cols, how="left")
        .merge(negative, on=group_cols, how="left")
        .merge(correct, on=group_cols, how="left")
    )


if __name__ == "__main__":
    main()
