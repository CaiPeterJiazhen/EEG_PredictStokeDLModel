from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.channels.layout import _find_topomap_coords

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.config import load_path_config
from eeg_recovery.visualization.topomap import normalize_channel_name


STATES = ("EO", "EC")
BANDS = ("Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High", "Gamma")
VALUE_COLUMN = "mean_signed_attribution"

REQUESTED_INPUT = (
    PROJECT_ROOT
    / "final_model_ablation_explainability_results_20260610"
    / "results"
    / "psd_channel_band_importance.csv"
)
REQUESTED_SEARCH_ROOT = PROJECT_ROOT / "final_model_ablation_explainability_results_20260610" / "results"
FALLBACK_INPUT = PROJECT_ROOT / "results" / "explainability" / "psd_channel_band_importance.csv"
OUTPUT_BASENAME = PROJECT_ROOT / "results" / "figures" / "paper_panels" / "figure6b_psd_topomap_grid"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a 2x7 PSD channel-band signed attribution topomap grid."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-basename", type=Path, default=OUTPUT_BASENAME)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument(
        "--vlim-percentile",
        type=float,
        default=98.0,
        help=(
            "Symmetric color limit percentile computed over absolute signed attribution values. "
            "Use 100 for the raw global maximum."
        ),
    )
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    input_path = args.input if args.input is not None else resolve_input_path()
    frame = pd.read_csv(input_path)
    validate_input(frame, input_path)

    config = load_path_config(args.config)
    ch_names, pos = topomap_positions(config.standard_1005_ced)
    data_grid = build_data_grid(frame)
    raw_max_abs = max(float(np.nanmax(np.abs(values))) for values in data_grid.values())
    max_abs = robust_symmetric_limit(data_grid, args.vlim_percentile)

    fig, axes = plt.subplots(
        len(STATES),
        len(BANDS),
        figsize=(12.0, 3.65),
        constrained_layout=False,
        facecolor="white",
    )
    plt.subplots_adjust(left=0.07, right=0.91, top=0.84, bottom=0.08, wspace=0.16, hspace=0.10)

    image = None
    for row_index, state in enumerate(STATES):
        for col_index, band in enumerate(BANDS):
            ax = axes[row_index, col_index]
            image, _ = mne.viz.plot_topomap(
                data_grid[(state, band)],
                pos,
                axes=ax,
                show=False,
                sensors=True,
                contours=6,
                outlines="head",
                sphere=(0.0, 0.0, 0.0, 0.105),
                extrapolate="head",
                border="mean",
                image_interp="cubic",
                cmap="RdBu_r",
                vlim=(-max_abs, max_abs),
            )
            if row_index == 0:
                ax.set_title(band, fontsize=8.5, pad=4)
            if col_index == 0:
                ax.text(
                    -0.24,
                    0.50,
                    state,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=9,
                    fontweight="bold",
                )

    if image is None:
        raise RuntimeError("No topomap was rendered.")

    cbar_ax = fig.add_axes([0.925, 0.18, 0.014, 0.58])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_ticks([-max_abs, 0.0, max_abs])
    cbar.set_ticklabels(["Min", "0", "Max"])
    cbar.set_label("Signed PSD attribution", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)

    fig.suptitle("PSD channel-band attribution", fontsize=12, y=0.96)
    args.output_basename.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_basename.with_suffix(".png")
    pdf_path = args.output_basename.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Input: {input_path}")
    print(f"raw max_abs signed attribution: {raw_max_abs:.8g}")
    print(f"plotted symmetric color limit: +/-{max_abs:.8g} ({args.vlim_percentile:g}th percentile)")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


def resolve_input_path() -> Path:
    if REQUESTED_INPUT.exists():
        return REQUESTED_INPUT
    matches = sorted(REQUESTED_SEARCH_ROOT.rglob("psd_channel_band_importance.csv"))
    if matches:
        return matches[0]
    if FALLBACK_INPUT.exists():
        warnings.warn(
            f"Requested PSD importance file was not found under {REQUESTED_SEARCH_ROOT}; "
            f"using canonical project explainability file {FALLBACK_INPUT}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return FALLBACK_INPUT
    raise FileNotFoundError(
        "Could not locate psd_channel_band_importance.csv in the requested final-model results "
        f"tree or fallback path: {FALLBACK_INPUT}"
    )


def validate_input(frame: pd.DataFrame, path: Path) -> None:
    required = {"state", "channel", "band", VALUE_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
    missing_states = sorted(set(STATES) - set(frame["state"].dropna().astype(str)))
    missing_bands = sorted(set(BANDS) - set(frame["band"].dropna().astype(str)))
    if missing_states or missing_bands:
        raise ValueError(f"{path} is missing states={missing_states} or bands={missing_bands}")


def build_data_grid(frame: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    frame = frame.copy()
    frame["channel_norm"] = frame["channel"].map(normalize_channel_name)
    grid: dict[tuple[str, str], np.ndarray] = {}
    for state in STATES:
        for band in BANDS:
            subset = frame[(frame["state"] == state) & (frame["band"] == band)]
            values = subset.groupby("channel_norm", as_index=True)[VALUE_COLUMN].mean().to_dict()
            missing = [
                channel
                for channel in CANONICAL_CHANNELS_62
                if normalize_channel_name(channel) not in values
            ]
            if missing:
                raise ValueError(f"Missing {state} {band} attribution for: {', '.join(missing)}")
            grid[(state, band)] = np.asarray(
                [float(values[normalize_channel_name(channel)]) for channel in CANONICAL_CHANNELS_62],
                dtype=float,
            )
    return grid


def robust_symmetric_limit(data_grid: Mapping[tuple[str, str], np.ndarray], percentile: float) -> float:
    if not 0 < percentile <= 100:
        raise ValueError("--vlim-percentile must be in (0, 100].")
    values = np.concatenate([np.ravel(values.astype(float)) for values in data_grid.values()])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite attribution values were available for color scaling.")
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    return max(limit, 1e-12)


def topomap_positions(ced_fallback_path: Path) -> tuple[list[str], np.ndarray]:
    ch_names = list(CANONICAL_CHANNELS_62)
    normalized_to_pos = load_mne_standard_position_map("standard_1005")
    cb_warnings: list[str] = []
    for channel, fallback in manual_cb_positions(normalized_to_pos).items():
        if channel not in normalized_to_pos:
            normalized_to_pos[channel] = fallback
            cb_warnings.append(channel)
    if cb_warnings:
        warnings.warn(
            "MNE standard_1005 does not include "
            f"{', '.join(cb_warnings)}; using approximate posterior-inferior coordinates.",
            RuntimeWarning,
            stacklevel=2,
        )

    ced_positions = load_ced_eeglab_3d_fallback(ced_fallback_path)
    unresolved: list[str] = []
    ch_pos: dict[str, np.ndarray] = {}
    for channel in ch_names:
        key = normalize_channel_name(channel)
        if key in normalized_to_pos:
            ch_pos[channel] = normalized_to_pos[key]
        elif key in ced_positions:
            ch_pos[channel] = ced_positions[key]
            warnings.warn(
                f"{channel} was missing from MNE standard_1005; using project CED fallback coordinates.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            unresolved.append(channel)
    if unresolved:
        raise ValueError(f"Could not resolve electrode positions for: {', '.join(unresolved)}")

    info = mne.create_info(ch_names=ch_names, sfreq=1000.0, ch_types="eeg")
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage, on_missing="raise")
    pos = _find_topomap_coords(info, picks=np.arange(len(ch_names)), sphere=(0.0, 0.0, 0.0, 0.105))
    return ch_names, np.asarray(pos, dtype=float)


def load_mne_standard_position_map(name: str) -> dict[str, np.ndarray]:
    montage = mne.channels.make_standard_montage(name)
    return {
        normalize_channel_name(channel): np.asarray(position, dtype=float)
        for channel, position in montage.get_positions()["ch_pos"].items()
    }


def manual_cb_positions(position_map: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if {"O1", "PO7"} <= set(position_map):
        result["CB1"] = position_map["O1"] + 0.65 * (position_map["O1"] - position_map["PO7"])
    if {"O2", "PO8"} <= set(position_map):
        result["CB2"] = position_map["O2"] + 0.65 * (position_map["O2"] - position_map["PO8"])
    return result


def load_ced_eeglab_3d_fallback(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    positions_2d: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 4 or parts[-1].upper() != "EEG":
                continue
            try:
                theta_degrees = float(parts[2])
                radius = float(parts[3])
            except ValueError:
                continue
            theta = np.deg2rad(theta_degrees)
            positions_2d[normalize_channel_name(parts[1])] = np.asarray(
                [radius * np.sin(theta), radius * np.cos(theta)],
                dtype=float,
            )
    if not positions_2d:
        return {}
    scale = 0.095 / max(float(np.nanmax(np.linalg.norm(np.vstack(list(positions_2d.values())), axis=1))), 1e-12)
    return {
        channel: np.asarray([xy[0] * scale, xy[1] * scale, 0.0], dtype=float)
        for channel, xy in positions_2d.items()
    }


if __name__ == "__main__":
    main()
