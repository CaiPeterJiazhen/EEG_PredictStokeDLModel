from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from mne.channels.layout import _find_topomap_coords

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.visualization.topomap import normalize_channel_name


RUN_ROOT = PROJECT_ROOT / "final_model_ablation_explainability_results_20260610"
REQUESTED_INPUT = RUN_ROOT / "results" / "tables" / "table4_explainability_top_features_for_paper.csv"
REQUESTED_SEARCH_ROOT = RUN_ROOT / "results"
PROJECT_WIDE_FALLBACK = (
    PROJECT_ROOT
    / "rerun_ablation_explainability_secondary_20260609"
    / "results"
    / "tables"
    / "table4_explainability_top_features_for_paper.csv"
)
OUTPUT_BASENAME = RUN_ROOT / "results" / "figures" / "paper_panels" / "figure6c_wpli_connectivity_grid_2x6_mne"

STATES = ("EO", "EC")
BANDS = ("Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High")
TOP_N = 5


@dataclass(frozen=True)
class Layout:
    channel_names: tuple[str, ...]
    positions_2d: np.ndarray
    position_by_channel: Mapping[str, np.ndarray]
    montage_name: str
    missing_channels: tuple[str, ...]


@dataclass(frozen=True)
class DrawnPanelCount:
    state: str
    band: str
    available_edges: int
    selected_edges: int
    drawn_edges: int
    skipped_edges: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a 2x6 MNE-positioned WPLI connectivity attribution grid."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-basename", type=Path, default=OUTPUT_BASENAME)
    parser.add_argument("--montage", choices=("standard_1005", "standard_1020"), default="standard_1005")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    set_style()
    input_path = args.input if args.input is not None else resolve_input_path()
    raw = pd.read_csv(input_path)
    validate_input(raw, input_path)
    wpli = prepare_wpli_edges(raw)
    panel_edges = select_panel_edges(wpli, top_n=args.top_n)
    layout = get_mne_projected_layout(args.montage)
    width_limits = line_width_limits(panel_edges)
    color_limit = signed_color_limit(panel_edges)

    args.output_basename.parent.mkdir(parents=True, exist_ok=True)
    counts = plot_grid(
        panel_edges=panel_edges,
        layout=layout,
        output_basename=args.output_basename,
        width_limits=width_limits,
        color_limit=color_limit,
    )

    print(f"Input: {input_path}")
    print(f"MNE montage: {layout.montage_name}")
    if layout.missing_channels:
        print("Missing montage channels skipped when needed: " + ", ".join(layout.missing_channels))
    print(f"Line width mean_abs_attribution range: {width_limits[0]:.8g} to {width_limits[1]:.8g}")
    print(f"Signed attribution color range: +/-{color_limit:.8g}")
    for count in counts:
        print(
            f"{count.state} {count.band}: available={count.available_edges}, "
            f"selected={count.selected_edges}, drawn={count.drawn_edges}, skipped={count.skipped_edges}"
        )
    print(f"Wrote: {args.output_basename.with_suffix('.png')}")
    print(f"Wrote: {args.output_basename.with_suffix('.pdf')}")


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.linewidth": 0.7,
        }
    )


def resolve_input_path() -> Path:
    if REQUESTED_INPUT.exists():
        return REQUESTED_INPUT
    matches = sorted(REQUESTED_SEARCH_ROOT.rglob("table4_explainability_top_features_for_paper.csv"))
    if matches:
        return matches[0]
    if PROJECT_WIDE_FALLBACK.exists():
        warnings.warn(
            f"Requested table was not found under {REQUESTED_SEARCH_ROOT}; using project-wide fallback "
            f"{PROJECT_WIDE_FALLBACK}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return PROJECT_WIDE_FALLBACK
    raise FileNotFoundError(
        "Could not locate table4_explainability_top_features_for_paper.csv in the requested run tree "
        f"or fallback path: {PROJECT_WIDE_FALLBACK}"
    )


def validate_input(frame: pd.DataFrame, path: Path) -> None:
    required = {
        "feature_type",
        "state",
        "band",
        "mean_abs_attribution",
        "mean_signed_attribution",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
    if not {"node_1", "node_2"} <= set(frame.columns) and "channel_or_edge" not in frame.columns:
        raise ValueError(f"{path} must contain node_1/node_2 or channel_or_edge for WPLI edges.")


def prepare_wpli_edges(frame: pd.DataFrame) -> pd.DataFrame:
    wpli = frame[frame["feature_type"].astype(str).str.upper().eq("WPLI")].copy()
    if "node_1" not in wpli.columns or "node_2" not in wpli.columns:
        nodes = wpli["channel_or_edge"].astype(str).str.split("-", n=1, expand=True)
        wpli["node_1"] = nodes[0]
        wpli["node_2"] = nodes[1]
    wpli["state"] = wpli["state"].astype(str)
    wpli["band"] = wpli["band"].astype(str)
    wpli["node_1"] = wpli["node_1"].astype(str)
    wpli["node_2"] = wpli["node_2"].astype(str)
    wpli["mean_abs_attribution"] = pd.to_numeric(wpli["mean_abs_attribution"], errors="coerce")
    wpli["mean_signed_attribution"] = pd.to_numeric(wpli["mean_signed_attribution"], errors="coerce")
    return wpli.dropna(subset=["mean_abs_attribution", "mean_signed_attribution"])


def select_panel_edges(wpli: pd.DataFrame, *, top_n: int) -> dict[tuple[str, str], pd.DataFrame]:
    # Each panel independently keeps the top-N WPLI edges by mean absolute attribution.
    panels: dict[tuple[str, str], pd.DataFrame] = {}
    for state in STATES:
        for band in BANDS:
            subset = wpli[(wpli["state"].eq(state)) & (wpli["band"].eq(band))]
            panels[(state, band)] = (
                subset.sort_values("mean_abs_attribution", ascending=False)
                .head(max(int(top_n), 0))
                .reset_index(drop=True)
            )
    return panels


def get_mne_projected_layout(montage_name: str) -> Layout:
    # MNE supplies the standard 3D montage; _find_topomap_coords performs the 2D scalp projection.
    montage = mne.channels.make_standard_montage(montage_name)
    normalized_to_mne_name = {
        normalize_channel_name(name): name
        for name in montage.get_positions()["ch_pos"]
    }
    aliases = common_aliases()
    resolved_project_names: list[str] = []
    resolved_mne_names: list[str] = []
    missing: list[str] = []
    for channel in CANONICAL_CHANNELS_62:
        key = normalize_channel_name(channel)
        montage_key = key if key in normalized_to_mne_name else aliases.get(key)
        if montage_key in normalized_to_mne_name:
            resolved_project_names.append(channel)
            resolved_mne_names.append(normalized_to_mne_name[montage_key])
        else:
            missing.append(channel)
            warnings.warn(
                f"{channel} was not found in MNE {montage_name} after alias mapping; edges using it will be skipped.",
                RuntimeWarning,
                stacklevel=2,
            )

    info = mne.create_info(ch_names=resolved_mne_names, sfreq=1000.0, ch_types="eeg")
    info.set_montage(montage, on_missing="raise")
    coords = _find_topomap_coords(info, picks=np.arange(len(resolved_mne_names)), sphere=(0.0, 0.0, 0.0, 0.105))
    position_by_channel = {
        normalize_channel_name(project_name): np.asarray(xy, dtype=float)
        for project_name, xy in zip(resolved_project_names, coords)
    }
    return Layout(
        channel_names=tuple(resolved_project_names),
        positions_2d=np.asarray(coords, dtype=float),
        position_by_channel=position_by_channel,
        montage_name=montage_name,
        missing_channels=tuple(missing),
    )


def common_aliases() -> dict[str, str]:
    return {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
        "A1": "M1",
        "A2": "M2",
    }


def line_width_limits(panel_edges: Mapping[tuple[str, str], pd.DataFrame]) -> tuple[float, float]:
    selected = collect_selected_edges(panel_edges)
    if selected.empty:
        return (0.0, 1.0)
    values = selected["mean_abs_attribution"].astype(float).to_numpy()
    return float(np.nanmin(values)), float(np.nanmax(values))


def signed_color_limit(panel_edges: Mapping[tuple[str, str], pd.DataFrame]) -> float:
    selected = collect_selected_edges(panel_edges)
    if selected.empty:
        return 1.0
    limit = float(np.nanmax(np.abs(selected["mean_signed_attribution"].astype(float).to_numpy())))
    return max(limit, 1e-12)


def collect_selected_edges(panel_edges: Mapping[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in panel_edges.values() if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_grid(
    *,
    panel_edges: Mapping[tuple[str, str], pd.DataFrame],
    layout: Layout,
    output_basename: Path,
    width_limits: tuple[float, float],
    color_limit: float,
) -> list[DrawnPanelCount]:
    fig, axes = plt.subplots(
        len(STATES),
        len(BANDS),
        figsize=(14.0, 6.5),
        constrained_layout=False,
        facecolor="white",
    )
    plt.subplots_adjust(left=0.055, right=0.89, top=0.84, bottom=0.15, wspace=0.08, hspace=0.12)

    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    cmap = mpl.colormaps["RdBu_r"]
    counts: list[DrawnPanelCount] = []
    for row_index, state in enumerate(STATES):
        for col_index, band in enumerate(BANDS):
            ax = axes[row_index, col_index]
            frame = panel_edges[(state, band)]
            count = draw_connectivity_panel(
                ax=ax,
                frame=frame,
                layout=layout,
                norm=norm,
                cmap=cmap,
                width_limits=width_limits,
                state=state,
                band=band,
            )
            counts.append(count)
            if row_index == 0:
                ax.set_title(band, fontsize=9.5, pad=5)
            if col_index == 0:
                ax.text(
                    -0.17,
                    0.5,
                    state,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

    cbar_ax = fig.add_axes([0.915, 0.26, 0.015, 0.48])
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(scalar, cax=cbar_ax)
    cbar.set_label("Signed attribution", fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=2)

    legend_ax = fig.add_axes([0.32, 0.045, 0.36, 0.06])
    legend_ax.axis("off")
    legend_ax.plot([0.05, 0.26], [0.5, 0.5], color="#4A4A4A", linewidth=0.8, solid_capstyle="round")
    legend_ax.plot([0.38, 0.59], [0.5, 0.5], color="#4A4A4A", linewidth=2.1, solid_capstyle="round")
    legend_ax.plot([0.71, 0.92], [0.5, 0.5], color="#4A4A4A", linewidth=3.5, solid_capstyle="round")
    legend_ax.text(0.5, 0.02, "Line width proportional to mean absolute attribution", ha="center", va="bottom", fontsize=8)

    fig.suptitle("WPLI connectivity attribution across states and frequency bands", fontsize=13, y=0.95)
    fig.savefig(output_basename.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_basename.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return counts


def draw_connectivity_panel(
    *,
    ax: plt.Axes,
    frame: pd.DataFrame,
    layout: Layout,
    norm: TwoSlopeNorm,
    cmap: mpl.colors.Colormap,
    width_limits: tuple[float, float],
    state: str,
    band: str,
) -> DrawnPanelCount:
    # A transparent MNE topomap is used only to draw the standard scalp outline.
    image, _ = mne.viz.plot_topomap(
        np.zeros(len(layout.channel_names), dtype=float),
        layout.positions_2d,
        axes=ax,
        show=False,
        sensors=False,
        contours=0,
        outlines="head",
        sphere=(0.0, 0.0, 0.0, 0.105),
        extrapolate="head",
        border="mean",
        cmap="Greys",
        vlim=(-1.0, 1.0),
    )
    image.set_alpha(0.0)

    ax.scatter(
        layout.positions_2d[:, 0],
        layout.positions_2d[:, 1],
        s=5,
        c="#2B2B2B",
        linewidths=0,
        alpha=0.85,
        zorder=3,
    )

    drawn_nodes: set[str] = set()
    drawn_edges = 0
    skipped_edges = 0
    for _, edge in frame.iterrows():
        first = normalize_channel_name(str(edge["node_1"]))
        second = normalize_channel_name(str(edge["node_2"]))
        if first not in layout.position_by_channel or second not in layout.position_by_channel:
            skipped_edges += 1
            continue
        start = layout.position_by_channel[first]
        end = layout.position_by_channel[second]
        signed = float(edge["mean_signed_attribution"])
        magnitude = float(edge["mean_abs_attribution"])
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=cmap(norm(signed)),
            linewidth=line_width(magnitude, width_limits),
            alpha=0.82,
            solid_capstyle="round",
            zorder=4,
        )
        drawn_edges += 1
        drawn_nodes.update([first, second])

    for node in sorted(drawn_nodes):
        xy = layout.position_by_channel[node]
        ax.text(
            xy[0],
            xy[1] + 0.010,
            node,
            ha="center",
            va="bottom",
            fontsize=5.4,
            color="#202020",
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    return DrawnPanelCount(
        state=state,
        band=band,
        available_edges=int(len(frame)),
        selected_edges=int(len(frame)),
        drawn_edges=drawn_edges,
        skipped_edges=skipped_edges,
    )


def line_width(value: float, limits: tuple[float, float]) -> float:
    lower, upper = limits
    if upper <= lower:
        return 2.15
    return 0.8 + (float(value) - lower) / (upper - lower) * (3.5 - 0.8)


if __name__ == "__main__":
    main()
