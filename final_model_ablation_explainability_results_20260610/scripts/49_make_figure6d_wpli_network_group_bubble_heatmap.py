"""Create Figure 6D: WPLI network-group attribution bubble heatmap."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


LOGGER = logging.getLogger("figure6d_wpli_network_group_bubble_heatmap")

PROJECT_ROOT = Path("F:/CJZProjectFile/EEG_PredictStokeDLModel")
RESULTS_ROOT = (
    PROJECT_ROOT / "final_model_ablation_explainability_results_20260610" / "results"
)

INPUT_PATH = RESULTS_ROOT / "wpli_network_group_importance.csv"
FALLBACK_INPUT_PATHS = [
    PROJECT_ROOT
    / "rerun_ablation_explainability_secondary_20260609"
    / "results"
    / "explainability"
    / "wpli_network_group_importance.csv",
    PROJECT_ROOT / "results" / "explainability" / "wpli_network_group_importance.csv",
]

OUTPUT_BASE = (
    RESULTS_ROOT
    / "figures"
    / "paper_panels"
    / "figure6d_wpli_network_group_bubble_heatmap"
)

STATE_BAND_ORDER = [
    ("EC", "Delta"),
    ("EC", "Theta"),
    ("EC", "Alpha"),
    ("EC", "Beta Low"),
    ("EC", "Beta Medium"),
    ("EC", "Beta High"),
]

STATE_BAND_LABELS = {
    ("EC", "Delta"): "EC\nDelta",
    ("EC", "Theta"): "EC\nTheta",
    ("EC", "Alpha"): "EC\nAlpha",
    ("EC", "Beta Low"): "EC\nBeta\nLow",
    ("EC", "Beta Medium"): "EC\nBeta\nMedium",
    ("EC", "Beta High"): "EC\nBeta\nHigh",
}

MIN_BUBBLE_AREA = 28.0
MAX_BUBBLE_AREA = 520.0


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def _find_input(primary: Path, recursive_root: Path, fallbacks: list[Path]) -> Path:
    if primary.exists():
        return primary

    if recursive_root.exists():
        matches = sorted(recursive_root.rglob(primary.name))
        if matches:
            LOGGER.warning(
                "Primary input not found: %s. Using recursively discovered file: %s",
                primary,
                matches[0],
            )
            return matches[0]

    for fallback in fallbacks:
        if fallback.exists():
            LOGGER.warning(
                "Primary input not found under requested results root: %s. "
                "Using project fallback file: %s",
                primary,
                fallback,
            )
            return fallback

    raise FileNotFoundError(
        f"Could not find {primary.name} at {primary}, under {recursive_root}, "
        f"or in project fallback paths."
    )


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{source} is missing required columns: {missing_text}")


def _normalise_band(value: object) -> str:
    text = str(value).strip()
    compact = text.lower().replace("_", " ").replace("-", " ")
    compact = " ".join(compact.split())
    aliases = {
        "delta": "Delta",
        "theta": "Theta",
        "alpha": "Alpha",
        "beta low": "Beta Low",
        "low beta": "Beta Low",
        "betalow": "Beta Low",
        "beta medium": "Beta Medium",
        "medium beta": "Beta Medium",
        "betamedium": "Beta Medium",
        "beta high": "Beta High",
        "high beta": "Beta High",
        "betahigh": "Beta High",
    }
    return aliases.get(compact, text)


def _direction_to_signed_value(value: object) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric > 0:
            return 1.0
        if numeric < 0:
            return -1.0
        return 0.0

    text = str(value).strip().lower()
    if not text:
        return np.nan

    positive_terms = (
        "positive",
        "proportional",
        "good",
        "favorable",
        "improved",
        "increase",
        "higher",
        "pos",
        "+",
    )
    negative_terms = (
        "negative",
        "poor",
        "unfavorable",
        "worse",
        "decrease",
        "lower",
        "neg",
        "-",
    )
    neutral_terms = ("neutral", "zero", "none", "mixed")

    if any(term in text for term in positive_terms):
        return 1.0
    if any(term in text for term in negative_terms):
        return -1.0
    if any(term in text for term in neutral_terms):
        return 0.0
    return np.nan


def _filter_wpli_network_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep WPLI network-group attribution rows when optional type columns exist."""
    data = frame.copy()

    wpli_columns = [
        "feature_type",
        "feature",
        "modality",
        "data_type",
        "input_type",
        "measure",
        "metric",
        "connectivity_metric",
    ]
    for column in wpli_columns:
        if column in data.columns:
            values = data[column].astype(str).str.lower()
            if values.str.contains("wpli", na=False).any():
                data = data[values.str.contains("wpli", na=False)]

    group_columns = [
        "analysis_type",
        "result_type",
        "aggregation",
        "grouping",
        "level",
    ]
    for column in group_columns:
        if column in data.columns:
            values = data[column].astype(str).str.lower()
            if values.str.contains("network|group", regex=True, na=False).any():
                data = data[values.str.contains("network|group", regex=True, na=False)]

    return data


def _prepare_plot_table(frame: pd.DataFrame, source: Path) -> tuple[pd.DataFrame, list[str]]:
    required = {"state", "band", "network_group", "mean_abs_attribution"}
    _require_columns(frame, required, source)

    data = _filter_wpli_network_group(frame)
    data["state"] = data["state"].astype(str).str.strip().str.upper()
    data["band"] = data["band"].map(_normalise_band)
    data["network_group"] = data["network_group"].astype(str).str.strip()
    data["mean_abs_attribution"] = pd.to_numeric(
        data["mean_abs_attribution"], errors="coerce"
    )

    if "mean_signed_attribution" in data.columns:
        data["signed_attribution"] = pd.to_numeric(
            data["mean_signed_attribution"], errors="coerce"
        )
        color_source = "mean_signed_attribution"
    elif "direction" in data.columns:
        data["signed_attribution"] = data["direction"].map(_direction_to_signed_value)
        color_source = "direction"
    else:
        raise ValueError(
            f"{source} must contain mean_signed_attribution or direction for colors."
        )

    data = data.dropna(
        subset=["state", "band", "network_group", "mean_abs_attribution"]
    )
    if data.empty:
        raise ValueError(f"{source} has no valid WPLI network-group rows.")

    data["state_band"] = list(zip(data["state"], data["band"]))
    data = data[data["state_band"].isin(STATE_BAND_ORDER)]
    if data.empty:
        raise ValueError(
            f"{source} has no rows matching the fixed EC-only state-band display order."
        )

    # Top-12 network_group selection:
    # rank each network_group by its maximum mean_abs_attribution across the
    # displayed EC-only state-band combinations, then keep the 12 strongest groups.
    top_groups = (
        data.groupby("network_group", as_index=True)["mean_abs_attribution"]
        .max()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    if len(top_groups) < 12:
        LOGGER.warning("Only %d network groups are available.", len(top_groups))

    data = data[data["network_group"].isin(top_groups)]

    plot_table = (
        data.groupby(["network_group", "state_band"], as_index=False)
        .agg(
            mean_abs_attribution=("mean_abs_attribution", "mean"),
            signed_attribution=("signed_attribution", "mean"),
        )
        .dropna(subset=["mean_abs_attribution"])
    )
    plot_table.attrs["color_source"] = color_source
    return plot_table, top_groups


def _scale_bubble_area(values: pd.Series | np.ndarray, vmax: float) -> np.ndarray:
    # Bubble size mapping:
    # matplotlib scatter uses marker area in points^2, so a linear area scale
    # keeps bubble area proportional to mean_abs_attribution while a small
    # minimum preserves visibility for near-zero valid cells.
    numeric = np.asarray(values, dtype=float)
    if vmax <= 0 or not np.isfinite(vmax):
        return np.full_like(numeric, MIN_BUBBLE_AREA, dtype=float)
    scaled = numeric / vmax
    return MIN_BUBBLE_AREA + scaled * (MAX_BUBBLE_AREA - MIN_BUBBLE_AREA)


def _format_value(value: float) -> str:
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2e}"


def _plot(
    plot_table: pd.DataFrame,
    top_groups: list[str],
    output_base: Path,
    *,
    show_title: bool = True,
) -> None:
    _configure_matplotlib()

    x_lookup = {state_band: index for index, state_band in enumerate(STATE_BAND_ORDER)}
    y_lookup = {group: index for index, group in enumerate(top_groups)}
    plot_table = plot_table.copy()
    plot_table["x"] = plot_table["state_band"].map(x_lookup)
    plot_table["y"] = plot_table["network_group"].map(y_lookup)

    abs_vmax = float(plot_table["mean_abs_attribution"].max())
    bubble_area = _scale_bubble_area(plot_table["mean_abs_attribution"], abs_vmax)

    signed_values = plot_table["signed_attribution"].to_numpy(dtype=float)
    signed_limit = float(np.nanmax(np.abs(signed_values)))
    if not np.isfinite(signed_limit) or signed_limit == 0:
        signed_limit = 1.0

    # Color mapping:
    # signed attribution is centered at zero; negative values use cool blue
    # for poor-recovery direction and positive values use warm orange/red for
    # proportional-recovery direction.
    color_map = LinearSegmentedColormap.from_list(
        "signed_wpli_direction",
        ["#2F6EB3", "#F7F7F7", "#D55A3A"],
    )
    color_norm = TwoSlopeNorm(vmin=-signed_limit, vcenter=0.0, vmax=signed_limit)

    fig = plt.figure(figsize=(10.5, 6.5))
    grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, 0.16],
        left=0.20,
        right=0.96,
        bottom=0.15,
        top=0.88 if show_title else 0.93,
        wspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 0])
    side_grid = grid[0, 1].subgridspec(
        nrows=2,
        ncols=1,
        height_ratios=[0.58, 0.42],
        hspace=0.35,
    )
    cax = fig.add_subplot(side_grid[0, 0])
    size_ax = fig.add_subplot(side_grid[1, 0])
    position = cax.get_position()
    cax.set_position(
        [
            position.x0 + position.width * 0.23,
            position.y0,
            position.width * 0.54,
            position.height,
        ]
    )

    scatter = ax.scatter(
        plot_table["x"],
        plot_table["y"],
        s=bubble_area,
        c=plot_table["signed_attribution"],
        cmap=color_map,
        norm=color_norm,
        edgecolors="#4D4D4D",
        linewidths=0.35,
        alpha=0.96,
    )

    if show_title:
        ax.set_title(
            "Network-group WPLI attribution summary",
            fontweight="bold",
            pad=12,
        )
    ax.set_xlabel("State-band")
    ax.set_ylabel("Network group")
    ax.set_xlim(-0.5, len(STATE_BAND_ORDER) - 0.5)
    ax.set_ylim(len(top_groups) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(STATE_BAND_ORDER)))
    ax.set_xticklabels([STATE_BAND_LABELS[item] for item in STATE_BAND_ORDER])
    ax.set_yticks(np.arange(len(top_groups)))
    ax.set_yticklabels(top_groups)

    ax.set_xticks(np.arange(-0.5, len(STATE_BAND_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(top_groups), 1), minor=True)
    ax.grid(which="minor", color="#E8E8E8", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8B8B8")
    ax.spines["bottom"].set_color("#B8B8B8")

    colorbar = fig.colorbar(scatter, cax=cax)
    colorbar.set_label("Color = signed attribution direction", labelpad=12)
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.yaxis.tick_right()
    colorbar.set_ticks([-signed_limit, 0.0, signed_limit])
    colorbar.set_ticklabels(
        [_format_value(-signed_limit), "0", _format_value(signed_limit)]
    )
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.text(
        0.5,
        1.06,
        "Proportional\nrecovery",
        transform=colorbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#B94931",
    )
    colorbar.ax.text(
        0.5,
        -0.10,
        "Poor\nrecovery",
        transform=colorbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#2F6EB3",
    )

    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)
    size_ax.axis("off")
    size_ax.text(
        0.02,
        0.98,
        "Bubble size $\\propto$\nmean absolute attribution",
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
    )

    legend_values = [abs_vmax * 0.25, abs_vmax * 0.50, abs_vmax]
    legend_y = [0.70, 0.47, 0.20]
    legend_sizes = _scale_bubble_area(np.asarray(legend_values), abs_vmax)
    for y_value, area, label_value in zip(legend_y, legend_sizes, legend_values):
        size_ax.scatter(
            0.28,
            y_value,
            s=area,
            facecolors="#D0D0D0",
            edgecolors="#4D4D4D",
            linewidths=0.35,
        )
        size_ax.text(
            0.55,
            y_value,
            _format_value(label_value),
            ha="left",
            va="center",
            fontsize=8.0,
        )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Figure 6D WPLI network-group bubble heatmap."
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=OUTPUT_BASE,
        help="Output path without extension. PNG and PDF are written.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw the main figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    source = _find_input(INPUT_PATH, RESULTS_ROOT, FALLBACK_INPUT_PATHS)
    frame = pd.read_csv(source)
    plot_table, top_groups = _prepare_plot_table(frame, source)
    _plot(plot_table, top_groups, args.output_base, show_title=not args.no_title)

    print(f"Source: {source}")
    print(f"Rows plotted: {len(plot_table)}")
    print(f"Top groups: {', '.join(top_groups)}")
    print(f"Wrote {args.output_base.with_suffix('.png')}")
    print(f"Wrote {args.output_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
