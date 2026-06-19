"""Create Figure 6A: ranked EEG feature attribution plot."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


LOGGER = logging.getLogger("figure6a_ranked_top_features")

PROJECT_ROOT = Path("F:/CJZProjectFile/EEG_PredictStokeDLModel")
RESULTS_ROOT = (
    PROJECT_ROOT / "final_model_ablation_explainability_results_20260610" / "results"
)

TABLE_PATH = RESULTS_ROOT / "tables" / "table4_explainability_top_features_for_paper.csv"

FALLBACK_TABLE_PATHS = [
    PROJECT_ROOT
    / "rerun_ablation_explainability_secondary_20260609"
    / "results"
    / "tables"
    / "table4_explainability_top_features_for_paper.csv",
]

OUTPUT_BASE = (
    RESULTS_ROOT / "figures" / "paper_panels" / "figure6a_ranked_top_features"
)

FEATURE_COUNTS = {
    "WPLI": 8,
    "NETWORK": 4,
    "PSD": 4,
}

POSITIVE_DIRECTION = "positive_to_proportional_recovery"
NEGATIVE_DIRECTION = "negative_to_poor_recovery"

COLORS = {
    "positive_outer": "#F4D8CB",
    "positive_inner": "#E7B69E",
    "positive_edge": "#9B624C",
    "negative_outer": "#D9E8F5",
    "negative_inner": "#BFD6EA",
    "negative_edge": "#3F7196",
    "panel": "#F6F6FB",
    "axis": "#5C6670",
    "text": "#2C3038",
    "muted": "#6F7884",
}


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["axis"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
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
                "Using fallback file: %s",
                primary,
                fallback,
            )
            return fallback

    raise FileNotFoundError(
        f"Could not find {primary.name} at {primary}, under {recursive_root}, "
        f"or in fallback paths."
    )


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{source} is missing required columns: {missing_text}")


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _feature_key(row: pd.Series) -> str:
    return "|".join(
        [
            _clean_text(row["state"]),
            _clean_text(row["channel_or_edge"]),
            _clean_text(row["band"]),
        ]
    )


def _display_label(row: pd.Series) -> str:
    state = _clean_text(row["state"])
    band = _clean_text(row["band"])
    feature_type = _clean_text(row["feature_type"]).upper()
    if feature_type == "NETWORK":
        endpoint = _clean_text(row.get("network_group", "")) or _clean_text(
            row["channel_or_edge"]
        )
    else:
        endpoint = _clean_text(row["channel_or_edge"])
    return f"{state} {band} {endpoint}".strip()


def _direction(row: pd.Series) -> str:
    direction = _clean_text(row.get("direction", ""))
    if direction in {POSITIVE_DIRECTION, NEGATIVE_DIRECTION}:
        return direction
    signed = float(row["mean_signed_attribution"])
    return POSITIVE_DIRECTION if signed >= 0 else NEGATIVE_DIRECTION


def _select_representative_features(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "feature_rank",
        "feature_type",
        "state",
        "band",
        "channel_or_edge",
        "mean_signed_attribution",
        "mean_abs_attribution",
    }
    _require_columns(table, required, TABLE_PATH)

    frame = table.copy()
    for column in ["feature_rank", "mean_signed_attribution", "mean_abs_attribution"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    selected = []
    for feature_type, count in FEATURE_COUNTS.items():
        subset = (
            frame[frame["feature_type"].astype(str).str.upper() == feature_type]
            .sort_values("feature_rank", ascending=True)
            .head(count)
            .copy()
        )
        if len(subset) < count:
            LOGGER.warning(
                "Expected %d %s rows but found %d.",
                count,
                feature_type,
                len(subset),
            )
        selected.append(subset)

    result = (
        pd.concat(selected, axis=0, ignore_index=True)
        .sort_values("mean_abs_attribution", ascending=False)
        .reset_index(drop=True)
    )
    result["display_label"] = result.apply(_display_label, axis=1)
    result["feature_key"] = result.apply(_feature_key, axis=1)
    result["direction_for_plot"] = result.apply(_direction, axis=1)
    result["abs_mean_signed_attribution"] = result["mean_signed_attribution"].abs()

    if len(result) != sum(FEATURE_COUNTS.values()):
        LOGGER.warning("Expected 16 representative features but selected %d.", len(result))
    return result


def _rounded_barh(
    ax: plt.Axes,
    y: float,
    width: float,
    height: float,
    color: str,
    edgecolor: str | None = None,
    linewidth: float = 0.0,
    alpha: float = 1.0,
    zorder: int = 2,
) -> None:
    if not math.isfinite(width) or width <= 0:
        return
    rounding_size = min(max(width * 0.04, 0.0025), 0.006)
    patch = FancyBboxPatch(
        (0, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={rounding_size}",
        linewidth=linewidth,
        edgecolor=edgecolor or color,
        facecolor=color,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _rounded_panel(ax: plt.Axes, width: float, y_min: float, y_max: float) -> None:
    patch = FancyBboxPatch(
        (0, y_min),
        width,
        y_max - y_min,
        boxstyle=f"round,pad=0,rounding_size={width * 0.035}",
        linewidth=0,
        facecolor=COLORS["panel"],
        zorder=0,
    )
    ax.add_patch(patch)


def _plot(features: pd.DataFrame, output_base: Path) -> None:
    _configure_matplotlib()

    n_rows = len(features)
    y_positions = list(range(n_rows))
    max_abs = max(float(features["mean_abs_attribution"].max()), 1e-12)
    x_limit = max_abs * 1.24

    fig = plt.figure(figsize=(9.0, 7.0), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[0.48, 2.65, 5.25],
        left=0.065,
        right=0.955,
        top=0.84,
        bottom=0.14,
        wspace=0.055,
    )
    ax_rank = fig.add_subplot(grid[0, 0])
    ax_label = fig.add_subplot(grid[0, 1], sharey=ax_rank)
    ax_bar = fig.add_subplot(grid[0, 2], sharey=ax_rank)

    for ax in [ax_rank, ax_label, ax_bar]:
        ax.set_ylim(-0.65, n_rows - 0.35)
        ax.invert_yaxis()
        ax.tick_params(axis="y", left=False, labelleft=False)

    for ax in [ax_rank, ax_label, ax_bar]:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])

    ax_rank.set_title("#", color=COLORS["muted"], pad=10, fontweight="bold")
    ax_label.set_title("Feature", loc="left", color=COLORS["muted"], pad=10)
    ax_bar.set_title(
        "|Mean signed attribution| / Mean absolute attribution",
        color=COLORS["text"],
        pad=10,
        fontweight="bold",
    )

    ax_bar.set_xlim(0, x_limit)
    ax_bar.set_yticks([])
    _rounded_panel(ax_bar, x_limit * 0.985, -0.58, n_rows - 0.42)

    outer_height = 0.64
    inner_height = 0.38

    for y, (_, row) in zip(y_positions, features.iterrows()):
        direction = row["direction_for_plot"]
        if direction == POSITIVE_DIRECTION:
            outer_color = COLORS["positive_outer"]
            inner_color = COLORS["positive_inner"]
            edge_color = COLORS["positive_edge"]
        else:
            outer_color = COLORS["negative_outer"]
            inner_color = COLORS["negative_inner"]
            edge_color = COLORS["negative_edge"]

        mean_abs = float(row["mean_abs_attribution"])
        signed_abs = float(row["abs_mean_signed_attribution"])

        ax_rank.text(
            0.5,
            y,
            f"{y + 1}.",
            ha="center",
            va="center",
            color=edge_color,
            fontweight="bold",
            fontsize=8.4,
        )
        ax_label.text(
            0.0,
            y,
            row["display_label"],
            ha="left",
            va="center",
            color=edge_color,
            fontsize=9.0,
        )

        _rounded_barh(
            ax_bar,
            y,
            mean_abs,
            outer_height,
            outer_color,
            edgecolor=edge_color,
            linewidth=0.9,
            zorder=2,
        )
        _rounded_barh(
            ax_bar,
            y,
            signed_abs,
            inner_height,
            inner_color,
            alpha=0.82,
            zorder=3,
        )

        ax_bar.text(
            mean_abs + x_limit * 0.018,
            y,
            f"{mean_abs:.3f}",
            ha="left",
            va="center",
            color=edge_color,
            fontsize=8.0,
        )

        if signed_abs > x_limit * 0.16:
            signed_text_x = signed_abs - x_limit * 0.035
            ha = "right"
        else:
            signed_text_x = max(signed_abs + x_limit * 0.014, x_limit * 0.065)
            ha = "left"
        ax_bar.text(
            signed_text_x,
            y,
            f"{signed_abs:.3f}",
            ha=ha,
            va="center",
            color=edge_color,
            fontsize=7.4,
            fontweight="bold",
            zorder=4,
        )

    fig.suptitle(
        "Ranked top EEG feature attributions",
        x=0.065,
        y=0.945,
        ha="left",
        fontsize=12.0,
        fontweight="bold",
        color=COLORS["text"],
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    table_path = _find_input(TABLE_PATH, RESULTS_ROOT, FALLBACK_TABLE_PATHS)
    table = pd.read_csv(table_path)

    features = _select_representative_features(table)
    _plot(features, OUTPUT_BASE)

    print(f"Wrote {OUTPUT_BASE.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_BASE.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
