"""Create Figure 6E: occlusion sensitivity diverging forest plot."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("figure6e_occlusion_sensitivity_forest")

PROJECT_ROOT = Path("F:/CJZProjectFile/EEG_PredictStokeDLModel")
RESULTS_ROOT = (
    PROJECT_ROOT / "final_model_ablation_explainability_results_20260610" / "results"
)

INPUT_PATH = RESULTS_ROOT / "final_model_occlusion_modality_state_band_summary.csv"
FALLBACK_INPUT_PATHS = [
    PROJECT_ROOT
    / "rerun_ablation_explainability_secondary_20260609"
    / "results"
    / "explainability"
    / "final_model_occlusion_modality_state_band_summary.csv",
]

OUTPUT_BASE = (
    RESULTS_ROOT
    / "figures"
    / "paper_panels"
    / "figure6e_occlusion_sensitivity_forest"
)

# Occlusion conditions are arranged by the requested manuscript logic:
# modality-level branches, recording state, then band/network features.
GROUPS = [
    (
        "Modality",
        [
            ("occlude_psd_branch", "PSD branch"),
            ("occlude_wpli_branch", "WPLI branch"),
        ],
    ),
    (
        "State",
        [
            ("occlude_eo_state", "EO state"),
            ("occlude_ec_state", "EC state"),
        ],
    ),
    (
        "Band / network",
        [
            ("occlude_psd_gamma", "PSD Gamma"),
            ("occlude_wpli_beta_low", "WPLI Beta Low"),
            ("occlude_wpli_beta_high", "WPLI Beta High"),
            ("occlude_motor_wpli_edges", "Motor WPLI edges"),
            ("occlude_non_motor_wpli_edges", "Non-motor WPLI edges"),
        ],
    ),
]

# Positive mean_probability_drop means full-input probability exceeded
# occluded probability; this is drawn in a warm color because occlusion reduced
# proportional-recovery probability. Negative values are cold because occlusion
# increased the probability.
WARM = "#C85D3E"
COLD = "#3E79A8"
NEUTRAL = "#6F7782"

# Error bars represent the available dispersion statistic for
# probability_drop. The preferred statistic is sd_probability_drop; the fallback
# names below support compatible summary files. If none are present, the script
# draws points only.
DISPERSION_COLUMNS = [
    "sd_probability_drop",
    "std_probability_drop",
    "se_probability_drop",
    "sem_probability_drop",
    "ci95_probability_drop",
    "iqr_probability_drop",
]


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#606A75",
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


def _condition_column(frame: pd.DataFrame, source: Path) -> str:
    for column in ["occlusion_name", "occlusion_condition", "condition", "ablation_name"]:
        if column in frame.columns:
            return column
    raise ValueError(
        f"{source} must contain an occlusion condition column such as occlusion_name."
    )


def _dispersion_column(frame: pd.DataFrame) -> str | None:
    for column in DISPERSION_COLUMNS:
        if column in frame.columns:
            return column
    return None


def _prepare_plot_data(frame: pd.DataFrame, source: Path) -> tuple[pd.DataFrame, str | None]:
    condition_col = _condition_column(frame, source)
    _require_columns(frame, {condition_col, "mean_probability_drop"}, source)
    dispersion_col = _dispersion_column(frame)

    data = frame.copy()
    data[condition_col] = data[condition_col].astype(str).str.strip()
    data["mean_probability_drop"] = pd.to_numeric(
        data["mean_probability_drop"], errors="coerce"
    )
    if dispersion_col is not None:
        data[dispersion_col] = pd.to_numeric(data[dispersion_col], errors="coerce")

    rows: list[dict[str, object]] = []
    y_value = 0.0
    for group_name, conditions in GROUPS:
        group_start = y_value
        y_value += 0.62
        group_rows = []
        for condition_name, label in conditions:
            match = data[data[condition_col] == condition_name]
            if match.empty:
                LOGGER.warning("Skipping missing occlusion condition: %s", condition_name)
                continue
            if len(match) > 1:
                LOGGER.warning(
                    "Found %d rows for %s; using the first row.", len(match), condition_name
                )
            current = match.iloc[0]
            mean = float(current["mean_probability_drop"])
            if not math.isfinite(mean):
                LOGGER.warning(
                    "Skipping %s because mean_probability_drop is not finite.",
                    condition_name,
                )
                continue
            dispersion = (
                float(current[dispersion_col])
                if dispersion_col is not None
                and pd.notna(current[dispersion_col])
                and float(current[dispersion_col]) >= 0
                else np.nan
            )
            group_rows.append(
                {
                    "group": group_name,
                    "group_y": group_start,
                    "condition": condition_name,
                    "label": label,
                    "mean": mean,
                    "dispersion": dispersion,
                    "y": y_value,
                }
            )
            y_value += 1.0
        if group_rows:
            rows.extend(group_rows)
            y_value += 0.55
        else:
            y_value = group_start

    if not rows:
        raise ValueError("None of the requested occlusion conditions were found.")

    return pd.DataFrame(rows), dispersion_col


def _point_color(mean: float) -> str:
    if mean > 0:
        return WARM
    if mean < 0:
        return COLD
    return NEUTRAL


def _x_limits(plot_data: pd.DataFrame) -> tuple[float, float]:
    lower_values = []
    upper_values = []
    for row in plot_data.itertuples(index=False):
        dispersion = float(row.dispersion) if pd.notna(row.dispersion) else 0.0
        lower_values.append(float(row.mean) - dispersion)
        upper_values.append(float(row.mean) + dispersion)
    max_abs = max(abs(min(lower_values)), abs(max(upper_values)), 0.05)
    limit = max_abs * 1.28
    return -limit, limit


def _plot(plot_data: pd.DataFrame, dispersion_col: str | None, output_base: Path) -> None:
    _configure_matplotlib()

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    x_min, x_max = _x_limits(plot_data)

    ax.axvline(0.0, color="#2F3540", linewidth=1.15, zorder=1)
    ax.grid(axis="x", color="#D7DCE2", linewidth=0.75, alpha=0.8)
    ax.set_axisbelow(True)

    label_offset = 0.018 * (x_max - x_min)
    for row in plot_data.itertuples(index=False):
        mean = float(row.mean)
        y = float(row.y)
        color = _point_color(mean)
        dispersion = float(row.dispersion) if pd.notna(row.dispersion) else np.nan
        if math.isfinite(dispersion):
            ax.errorbar(
                mean,
                y,
                xerr=dispersion,
                fmt="o",
                markersize=5.8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.75,
                ecolor=color,
                elinewidth=1.6,
                capsize=3.0,
                capthick=1.2,
                alpha=0.95,
                zorder=3,
            )
        else:
            ax.scatter(
                [mean],
                [y],
                s=38,
                color=color,
                edgecolor="white",
                linewidth=0.75,
                zorder=3,
            )

        if mean >= 0:
            text_x = mean + label_offset
            ha = "left"
        else:
            text_x = mean - label_offset
            ha = "right"
        ax.text(
            text_x,
            y,
            f"{mean:.3f}",
            va="center",
            ha=ha,
            fontsize=7.8,
            color="#2D333B",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.4},
        )

    group_positions = (
        plot_data[["group", "group_y"]].drop_duplicates().itertuples(index=False)
    )
    for group, group_y in group_positions:
        ax.text(
            x_min,
            float(group_y),
            str(group),
            ha="left",
            va="center",
            fontsize=8.3,
            fontweight="bold",
            color="#4B5563",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(float(plot_data["y"].max()) + 0.55, -0.75)
    ax.set_yticks(plot_data["y"].to_numpy(dtype=float))
    ax.set_yticklabels(plot_data["label"].tolist())
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", colors="#4B5563")

    ax.set_xlabel("Change in proportional-recovery probability after occlusion", labelpad=9)
    fig.suptitle(
        "Occlusion sensitivity analysis",
        x=0.14,
        y=0.965,
        ha="left",
        fontsize=12.2,
        fontweight="bold",
        color="#202631",
    )
    fig.text(
        0.14,
        0.923,
        "Positive values indicate probability reduction after occlusion",
        ha="left",
        va="center",
        fontsize=8.4,
        color="#5D6673",
    )

    if dispersion_col is not None:
        ax.text(
            0.99,
            0.02,
            f"Error bars: {dispersion_col.replace('_', ' ')}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.0,
            color="#707A86",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A939E")
    ax.spines["bottom"].set_color("#8A939E")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.25, right=0.97, bottom=0.14, top=0.86)
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    source = _find_input(INPUT_PATH, RESULTS_ROOT, FALLBACK_INPUT_PATHS)
    frame = pd.read_csv(source)
    plot_data, dispersion_col = _prepare_plot_data(frame, source)
    _plot(plot_data, dispersion_col, OUTPUT_BASE)

    dispersion_text = dispersion_col if dispersion_col is not None else "none"
    print(f"Source: {source}")
    print(f"Rows plotted: {len(plot_data)}")
    print(f"Dispersion statistic: {dispersion_text}")
    print(f"Wrote {OUTPUT_BASE.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_BASE.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
