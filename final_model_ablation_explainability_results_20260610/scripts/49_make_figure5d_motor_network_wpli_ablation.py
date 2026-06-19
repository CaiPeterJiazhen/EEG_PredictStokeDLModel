"""Create Figure 5D: network-constrained WPLI ablation slope plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
PRIMARY_CSV = TABLE_DIR / "table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv"
SUPPLEMENTARY_CSV = TABLE_DIR / "supplementary_final_ssl_cnn_feature_state_band_ablation.csv"
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "paper_panels"
OUT_STEM = OUT_DIR / "figure5d_motor_network_wpli_ablation"

TARGET_ABLATIONS = [
    "motor_wpli_edges_only",
    "full_minus_motor_wpli_edges",
]

ABLATION_LABELS = {
    "motor_wpli_edges_only": "Motor WPLI edges only",
    "full_minus_motor_wpli_edges": "Minus motor WPLI edges",
}

METRICS = [
    {
        "label": "Balanced accuracy",
        "mean": "balanced_accuracy_mean",
        "sd": "balanced_accuracy_sd",
        "reference": 0.8411,
    },
    {
        "label": "ROC-AUC",
        "mean": "roc_auc_mean",
        "sd": "roc_auc_sd",
        "reference": 0.8867,
    },
    {
        "label": "Brier score",
        "mean": "brier_mean",
        "sd": "brier_sd",
        "reference": 0.1324,
    },
]

MOTOR_COLOR = "#D9792B"
MINUS_COLOR = "#1B9E9A"
REFERENCE_COLOR = "#9AA0A6"
CONNECTOR_COLOR = "#CDD2D8"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#263238",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def load_unique_ablation_results(
    primary_csv: Path = PRIMARY_CSV,
    supplementary_csv: Path = SUPPLEMENTARY_CSV,
) -> pd.DataFrame:
    """Load both CSV files and keep primary-table rows when ablation names repeat."""
    primary = pd.read_csv(primary_csv)
    supplementary = pd.read_csv(supplementary_csv)

    # Tag source priority before concatenation so duplicate ablation_name rows
    # retain the main paper table value over the supplementary table value.
    combined = pd.concat(
        [
            primary.assign(_source_priority=0),
            supplementary.assign(_source_priority=1),
        ],
        ignore_index=True,
        sort=False,
    )
    combined = combined.sort_values("_source_priority", kind="mergesort")
    combined = combined.drop_duplicates("ablation_name", keep="first")
    return combined.drop(columns=["_source_priority"])


def prepare_plot_data(
    primary_csv: Path = PRIMARY_CSV,
    supplementary_csv: Path = SUPPLEMENTARY_CSV,
) -> pd.DataFrame:
    combined = load_unique_ablation_results(primary_csv, supplementary_csv)
    required_columns = {"ablation_name"}
    for metric in METRICS:
        required_columns.add(metric["mean"])
        required_columns.add(metric["sd"])

    missing_columns = required_columns.difference(combined.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    missing_ablations = [
        name for name in TARGET_ABLATIONS if name not in set(combined["ablation_name"])
    ]
    if missing_ablations:
        raise ValueError(f"Missing required ablation rows: {missing_ablations}")

    selected = combined.loc[combined["ablation_name"].isin(TARGET_ABLATIONS)].copy()
    for metric in METRICS:
        # Mean columns define point positions; SD columns define horizontal xerr.
        selected[metric["mean"]] = pd.to_numeric(
            selected[metric["mean"]], errors="raise"
        )
        selected[metric["sd"]] = pd.to_numeric(selected[metric["sd"]], errors="raise")

    selected["ablation_label"] = selected["ablation_name"].map(ABLATION_LABELS)
    selected = selected.set_index("ablation_name").loc[TARGET_ABLATIONS].reset_index()
    return selected


def _metric_points(plot_data: pd.DataFrame, metric: dict[str, object]) -> dict[str, float]:
    mean_col = str(metric["mean"])
    sd_col = str(metric["sd"])
    indexed = plot_data.set_index("ablation_name")
    return {
        "motor_mean": float(indexed.loc["motor_wpli_edges_only", mean_col]),
        "motor_sd": float(indexed.loc["motor_wpli_edges_only", sd_col]),
        "minus_mean": float(indexed.loc["full_minus_motor_wpli_edges", mean_col]),
        "minus_sd": float(indexed.loc["full_minus_motor_wpli_edges", sd_col]),
        "reference": float(metric["reference"]),
    }


def make_plot(plot_data: pd.DataFrame, output_stem: Path = OUT_STEM) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.84, bottom=0.25)

    y_positions = np.arange(len(METRICS))[::-1]
    y_labels = [str(metric["label"]) for metric in METRICS]

    for y, metric in zip(y_positions, METRICS, strict=True):
        points = _metric_points(plot_data, metric)

        # Slope connector links the two ablation scores for the same metric.
        ax.hlines(
            y,
            points["motor_mean"],
            points["minus_mean"],
            color=CONNECTOR_COLOR,
            linewidth=1.2,
            zorder=1,
        )

        # Horizontal error bars use the metric-specific SD columns from the
        # merged ablation table.
        ax.errorbar(
            points["motor_mean"],
            y,
            xerr=points["motor_sd"],
            fmt="o",
            markersize=7.0,
            markerfacecolor=MOTOR_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.7,
            ecolor=MOTOR_COLOR,
            elinewidth=1.1,
            capsize=3.0,
            capthick=1.0,
            linestyle="none",
            zorder=4,
        )
        ax.errorbar(
            points["minus_mean"],
            y,
            xerr=points["minus_sd"],
            fmt="o",
            markersize=7.0,
            markerfacecolor=MINUS_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.7,
            ecolor=MINUS_COLOR,
            elinewidth=1.1,
            capsize=3.0,
            capthick=1.0,
            linestyle="none",
            zorder=5,
        )

        # Final model values are fixed references, not ablation rows, so they
        # are drawn as light-gray diamonds without SD error bars.
        ax.scatter(
            points["reference"],
            y,
            marker="D",
            s=42,
            facecolor=REFERENCE_COLOR,
            edgecolor="white",
            linewidth=0.7,
            zorder=6,
        )

        ax.annotate(
            f"{points['motor_mean']:.3f}",
            (points["motor_mean"], y),
            xytext=(5, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            color=MOTOR_COLOR,
            fontsize=7.5,
        )
        ax.annotate(
            f"{points['minus_mean']:.3f}",
            (points["minus_mean"], y),
            xytext=(5, -11),
            textcoords="offset points",
            ha="left",
            va="top",
            color=MINUS_COLOR,
            fontsize=7.5,
        )
        reference_label_offset = (5, 0)
        reference_label_ha = "left"
        if metric["label"] == "Brier score":
            reference_label_offset = (-6, 8)
            reference_label_ha = "right"

        ax.annotate(
            f"{points['reference']:.3f}",
            (points["reference"], y),
            xytext=reference_label_offset,
            textcoords="offset points",
            ha=reference_label_ha,
            va="bottom" if metric["label"] == "Brier score" else "center",
            color="#70757A",
            fontsize=7.5,
        )

    ax.set_title(
        "Network-constrained WPLI ablation",
        loc="left",
        pad=10,
        fontweight="bold",
    )
    ax.set_xlabel("Score")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.55, len(METRICS) - 0.45)

    ax.grid(axis="x", color="#E3E7EB", linewidth=0.7, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MOTOR_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7,
            label="Motor WPLI edges only",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MINUS_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7,
            label="Minus motor WPLI edges",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=REFERENCE_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.5,
            label="Final model reference",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    plot_data = prepare_plot_data()
    make_plot(plot_data)
    print(f"Wrote {OUT_STEM.with_suffix('.png')}")
    print(f"Wrote {OUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
