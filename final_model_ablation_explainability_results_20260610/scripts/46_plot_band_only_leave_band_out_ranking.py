from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import transforms
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
PRIMARY_CSV = TABLE_DIR / "table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv"
SUPPLEMENTARY_CSV = TABLE_DIR / "supplementary_final_ssl_cnn_feature_state_band_ablation.csv"
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "paper_panels"
OUT_STEM = OUT_DIR / "figure5c_band_only_leave_band_out_ranking"

LEAVE_BAND_OUT = [
    "full_minus_delta",
    "full_minus_theta",
    "full_minus_alpha",
    "full_minus_beta_low",
    "full_minus_beta_medium",
    "full_minus_beta_high",
    "full_minus_psd_gamma",
    "full_minus_beta_medium_beta_high",
]
BAND_ONLY = [
    "delta_only",
    "theta_only",
    "alpha_only",
    "beta_low_only",
    "beta_medium_only",
    "beta_high_only",
    "psd_gamma_only",
    "beta_medium_beta_high",
]
TARGET_ABLATIONS = LEAVE_BAND_OUT + BAND_ONLY
DISPLAY_LABELS = {
    "delta_only": "Delta only",
    "theta_only": "Theta only",
    "alpha_only": "Alpha only",
    "beta_low_only": "Beta Low only",
    "beta_medium_only": "Beta Medium only",
    "beta_high_only": "Beta High only",
    "psd_gamma_only": "PSD Gamma only",
    "beta_medium_beta_high": "Beta Medium + Beta High only",
    "full_minus_delta": "Minus Delta",
    "full_minus_theta": "Minus Theta",
    "full_minus_alpha": "Minus Alpha",
    "full_minus_beta_low": "Minus Beta Low",
    "full_minus_beta_medium": "Minus Beta Medium",
    "full_minus_beta_high": "Minus Beta High",
    "full_minus_psd_gamma": "Minus PSD Gamma",
    "full_minus_beta_medium_beta_high": "Minus Beta Medium + Beta High",
}
BALANCED_ACCURACY_COLOR = "#2F6FAE"
ROC_AUC_COLOR = "#C9473D"
FINAL_BALANCED_ACCURACY = 0.8411
FINAL_ROC_AUC = 0.8867


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#263238",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_unique_ablation_results(primary_csv: Path, supplementary_csv: Path) -> pd.DataFrame:
    primary = pd.read_csv(primary_csv)
    supplementary = pd.read_csv(supplementary_csv)
    combined = pd.concat(
        [
            primary.assign(_source_priority=0),
            supplementary.assign(_source_priority=1),
        ],
        ignore_index=True,
        sort=False,
    )
    combined = combined.sort_values("_source_priority").drop_duplicates("ablation_name", keep="first")
    return combined.drop(columns=["_source_priority"])


def load_and_prepare_plot_data(primary_csv: Path = PRIMARY_CSV, supplementary_csv: Path = SUPPLEMENTARY_CSV) -> pd.DataFrame:
    combined = load_unique_ablation_results(primary_csv, supplementary_csv)
    required_columns = {"ablation_name", "balanced_accuracy_mean", "roc_auc_mean"}
    missing_columns = required_columns.difference(combined.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    missing_ablations = [name for name in TARGET_ABLATIONS if name not in set(combined["ablation_name"])]
    if missing_ablations:
        raise ValueError(f"Missing required ablation rows: {missing_ablations}")

    selected = combined.loc[combined["ablation_name"].isin(TARGET_ABLATIONS)].copy()
    selected["balanced_accuracy_mean"] = pd.to_numeric(selected["balanced_accuracy_mean"], errors="raise")
    selected["roc_auc_mean"] = pd.to_numeric(selected["roc_auc_mean"], errors="raise")
    selected["block"] = np.where(selected["ablation_name"].isin(LEAVE_BAND_OUT), "Leave-band-out", "Band-only")
    selected["display_label"] = selected["ablation_name"].map(DISPLAY_LABELS)

    leave = selected.loc[selected["block"] == "Leave-band-out"].sort_values("roc_auc_mean", ascending=False)
    band = selected.loc[selected["block"] == "Band-only"].sort_values("roc_auc_mean", ascending=False)
    return pd.concat([leave, band], ignore_index=True)


def make_plot(plot_data: pd.DataFrame, output_stem: Path = OUT_STEM) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)

    y_positions = np.arange(len(plot_data))[::-1]
    balanced_accuracy = plot_data["balanced_accuracy_mean"].to_numpy(dtype=float)
    roc_auc = plot_data["roc_auc_mean"].to_numpy(dtype=float)

    for y, bal, roc in zip(y_positions, balanced_accuracy, roc_auc, strict=True):
        ax.hlines(y, min(bal, roc), max(bal, roc), color="#C7CDD4", linewidth=1.15, zorder=1)

    ax.scatter(
        balanced_accuracy,
        y_positions,
        s=38,
        color=BALANCED_ACCURACY_COLOR,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
        label="Balanced accuracy",
    )
    ax.scatter(
        roc_auc,
        y_positions,
        s=38,
        color=ROC_AUC_COLOR,
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
        label="ROC-AUC",
    )

    ax.axvline(
        FINAL_BALANCED_ACCURACY,
        color="#7C838A",
        linestyle=(0, (3.0, 2.5)),
        linewidth=0.8,
        alpha=0.55,
        zorder=0,
    )
    ax.axvline(
        FINAL_ROC_AUC,
        color="#7C838A",
        linestyle=(0, (1.0, 2.2)),
        linewidth=0.8,
        alpha=0.55,
        zorder=0,
    )

    separator_y = (y_positions[7] + y_positions[8]) / 2
    ax.axhline(separator_y, color="#D3D8DE", linewidth=0.85, zorder=0)

    label_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.015,
        float(np.mean(y_positions[:8])),
        "Leave-band-out",
        transform=label_transform,
        ha="left",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="#5A626B",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    ax.text(
        0.015,
        float(np.mean(y_positions[8:])),
        "Band-only",
        transform=label_transform,
        ha="left",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="#5A626B",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    ax.set_title("Band-only and leave-band-out ablation ranking", loc="left", pad=8, fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_xlim(0.2, 1.0)
    ax.set_xticks(np.arange(0.2, 1.01, 0.1))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["display_label"])
    ax.set_ylim(-0.65, len(plot_data) - 0.35)
    ax.grid(axis="x", color="#E3E7EB", linewidth=0.55, zorder=0)
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
            markerfacecolor=BALANCED_ACCURACY_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=5.5,
            label="Balanced accuracy",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ROC_AUC_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=5.5,
            label="ROC-AUC",
        ),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    plot_data = load_and_prepare_plot_data()
    make_plot(plot_data)
    print(f"png={OUT_STEM.with_suffix('.png')}")
    print(f"pdf={OUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
