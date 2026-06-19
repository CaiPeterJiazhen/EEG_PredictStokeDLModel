from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures" / "revised_initial"
METRIC_DIR = ROOT / "results" / "metrics"
DESKTOP_DIR = Path(r"C:\Users\HPGZZ\Desktop\小论文\论文图片")

ROC_SOURCE = FIG_DIR / "figure4a_final_model_roc_source_data.csv"
CM_SOURCE = FIG_DIR / "figure4b_final_model_confusion_matrix_traditional_style.csv"
LOSS_SOURCE = METRIC_DIR / "final_model_loss_curve_summary.csv"

OUT_BASE = FIG_DIR / "figure4_abc_final_model_performance_loss_native"


PALETTE = {
    "purple": "#8E44AD",
    "pink": "#F06C91",
    "blue": "#2F5E8C",
    "orange": "#E66A00",
    "teal": "#1B9E77",
    "gold": "#E3A43B",
    "ink": "#1F2D3A",
    "grid": "#DCE4EE",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.9,
            "axes.edgecolor": PALETTE["ink"],
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "legend.frameon": False,
        }
    )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.12) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#000000",
    )


def plot_roc(ax: plt.Axes) -> None:
    roc = pd.read_csv(ROC_SOURCE)
    fpr = roc["fpr"].to_numpy(dtype=float)
    tpr = roc["tpr"].to_numpy(dtype=float)
    auc = float(roc["roc_auc"].iloc[0])

    ax.step(
        fpr,
        tpr,
        where="post",
        color=PALETTE["purple"],
        lw=1.8,
        label=f"AUC = {auc:.2f}",
    )
    ax.plot([0, 1], [0, 1], color=PALETTE["pink"], lw=1.0, ls=(0, (3, 2)))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=9.5, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], lw=0.7, ls=":")
    leg = ax.legend(loc="lower right", fontsize=8.2, frameon=True)
    leg.get_frame().set_edgecolor(PALETTE["purple"])
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(1)
    add_panel_label(ax, "A", x=-0.18, y=1.18)


def plot_confusion(ax: plt.Axes, fig: plt.Figure) -> None:
    cm_df = pd.read_csv(CM_SOURCE)
    labels = ["Positive", "Negative"]
    counts = np.zeros((2, 2), dtype=float)
    percents = np.zeros((2, 2), dtype=float)
    cells = np.empty((2, 2), dtype=object)

    for _, row in cm_df.iterrows():
        i = labels.index(row["true_label"])
        j = labels.index(row["predicted_label"])
        counts[i, j] = row["count"]
        percents[i, j] = row["percent_of_total"]
        cells[i, j] = row["cell"]

    cmap = mpl.colormaps["RdBu_r"]
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=5, vmax=10)
    im = ax.imshow(counts, cmap=cmap, norm=norm, aspect="equal")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, rotation=90, va="center")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=2)
    ax.set_ylabel("True label", fontsize=9.5, fontweight="bold", labelpad=12)
    ax.set_title("Predicted label", fontsize=10.5, fontweight="bold", pad=24)

    for i in range(2):
        for j in range(2):
            value = counts[i, j]
            text_color = "white" if value >= 7 or value <= 1 else PALETTE["ink"]
            label_color = "white" if value >= 7 or value <= 1 else PALETTE["ink"]
            ax.text(
                j - 0.42,
                i - 0.35,
                str(cells[i, j]),
                ha="left",
                va="top",
                fontsize=9.2,
                fontweight="bold",
                color=label_color,
            )
            ax.text(
                j,
                i - 0.03,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color=text_color,
            )
            ax.text(
                j,
                i + 0.35,
                f"{percents[i, j] * 100:.1f}% of total",
                ha="center",
                va="center",
                fontsize=6.6,
                color=text_color,
            )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color(PALETTE["ink"])
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Count", rotation=90, labelpad=8)
    cbar.set_ticks([0, 2, 4, 6, 8, 10])
    cbar.outline.set_linewidth(0.8)
    add_panel_label(ax, "B", x=-0.22, y=1.18)


def plot_loss_panel(axes: list[plt.Axes]) -> None:
    summary = pd.read_csv(LOSS_SOURCE)
    colors = {
        "train_loss_total": "#1F77B4",
        "val_loss_total": "#FF7F0E",
        "bce": PALETTE["blue"],
        "residual": PALETTE["orange"],
        "ranking": "#7A6FB3",
        "soft": PALETTE["teal"],
    }

    ax = axes[0]
    for metric, label, color in [
        ("train_loss_total", "Training total", colors["train_loss_total"]),
        ("val_loss_total", "Validation total", colors["val_loss_total"]),
    ]:
        s = summary[summary["metric"] == metric]
        x = s["epoch"].to_numpy(dtype=float)
        y = s["value_mean"].to_numpy(dtype=float)
        e = s["value_sem"].to_numpy(dtype=float)
        ax.plot(x, y, lw=1.45, color=color, label=label)
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.16, lw=0)
    ax.axvline(50, color="#7A7A7A", lw=0.9, ls="--")
    ax.text(48.0, 1.08, "SWA", color="#555555", fontsize=7.0, ha="right")
    ax.set_title("Total loss", fontsize=9.0)
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", fontsize=6.7)

    ax = axes[1]
    for metric, label, color_key in [
        ("train_weighted_bce", "BCE", "bce"),
        ("train_weighted_residual", "Residual", "residual"),
        ("train_weighted_rank", "Ranking", "ranking"),
        ("train_weighted_soft", "Soft label", "soft"),
    ]:
        s = summary[summary["metric"] == metric]
        ax.plot(s["epoch"], s["value_mean"], lw=1.25, color=colors[color_key], label=label)
    ax.axvline(50, color="#7A7A7A", lw=0.9, ls="--")
    ax.set_title("Weighted training components", fontsize=9.0)
    ax.set_ylabel("Weighted contribution")
    ax.legend(loc="upper right", fontsize=6.6)

    ax = axes[2]
    for metric, label, color_key in [
        ("val_weighted_bce", "BCE", "bce"),
        ("val_weighted_residual", "Residual", "residual"),
        ("val_weighted_rank", "Ranking", "ranking"),
        ("val_weighted_soft", "Soft label", "soft"),
    ]:
        s = summary[summary["metric"] == metric]
        ax.plot(s["epoch"], s["value_mean"], lw=1.25, color=colors[color_key], label=label)
    ax.axvline(50, color="#7A7A7A", lw=0.9, ls="--")
    ax.set_title("Weighted validation components", fontsize=9.0)
    ax.set_ylabel("Weighted contribution")
    ax.legend(loc="upper right", fontsize=6.6)

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.set_xlim(0, 105)


def main() -> None:
    setup_style()
    missing = [p for p in [ROC_SOURCE, CM_SOURCE, LOSS_SOURCE] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(str(p) for p in missing))

    fig = plt.figure(figsize=(8.4, 6.4), dpi=600, facecolor="white")
    outer = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.12, 0.88],
        width_ratios=[1.0, 1.0],
        hspace=0.55,
        wspace=0.34,
    )

    ax_roc = fig.add_subplot(outer[0, 0])
    ax_cm = fig.add_subplot(outer[0, 1])
    loss_grid = outer[1, :].subgridspec(1, 3, wspace=0.33)
    loss_axes = [fig.add_subplot(loss_grid[0, i]) for i in range(3)]

    plot_roc(ax_roc)
    plot_confusion(ax_cm, fig)
    plot_loss_panel(loss_axes)
    add_panel_label(loss_axes[0], "C", x=-0.18, y=1.32)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DESKTOP_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".svg", ".tiff"]:
        out = OUT_BASE.with_suffix(suffix)
        if suffix in {".png", ".tiff"}:
            fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(out, bbox_inches="tight", facecolor="white")
        shutil.copy2(out, DESKTOP_DIR / out.name)

    plt.close(fig)
    print(f"Saved {OUT_BASE.with_suffix('.png')}")
    print(f"Copied outputs to {DESKTOP_DIR}")


if __name__ == "__main__":
    main()
