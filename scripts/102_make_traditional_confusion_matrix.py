from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


PREDICTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "predictions"
    / "final_Residual_ssl_cnn_10seed_patient_predictions.csv"
)
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "requested_triptychs"
OUT_STEM = OUT_DIR / "final_model_confusion_traditional_reference_style"
REVISED_OUT_DIR = PROJECT_ROOT / "results" / "figures" / "revised_initial"
REVISED_OUT_STEM = REVISED_OUT_DIR / "figure4b_final_model_confusion_matrix_traditional_style"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REVISED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    predictions = load_predictions()
    counts = confusion_counts(predictions)

    fig = make_traditional_confusion_matrix(counts)
    for stem in (OUT_STEM, REVISED_OUT_STEM):
        save_figure(fig, stem)
    plt.close(fig)

    for stem in (OUT_STEM, REVISED_OUT_STEM):
        write_source_data(counts, stem.with_suffix(".csv"))
    print(f"Wrote traditional confusion matrix to {OUT_STEM}")
    print(f"Wrote revised Figure 4B variant to {REVISED_OUT_STEM}")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 10,
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_predictions() -> pd.DataFrame:
    pred = pd.read_csv(PREDICTION_PATH)
    final = (
        pred.groupby("subject_id", as_index=False)
        .agg(y_true=("y_true", "first"), y_score=("y_score", "mean"))
        .sort_values("subject_id")
    )
    final["y_pred"] = (final["y_score"] >= 0.5).astype(int)
    return final


def confusion_counts(pred: pd.DataFrame) -> dict[str, int]:
    y_true = pred["y_true"].astype(int).to_numpy()
    y_pred = pred["y_pred"].astype(int).to_numpy()
    return {
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def make_traditional_confusion_matrix(counts: dict[str, int]) -> plt.Figure:
    values = np.array([[counts["TP"], counts["FN"]], [counts["FP"], counts["TN"]]], dtype=float)
    labels = np.array([["TP", "FN"], ["FP", "TN"]])
    total = values.sum()
    vmax = max(10, int(values.max()))

    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.045],
        left=0.17,
        right=0.91,
        bottom=0.11,
        top=0.79,
        wspace=0.24,
    )
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])

    cmap = LinearSegmentedColormap.from_list(
        "topomap_blue_white_red",
        ["#0B4F8A", "#F7F7F7", "#A50026"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=0, vcenter=vmax / 2, vmax=vmax)
    image = ax.imshow(values, cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Positive", "Negative"], fontsize=22)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="major", length=0, pad=10, labeltop=True, labelbottom=False)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Positive", "Negative"], fontsize=21, rotation=90, va="center")
    ax.tick_params(axis="y", which="major", length=0, pad=8)

    ax.set_ylabel("True label", fontsize=27, fontweight="bold", labelpad=55)
    fig.text(0.52, 0.935, "Predicted label", ha="center", va="center", fontsize=30, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=4.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#172033")
        spine.set_linewidth(1.3)
    ax.add_patch(Rectangle((-0.5, -0.5), 2, 2, fill=False, edgecolor="#172033", linewidth=1.3, clip_on=False))

    for row in range(2):
        for col in range(2):
            count = int(values[row, col])
            percentage = 100 * count / total if total else 0
            text_color = "white" if count in (0, vmax) or count >= 8 else "#101828"
            ax.text(
                col - 0.42,
                row - 0.36,
                labels[row, col],
                ha="left",
                va="top",
                fontsize=20,
                fontweight="bold",
                color=text_color,
            )
            ax.text(
                col,
                row - 0.01,
                f"{count}",
                ha="center",
                va="center",
                fontsize=38,
                fontweight="bold",
                color=text_color,
            )
            ax.text(
                col,
                row + 0.31,
                f"{percentage:.1f}% of total",
                ha="center",
                va="center",
                fontsize=13,
                color=text_color,
            )

    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Count", fontsize=14, labelpad=12)
    colorbar.set_ticks(np.arange(0, vmax + 0.1, 2))
    colorbar.ax.tick_params(labelsize=13, width=1.2, length=5)
    colorbar.outline.set_linewidth(1.2)

    return fig


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")


def write_source_data(counts: dict[str, int], path: Path) -> None:
    total = sum(counts.values())
    rows = [
        ("Positive", "Positive", "TP", counts["TP"]),
        ("Positive", "Negative", "FN", counts["FN"]),
        ("Negative", "Positive", "FP", counts["FP"]),
        ("Negative", "Negative", "TN", counts["TN"]),
    ]
    pd.DataFrame(
        [
            {
                "true_label": true_label,
                "predicted_label": predicted_label,
                "cell": cell,
                "count": count,
                "percent_of_total": count / total if total else np.nan,
            }
            for true_label, predicted_label, cell, count in rows
        ]
    ).to_csv(path, index=False)


if __name__ == "__main__":
    main()
