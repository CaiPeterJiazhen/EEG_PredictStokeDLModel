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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


FINAL_MODEL = "residual_aware_SSL_CNN_seedmean10"
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "requested_triptychs"

COLORS = {
    "ink": "#1F2933",
    "grid": "#D9E1E8",
    "correct": "#5B8C62",
    "correct_light": "#E8F1E9",
    "error": "#B84A39",
    "error_light": "#F6E7E3",
    "prop": "#B84A39",
    "poor": "#4D7895",
    "neutral": "#F5F7F8",
    "metric": "#7A6C9E",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    pred = load_final_predictions()
    counts = confusion_counts(pred)
    metrics = compute_metrics(counts)

    fig1 = make_lollipop_confusion(pred, counts, metrics)
    save_pub(fig1, OUT_DIR / "final_model_confusion_alt_lollipop")
    plt.close(fig1)

    fig2 = make_patient_dot_confusion(pred, counts, metrics)
    save_pub(fig2, OUT_DIR / "final_model_confusion_alt_patient_dots")
    plt.close(fig2)

    fig3 = make_side_by_side_preview(pred, counts, metrics)
    save_pub(fig3, OUT_DIR / "final_model_confusion_alternative_designs")
    plt.close(fig3)

    write_source_table(pred, counts, metrics)
    print(f"Wrote confusion matrix alternatives to {OUT_DIR}")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.titlesize": 9,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
        }
    )


def load_final_predictions() -> pd.DataFrame:
    path = PROJECT_ROOT / "results" / "predictions" / "paper_locked_model_predictions.csv"
    pred = pd.read_csv(path)
    final = pred[pred["model_name"] == FINAL_MODEL].copy()
    if final.empty:
        raise FileNotFoundError(f"{FINAL_MODEL} was not found in {path}")
    final["observed_group"] = np.where(final["y_true"].astype(int) == 1, "Proportional recovery", "Poor recovery")
    final["predicted_group"] = np.where(final["y_pred"].astype(int) == 1, "Predicted proportional", "Predicted poor")
    final["correct"] = final["y_true"].astype(int) == final["y_pred"].astype(int)
    return final.sort_values("subject_id").reset_index(drop=True)


def confusion_counts(pred: pd.DataFrame) -> dict[str, int]:
    y_true = pred["y_true"].astype(int).to_numpy()
    y_pred = pred["y_pred"].astype(int).to_numpy()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def compute_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = counts["tp"], counts["fn"], counts["fp"], counts["tn"]
    return {
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "Precision": tp / (tp + fp) if tp + fp else np.nan,
        "NPV": tn / (tn + fn) if tn + fn else np.nan,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
    }


def make_lollipop_confusion(pred: pd.DataFrame, counts: dict[str, int], metrics: dict[str, float]) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 3.75), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    draw_core_matrix(ax, counts)
    draw_metric_lollipop(ax_m, metrics)
    fig.suptitle("Final model classification summary", x=0.52, y=1.02, fontsize=10)
    return fig


def draw_core_matrix(ax: plt.Axes, counts: dict[str, int]) -> None:
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5], ["Predicted\nproportional", "Predicted\npoor"])
    ax.set_yticks([1.5, 0.5], ["Observed\nproportional", "Observed\npoor"])
    ax.tick_params(length=0)
    ax.set_title("Patient counts and row percentages")
    for spine in ax.spines.values():
        spine.set_visible(False)

    cells = [
        (0, 1, "tp", True, "True positive"),
        (1, 1, "fn", False, "False negative"),
        (0, 0, "fp", False, "False positive"),
        (1, 0, "tn", True, "True negative"),
    ]
    row_totals = {"tp": counts["tp"] + counts["fn"], "fn": counts["tp"] + counts["fn"], "fp": counts["fp"] + counts["tn"], "tn": counts["fp"] + counts["tn"]}
    for x, y, key, correct, label in cells:
        count = counts[key]
        total = row_totals[key]
        pct = 100 * count / total if total else 0
        face = COLORS["correct_light"] if correct else COLORS["error_light"]
        edge = COLORS["correct"] if correct else COLORS["error"]
        ax.add_patch(Rectangle((x + 0.04, y + 0.04), 0.92, 0.92, facecolor=face, edgecolor=edge, linewidth=1.2))
        ax.text(x + 0.5, y + 0.66, label, ha="center", va="center", fontsize=6.2, color=edge)
        ax.text(x + 0.5, y + 0.47, f"{count}/{total}", ha="center", va="center", fontsize=18, fontweight="bold", color=COLORS["ink"])
        ax.text(x + 0.5, y + 0.25, f"{pct:.1f}%", ha="center", va="center", fontsize=7, color=COLORS["ink"])

    ax.text(-0.08, 2.08, "A", ha="left", va="top", fontsize=10, fontweight="bold", transform=ax.transData)


def draw_metric_lollipop(ax: plt.Axes, metrics: dict[str, float]) -> None:
    labels = ["Sensitivity", "Specificity", "Precision", "NPV", "Accuracy"]
    values = np.array([metrics[label] for label in labels], dtype=float)
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, 0, values, color=COLORS["grid"], linewidth=5, zorder=1)
    ax.scatter(values, y, s=58, color=[COLORS["correct"], COLORS["correct"], COLORS["metric"], COLORS["metric"], COLORS["error"]], zorder=3)
    for yi, value in zip(y, values, strict=True):
        ax.text(value + 0.035, yi, f"{value:.2f}", va="center", ha="left", fontsize=7, color=COLORS["ink"])
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Metric value")
    ax.set_title("Derived performance metrics")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.06, "B", ha="left", va="top", fontsize=10, fontweight="bold", transform=ax.transAxes)


def make_patient_dot_confusion(pred: pd.DataFrame, counts: dict[str, int], metrics: dict[str, float]) -> plt.Figure:
    fig = plt.figure(figsize=(6.1, 4.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    draw_patient_dot_panel(ax, pred, counts, metrics)
    return fig


def draw_patient_dot_panel(
    ax: plt.Axes,
    pred: pd.DataFrame,
    counts: dict[str, int],
    metrics: dict[str, float],
    *,
    panel_letter: str | None = None,
) -> None:
    rng = np.random.default_rng(20260611)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.55, 1.55)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([0, 1], ["Predicted\nproportional", "Predicted\npoor"])
    ax.set_yticks([1, 0], ["Observed\nproportional", "Observed\npoor"])
    ax.tick_params(length=0)
    ax.set_title(f"Patient-level confusion plot (accuracy = {metrics['Accuracy']:.2f})")

    zone_specs = [
        (-0.42, 0.58, COLORS["correct_light"], COLORS["correct"]),
        (0.58, 0.58, COLORS["error_light"], COLORS["error"]),
        (-0.42, -0.42, COLORS["error_light"], COLORS["error"]),
        (0.58, -0.42, COLORS["correct_light"], COLORS["correct"]),
    ]
    for x, y, face, edge in zone_specs:
        ax.add_patch(Rectangle((x, y), 0.84, 0.84, facecolor=face, edgecolor=edge, linewidth=1.0, zorder=0))

    x_map = {1: 0, 0: 1}
    y_map = {1: 1, 0: 0}
    color_map = {1: COLORS["prop"], 0: COLORS["poor"]}
    for _, row in pred.iterrows():
        x = x_map[int(row["y_pred"])] + rng.uniform(-0.22, 0.22)
        y = y_map[int(row["y_true"])] + rng.uniform(-0.22, 0.22)
        marker = "o" if row["correct"] else "X"
        size = 54 if row["correct"] else 66
        edge = "white" if row["correct"] else COLORS["ink"]
        ax.scatter(x, y, s=size, marker=marker, color=color_map[int(row["y_true"])], edgecolor=edge, linewidth=0.55, zorder=3)
        ax.text(
            x,
            y - 0.065,
            str(row["subject_id"]).replace("sub", ""),
            ha="center",
            va="top",
            fontsize=4.6,
            color=COLORS["ink"],
            alpha=0.82,
            zorder=4,
        )

    annotate_quadrants(ax, counts)
    if panel_letter:
        ax.text(-0.52, 1.55, panel_letter, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["prop"], markeredgecolor="white", markersize=7, label="Observed proportional"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["poor"], markeredgecolor="white", markersize=7, label="Observed poor"),
            Line2D([0], [0], marker="X", color="none", markerfacecolor=COLORS["ink"], markeredgecolor=COLORS["ink"], markersize=7, label="Misclassified"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=3,
    )


def annotate_quadrants(ax: plt.Axes, counts: dict[str, int]) -> None:
    entries = [
        (0, 1, counts["tp"], counts["tp"] + counts["fn"], COLORS["correct"]),
        (1, 1, counts["fn"], counts["tp"] + counts["fn"], COLORS["error"]),
        (0, 0, counts["fp"], counts["fp"] + counts["tn"], COLORS["error"]),
        (1, 0, counts["tn"], counts["fp"] + counts["tn"], COLORS["correct"]),
    ]
    for x, y, count, total, color in entries:
        pct = 100 * count / total if total else 0
        ax.text(x + 0.31, y + 0.31, f"{count}/{total}\n{pct:.0f}%", ha="right", va="top", fontsize=7.5, color=color, fontweight="bold")


def make_side_by_side_preview(pred: pd.DataFrame, counts: dict[str, int], metrics: dict[str, float]) -> plt.Figure:
    fig = plt.figure(figsize=(10.4, 4.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 0.72, 1.05], wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    draw_core_matrix(ax_a, counts)
    draw_metric_lollipop(ax_b, metrics)
    draw_patient_dot_panel(ax_c, pred, counts, metrics, panel_letter="C")
    fig.suptitle("Alternative confusion-matrix designs", x=0.51, y=1.02, fontsize=10)
    return fig


def save_pub(fig: plt.Figure, base: Path) -> None:
    for suffix in (".png", ".svg", ".pdf", ".tiff"):
        path = base.with_suffix(suffix)
        if suffix in {".png", ".tiff"}:
            fig.savefig(path, dpi=400, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")


def write_source_table(pred: pd.DataFrame, counts: dict[str, int], metrics: dict[str, float]) -> None:
    pred[["subject_id", "y_true", "y_pred", "y_score", "observed_group", "predicted_group", "correct"]].to_csv(
        OUT_DIR / "final_model_confusion_alternatives_patient_source.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"type": "count", "name": key, "value": value}
            for key, value in counts.items()
        ]
        + [
            {"type": "metric", "name": key, "value": value}
            for key, value in metrics.items()
        ]
    ).to_csv(OUT_DIR / "final_model_confusion_alternatives_summary.csv", index=False)


if __name__ == "__main__":
    main()
