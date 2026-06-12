from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures" / "revised_initial"
POLAR_SOURCE = FIG_DIR / "ml_no_selector_5metric_polar_source_data.csv"
ML_TABLE = ROOT / "results" / "tables" / "eeg_only_ml_three_line_table.csv"
OUT_BASE = FIG_DIR / "figure_ml_baseline_two_panel"
SOURCE_OUT = FIG_DIR / "figure_ml_baseline_two_panel_source_data.csv"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.5,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#263238",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


METRICS_A = [
    ("roc_auc", "AUC", "#9fc8df"),
    ("accuracy", "Accuracy", "#9bd8cf"),
    ("precision", "Precision", "#fa8a7d"),
    ("sensitivity", "Recall", "#b7df75"),
    ("f1", "F1 score", "#c994c7"),
]

KEY_BASELINES = [
    ("Logistic L1 (No selector)", "Logistic L1", "#3B6EA8"),
    ("Logistic L2 (No selector)", "Logistic L2", "#5AB4AC"),
    ("SVM linear (No selector)", "SVM linear", "#8E7CC3"),
    ("Random forest (No selector)", "Random forest", "#D9A441"),
    ("Gaussian NB (No selector)", "Gaussian NB", "#D95F59"),
    ("KNN (No selector)", "KNN", "#6A9C78"),
]

A_LABELS = {
    "Logistic L1": "Log. L1",
    "Logistic L2": "Log. L2",
    "Random Forest": "RF",
    "Gaussian NB": "GNB",
    "KNN": "KNN",
    "SVM": "SVM",
}


def text_rotation(theta: float) -> tuple[float, str]:
    angle = np.degrees(theta)
    rotation = 90 - angle
    ha = "left"
    if rotation < -90:
        rotation += 180
        ha = "right"
    if rotation > 90:
        rotation -= 180
        ha = "right"
    return rotation, ha


def draw_polar_panel(ax: plt.Axes, polar_source: pd.DataFrame) -> None:
    n_metrics = len(METRICS_A)
    model_order = list(polar_source["model_display"].drop_duplicates())
    top_n = len(model_order)
    sector_width = 2 * np.pi / n_metrics
    group_width = sector_width * 0.72
    bar_width = group_width / top_n * 0.86
    inner_radius = 0.19

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.62)
    ax.set_xticks([])
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], fontsize=6.6, color="#56616B")
    ax.set_rlabel_position(8)
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.55, color="#D8DEE6")
    ax.spines["polar"].set_visible(False)

    for metric_idx, (metric_col, metric_label, color) in enumerate(METRICS_A):
        sector_center = metric_idx * sector_width
        group_start = sector_center - group_width / 2
        subset = polar_source[polar_source["metric_column"].eq(metric_col)].sort_values("rank")

        sector_thetas = np.linspace(group_start - 0.03, group_start + group_width + 0.03, 120)
        ax.plot(sector_thetas, np.full_like(sector_thetas, 1.045), color="#202020", lw=0.75)
        for tick_theta in [group_start - 0.03, group_start + group_width + 0.03]:
            ax.plot([tick_theta, tick_theta], [1.015, 1.073], color="#202020", lw=0.75)

        for rank_idx, (_, row) in enumerate(subset.iterrows()):
            theta = group_start + rank_idx * (group_width / top_n) + (group_width / top_n) / 2
            score = float(row["score"])
            ax.bar(
                theta,
                max(score - inner_radius, 0.0),
                width=bar_width,
                bottom=inner_radius,
                color=color,
                alpha=0.93,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

            label_rot, label_ha = text_rotation(theta)
            ax.text(
                theta,
                1.095,
                A_LABELS.get(str(row["model_display"]), str(row["model_display"])),
                rotation=label_rot,
                rotation_mode="anchor",
                ha=label_ha,
                va="center",
                fontsize=5.15,
                color="#222222",
            )

        text_offset = (15, -1) if metric_label == "Accuracy" else (0, 0)
        ax.annotate(
            metric_label,
            xy=(sector_center, 1.50),
            xycoords="data",
            xytext=text_offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.65, "alpha": 0.95},
            clip_on=False,
        )

    theta_full = np.linspace(0, 2 * np.pi, 720)
    ax.fill(theta_full, np.full_like(theta_full, inner_radius), color="white", zorder=5)
    ax.plot(theta_full, np.full_like(theta_full, inner_radius), color="#222222", lw=0.9, zorder=6)
    ax.text(
        0.5,
        0.5,
        "ML",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#333333",
        zorder=7,
    )


def load_key_baselines() -> pd.DataFrame:
    frame = pd.read_csv(ML_TABLE)
    rows = []
    for model, display, color in KEY_BASELINES:
        hit = frame.loc[frame["模型"].eq(model)]
        if hit.empty:
            raise ValueError(f"Missing baseline model row: {model}")
        row = hit.iloc[0].to_dict()
        row["model"] = model
        row["display"] = display
        row["color"] = color
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Brier", ascending=True).reset_index(drop=True)


def draw_baseline_panel(fig: plt.Figure, gs, baseline: pd.DataFrame) -> plt.Axes:
    ax_brier = fig.add_subplot(gs)
    y = np.arange(len(baseline))
    brier = baseline["Brier"].astype(float).to_numpy()
    x_ref = 0.25
    ax_brier.hlines(y, x_ref, brier, color="#D6DEE8", lw=2.0, zorder=1)
    ax_brier.scatter(
        brier,
        y,
        s=82,
        color=list(baseline["color"]),
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    for yi, val in zip(y, brier, strict=True):
        ax_brier.text(val + 0.006, yi, f"{val:.3f}", va="center", ha="left", fontsize=6.8, color="#1F2933")

    ax_brier.set_yticks(y)
    ax_brier.set_yticklabels(baseline["display"], fontsize=7.0)
    ax_brier.set_xlim(0.18, 0.39)
    ax_brier.set_xticks([0.20, 0.25, 0.30, 0.35])
    ax_brier.set_xlabel("Brier score", fontsize=7.2)
    ax_brier.grid(axis="x", color="#E5E9EF", lw=0.8)
    ax_brier.axvline(x_ref, color="#BAC6D3", lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax_brier.spines["top"].set_visible(False)
    ax_brier.spines["right"].set_visible(False)
    ax_brier.tick_params(axis="y", length=0, pad=5)
    ax_brier.invert_yaxis()
    return ax_brier


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    polar_source = pd.read_csv(POLAR_SOURCE)
    baseline = load_key_baselines()

    source_rows = []
    for _, row in polar_source.iterrows():
        source_rows.append(
            {
                "panel": "A",
                "model": row["model_display"],
                "setting": row["feature_selection"],
                "metric": row["metric"],
                "value": row["score"],
            }
        )
    for _, row in baseline.iterrows():
        source_rows.append(
            {
                "panel": "B",
                "model": row["display"].replace("\n", " "),
                "setting": row["model"],
                "metric": "Brier",
                "value": row["Brier"],
            }
        )
    pd.DataFrame(source_rows).to_csv(SOURCE_OUT, index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.7, 3.95), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        left=0.055,
        right=0.972,
        top=0.94,
        bottom=0.15,
        wspace=0.30,
    )
    ax_a = fig.add_subplot(gs[0, 0], polar=True)
    draw_polar_panel(ax_a, polar_source)
    ax_b = draw_baseline_panel(fig, gs[0, 1], baseline)

    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    ax_b.set_position([pos_b.x0, pos_a.y0, pos_b.width * 0.88, pos_a.height])

    fig.text(0.020, 0.940, "A", fontsize=9.5, fontweight="bold", va="top", ha="left")
    fig.text(0.480, 0.940, "B", fontsize=9.5, fontweight="bold", va="top", ha="left")

    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    print(f"saved={OUT_BASE}")
    print(f"source={SOURCE_OUT}")


if __name__ == "__main__":
    main()
