from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "results" / "tables" / "table2_main_model_performance.csv"
FIG_DIR = ROOT / "results" / "figures" / "revised_initial"
OUT_BASE = FIG_DIR / "ml_no_selector_5metric_polar_pastel"
SOURCE_PATH = FIG_DIR / "ml_no_selector_5metric_polar_source_data.csv"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


METRICS = [
    ("roc_auc", "AUC", "#9fc8df"),
    ("accuracy", "Accuracy", "#9bd8cf"),
    ("precision", "Precision", "#fa8a7d"),
    ("sensitivity", "Recall", "#b7df75"),
    ("f1", "F1 Score", "#c994c7"),
]

SELECTED_MODELS = [
    ("ML_EEG_updated_no_selector_svm_linear", "SVM"),
    ("ML_EEG_updated_no_selector_logistic_l1", "Logistic L1"),
    ("ML_EEG_updated_no_selector_logistic_l2", "Logistic L2"),
    ("ML_EEG_updated_no_selector_random_forest", "Random Forest"),
    ("ML_EEG_updated_no_selector_gaussian_nb", "Gaussian NB"),
    ("ML_EEG_updated_no_selector_knn", "KNN"),
]


def display_model_name(model_name: str) -> str:
    return dict(SELECTED_MODELS).get(model_name, model_name)


def build_plot_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, metric_label, color in METRICS:
        for rank, (model_name, model_display) in enumerate(SELECTED_MODELS, start=1):
            row = df.loc[df["model_name"].eq(model_name)]
            if row.empty:
                raise ValueError(f"Missing expected model row: {model_name}")
            row = row.iloc[0]
            rows.append(
                {
                    "metric": metric_label,
                    "metric_column": metric,
                    "rank": rank,
                    "model_name": row["model_name"],
                    "model_display": row["model_display"],
                    "feature_selection": row["feature_selection"],
                    "score": row[metric],
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["sensitivity"],
                    "f1": row["f1"],
                    "roc_auc": row["roc_auc"],
                    "color": color,
                }
            )
    return pd.DataFrame(rows)


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


def plot_polar(top5: pd.DataFrame) -> None:
    n_metrics = len(METRICS)
    top_n = len(SELECTED_MODELS)
    sector_width = 2 * np.pi / n_metrics
    group_width = sector_width * 0.72
    bar_width = group_width / top_n * 0.86
    inner_radius = 0.19

    fig = plt.figure(figsize=(8.4, 8.4))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.65)
    ax.set_xticks([])
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#555555")
    ax.set_rlabel_position(8)
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.7, color="#d7dce2")
    ax.spines["polar"].set_visible(False)

    for metric_idx, (metric_col, metric_label, color) in enumerate(METRICS):
        sector_center = metric_idx * sector_width
        sector_start = sector_center - sector_width / 2
        group_start = sector_center - group_width / 2
        subset = top5[top5["metric_column"] == metric_col].sort_values("rank")

        sector_thetas = np.linspace(group_start - 0.03, group_start + group_width + 0.03, 120)
        ax.plot(sector_thetas, np.full_like(sector_thetas, 1.045), color="black", lw=0.9)
        for tick_theta in [group_start - 0.03, group_start + group_width + 0.03]:
            ax.plot([tick_theta, tick_theta], [1.015, 1.075], color="black", lw=0.9)

        for rank_idx, (_, row) in enumerate(subset.iterrows()):
            theta = group_start + rank_idx * (group_width / top_n) + (group_width / top_n) / 2
            score = float(row["score"])
            height = max(score - inner_radius, 0.0)
            ax.bar(
                theta,
                height,
                width=bar_width,
                bottom=inner_radius,
                color=color,
                alpha=0.92,
                edgecolor="white",
                linewidth=1.1,
                zorder=3,
            )

            label_rot, label_ha = text_rotation(theta)
            ax.text(
                theta,
                1.095,
                row["model_display"],
                rotation=label_rot,
                rotation_mode="anchor",
                ha=label_ha,
                va="center",
                fontsize=6.2,
                color="#222222",
            )

        metric_theta = sector_center
        ax.text(
            metric_theta,
            1.555,
            metric_label,
            rotation=0,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.92},
            clip_on=False,
        )

    theta_full = np.linspace(0, 2 * np.pi, 720)
    ax.fill(theta_full, np.full_like(theta_full, inner_radius), color="white", zorder=5)
    ax.plot(theta_full, np.full_like(theta_full, inner_radius), color="black", lw=1.0, zorder=6)
    ax.text(
        0.5,
        0.5,
        "ML",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#333333",
        zorder=7,
    )

    fig.text(
        0.08,
        0.955,
        "Traditional EEG-ML model scorecard",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.928,
        "Each sector shows the same six traditional machine-learning models.",
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(TABLE_PATH)
    df = df[df["model_family"].eq("Traditional ML")].copy()
    df = df[df["model_name"].isin([m for m, _ in SELECTED_MODELS])].copy()
    df["model_display"] = df["model_name"].map(display_model_name)
    top5 = build_plot_table(df)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    top5.to_csv(SOURCE_PATH, index=False, encoding="utf-8-sig")
    plot_polar(top5)
    print(f"Saved figure base: {OUT_BASE}")
    print(f"Saved source data: {SOURCE_PATH}")


if __name__ == "__main__":
    main()
