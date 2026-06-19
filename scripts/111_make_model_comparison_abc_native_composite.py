from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "revised_initial"
DESKTOP_FIG_DIR = Path(r"C:\Users\HPGZZ\Desktop\小论文\论文图片")

SCORECARD_SOURCE = OUT_DIR / "figure4c_a_model_metric_histogram_source_data.csv"
SEED_SOURCE = OUT_DIR / "figure5a_seed_stability_source_data.csv"
BRIER_SOURCE = OUT_DIR / "figure4c_b_brier_calibration_source_data.csv"

OUT_STEM = OUT_DIR / "figure5_abc_model_scorecard_seed_brier_native"
DESKTOP_OUT_STEM = DESKTOP_FIG_DIR / "figure5_abc_model_scorecard_seed_brier_native"

METRICS = [
    ("accuracy", "Mean\naccuracy"),
    ("balanced_accuracy", "Balanced\naccuracy"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]

SCORECARD_ORDER = [
    "Logistic L1",
    "No-SSL CNN",
    "Barlow CNN",
    "Residual-aware CNN",
    "Residual-aware SSL-CNN",
]

SCORECARD_COLORS = {
    "Logistic L1": "#4E79A7",
    "No-SSL CNN": "#F28E2B",
    "Barlow CNN": "#D95F59",
    "Residual-aware CNN": "#7B6AB0",
    "Residual-aware SSL-CNN": "#2EAD74",
}

SEED_ORDER = ["No-SSL CNN", "Barlow CNN", "No-SSL residual-aware", "Residual-aware SSL-CNN"]
SEED_COLORS = {
    "No-SSL CNN": "#4C78A8",
    "Barlow CNN": "#4E8D7C",
    "No-SSL residual-aware": "#D9A43A",
    "Residual-aware SSL-CNN": "#D66A5C",
}
SEED_XLABELS = ["No-SSL\nCNN", "Barlow\nSSL-CNN", "No-SSL\nresidual", "Final\nSSL-CNN"]

BRIER_COLORS = {
    "Residual-aware CNN": "#D9A43A",
    "Residual-aware\nSSL-CNN": "#2EAD74",
    "Barlow CNN": "#7B6AB0",
    "No-SSL CNN": "#F28E2B",
    "Logistic L1": "#4E79A7",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="black",
    )


def plot_scorecard(ax: plt.Axes, source: pd.DataFrame) -> None:
    data = source.copy()
    data["model"] = pd.Categorical(data["model"], categories=SCORECARD_ORDER, ordered=True)
    data = data.sort_values("model").reset_index(drop=True)

    x = np.arange(len(METRICS), dtype=float)
    width = 0.13
    offsets = (np.arange(len(data)) - (len(data) - 1) / 2) * width

    for offset, (_, row) in zip(offsets, data.iterrows(), strict=True):
        model = str(row["model"])
        values = [float(row[metric]) for metric, _ in METRICS]
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=SCORECARD_COLORS[model],
            edgecolor="white",
            linewidth=0.45,
            label=model,
            zorder=3,
        )
        if model == "Residual-aware SSL-CNN":
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.012,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=5.4,
                    color=SCORECARD_COLORS[model],
                )

    ax.set_title("Patient-level model scorecard", loc="center", pad=7)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS])
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.58, zorder=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        ncol=3,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.35,
        borderaxespad=0.0,
    )
    panel_label(ax, "A", x=-0.065, y=1.03)


def plot_seed_stability(ax: plt.Axes, source: pd.DataFrame) -> None:
    positions = np.arange(1, len(SEED_ORDER) + 1)
    grouped = [
        source.loc[source["model"].eq(label), "accuracy"].astype(float).to_numpy()
        for label in SEED_ORDER
    ]

    violins = ax.violinplot(grouped, positions=positions, widths=0.70, showmeans=False, showmedians=False, showextrema=False)
    for body, label in zip(violins["bodies"], SEED_ORDER, strict=True):
        color = SEED_COLORS[label]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.16)
        body.set_linewidth(0.9)

    for pos, label, values in zip(positions, SEED_ORDER, grouped, strict=True):
        color = SEED_COLORS[label]
        values = np.asarray(values, dtype=float)
        ax.vlines(pos, values.min(), values.max(), color=color, linewidth=1.15, zorder=2)
        ax.hlines([values.min(), values.max()], pos - 0.17, pos + 0.17, color=color, linewidth=1.15, zorder=2)
        jitter = np.linspace(-0.06, 0.06, len(values))
        ax.scatter(
            np.full_like(values, pos, dtype=float) + jitter,
            values,
            s=17,
            color=color,
            edgecolor="white",
            linewidth=0.42,
            zorder=3,
        )
        ax.scatter(
            [pos],
            [np.median(values)],
            marker="D",
            s=31,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=4,
        )

    ax.set_title("Random-seed stability", loc="left", pad=6)
    ax.set_ylabel("Seed-level accuracy")
    ax.set_xticks(positions)
    ax.set_xticklabels(SEED_XLABELS)
    ax.set_ylim(0.50, 0.93)
    ax.set_yticks(np.arange(0.50, 0.95, 0.05))
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.56, zorder=0)
    ax.text(0.02, 0.02, "Diamonds mark medians", transform=ax.transAxes, fontsize=6.0, color="#667085")
    panel_label(ax, "B", x=-0.13, y=1.04)


def plot_brier(ax: plt.Axes, source: pd.DataFrame) -> None:
    data = source.copy().sort_values("brier_score", ascending=True).reset_index(drop=True)
    y = np.arange(len(data))
    values = data["brier_score"].astype(float).to_numpy()
    labels = data["display"].tolist()
    colors = [BRIER_COLORS.get(label, "#4E79A7") for label in labels]

    ax.hlines(y, 0, values, color="#D8DEE6", linewidth=1.0, zorder=1)
    ax.scatter(values, y, s=36, color=colors, edgecolor="white", linewidth=0.55, zorder=3)
    for yi, value in zip(y, values, strict=True):
        ax.text(value + 0.008, yi, f"{value:.3f}", va="center", ha="left", fontsize=6.2, color="#263238")

    ax.set_title("Brier score comparison", loc="left", pad=6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.245, float(values.max()) + 0.035))
    ax.set_xticks(np.arange(0, 0.26, 0.05))
    ax.set_xlabel("Brier score")
    ax.grid(axis="x", color="#D8DEE6", linewidth=0.52, zorder=0)
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.52, zorder=0)
    ax.tick_params(axis="y", length=2.5)
    panel_label(ax, "C", x=-0.16, y=1.04)


def save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    configure_matplotlib()
    scorecard = pd.read_csv(SCORECARD_SOURCE)
    seed = pd.read_csv(SEED_SOURCE)
    brier = pd.read_csv(BRIER_SOURCE)

    fig = plt.figure(figsize=(8.2, 6.85), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.06, 1.0],
        hspace=0.58,
        wspace=0.38,
        left=0.07,
        right=0.985,
        top=0.955,
        bottom=0.078,
    )
    ax_score = fig.add_subplot(gs[0, :])
    ax_seed = fig.add_subplot(gs[1, 0])
    ax_brier = fig.add_subplot(gs[1, 1])

    plot_scorecard(ax_score, scorecard)
    plot_seed_stability(ax_seed, seed)
    plot_brier(ax_brier, brier)

    save_all(fig, OUT_STEM)
    DESKTOP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_all(fig, DESKTOP_OUT_STEM)
    plt.close(fig)

    print(f"png={OUT_STEM.with_suffix('.png')}")
    print(f"pdf={OUT_STEM.with_suffix('.pdf')}")
    print(f"desktop={DESKTOP_OUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
