from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = (
    PROJECT_ROOT
    / "final_model_ablation_explainability_results_20260610"
    / "results"
    / "tables"
    / "table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv"
)
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "paper_panels"
OUT_STEM = OUT_DIR / "figure5b_state_ablation"
SOURCE_OUT = OUT_DIR / "figure5b_state_ablation_source_data.csv"
MIRROR_OUT_DIR = (
    PROJECT_ROOT
    / "final_model_ablation_explainability_results_20260610"
    / "results"
    / "figures"
    / "paper_panels"
)
MIRROR_OUT_STEM = MIRROR_OUT_DIR / "figure5b_state_ablation"

ABLATION_ORDER = ["eo_only", "ec_only"]
ABLATION_LABELS = {
    "eo_only": "EO only",
    "ec_only": "EC only",
}
METRICS = [
    ("balanced_accuracy_mean", "balanced_accuracy_sd", "Balanced accuracy", "#4C78A8"),
    ("roc_auc_mean", "roc_auc_sd", "ROC-AUC", "#2AA876"),
    ("brier_mean", "brier_sd", "Brier score", "#8A8F98"),
]
FINAL_REFERENCES = {
    "Balanced accuracy": (0.8411, "#4C78A8"),
    "ROC-AUC": (0.8867, "#2AA876"),
    "Brier score": (0.1324, "#B8BDC5"),
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
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


def load_source() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_CSV)
    selected = frame.loc[frame["ablation_name"].isin(ABLATION_ORDER)].copy()
    if selected.shape[0] != len(ABLATION_ORDER):
        found = selected["ablation_name"].tolist()
        raise ValueError(f"Expected {ABLATION_ORDER}, found {found}")
    selected["ablation_name"] = pd.Categorical(selected["ablation_name"], categories=ABLATION_ORDER, ordered=True)
    selected = selected.sort_values("ablation_name").reset_index(drop=True)
    selected["display"] = selected["ablation_name"].map(ABLATION_LABELS)
    return selected


def make_long_source(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in selected.iterrows():
        for mean_col, sd_col, label, _ in METRICS:
            rows.append(
                {
                    "ablation_name": record["ablation_name"],
                    "display": record["display"],
                    "metric": label,
                    "mean": float(record[mean_col]),
                    "sd": float(record[sd_col]),
                    "final_reference": FINAL_REFERENCES[label][0],
                }
            )
    return pd.DataFrame(rows)


def make_plot(selected: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MIRROR_OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_source = make_long_source(selected)
    long_source.to_csv(SOURCE_OUT, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(4.25, 3.15), constrained_layout=True)

    x = np.arange(len(ABLATION_ORDER), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, len(METRICS))

    for offset, (mean_col, sd_col, label, color) in zip(offsets, METRICS, strict=True):
        means = selected[mean_col].astype(float).to_numpy()
        sds = selected[sd_col].astype(float).to_numpy()
        ax.bar(
            x + offset,
            means,
            width=width * 0.88,
            yerr=sds,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            capsize=2.5,
            error_kw={
                "elinewidth": 0.75,
                "capthick": 0.75,
                "ecolor": "#263238",
            },
            zorder=3,
            label=label,
        )

    for label, (value, color) in FINAL_REFERENCES.items():
        ax.axhline(value, color=color, linestyle=(0, (3.2, 2.3)), linewidth=0.95, alpha=0.85, zorder=1)
    label_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        1.01,
        FINAL_REFERENCES["ROC-AUC"][0] + 0.006,
        "Final ROC-AUC",
        transform=label_transform,
        color=FINAL_REFERENCES["ROC-AUC"][1],
        ha="left",
        va="bottom",
        fontsize=6.2,
        clip_on=False,
    )
    ax.text(
        1.01,
        FINAL_REFERENCES["Balanced accuracy"][0] - 0.006,
        "Final balanced acc.",
        transform=label_transform,
        color=FINAL_REFERENCES["Balanced accuracy"][1],
        ha="left",
        va="top",
        fontsize=6.2,
        clip_on=False,
    )
    ax.text(
        1.01,
        FINAL_REFERENCES["Brier score"][0] + 0.006,
        "Final Brier",
        transform=label_transform,
        color="#7A7F87",
        ha="left",
        va="bottom",
        fontsize=6.2,
        clip_on=False,
    )

    ax.set_title("State ablation", loc="left", pad=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ABLATION_LABELS[name] for name in ABLATION_ORDER])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_xlim(-0.5, 1.5)
    ax.grid(axis="y", color="#D7DEE8", linewidth=0.55, zorder=0)

    handles = [Patch(facecolor=color, edgecolor="white", label=label) for _, _, label, color in METRICS]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#7A7F87",
            linestyle=(0, (3.2, 2.3)),
            linewidth=0.95,
            label="Final model reference",
        )
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        ncol=2,
        handlelength=1.45,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for stem in [OUT_STEM, MIRROR_OUT_STEM]:
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    selected = load_source()
    make_plot(selected)
    print(f"png={OUT_STEM.with_suffix('.png')}")
    print(f"pdf={OUT_STEM.with_suffix('.pdf')}")
    print(f"mirror_png={MIRROR_OUT_STEM.with_suffix('.png')}")
    print(f"mirror_pdf={MIRROR_OUT_STEM.with_suffix('.pdf')}")
    print(f"source={SOURCE_OUT}")


if __name__ == "__main__":
    main()
