from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "requested_triptychs"
FINAL_RESIDUAL_SSL_CNN_PATH = (
    PROJECT_ROOT
    / "final_model_ablation_explainability_results_20260610"
    / "independent_train_gpu_main"
    / "results"
    / "tables"
    / "final_Residual_ssl_cnn.csv"
)

METRICS = [
    ("accuracy", "Mean\naccuracy"),
    ("balanced_accuracy", "Balanced\naccuracy"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]

MODELS = [
    ("a_ml_psd_wpli_baseline", "reported", "Logistic L1", "#4E79A7"),
    ("b_no_ssl_cnn_same_arch_10seed", "mean", "No-SSL CNN", "#F28E2B"),
    ("c_patient_barlow_ssl_no_residual_heads", "mean", "Barlow CNN", "#E15759"),
    ("d_no_ssl_cnn_residual_heads", "mean", "Residual-aware CNN", "#7B6AB0"),
    ("e_patient_barlow_ssl_residual_heads", "mean", "Residual-aware SSL-CNN", "#2EAD74"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    core = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "core_ablation_10seed_summary.csv")
    final_residual_ssl_cnn = pd.read_csv(FINAL_RESIDUAL_SSL_CNN_PATH)
    data = collect_rows(core, final_residual_ssl_cnn)
    fig, ax = plt.subplots(figsize=(8.2, 3.75), constrained_layout=True)
    plot_scorecard(ax, data)
    base = OUT_DIR / "patient_level_model_scorecard_with_barlow"
    for suffix in (".png", ".svg", ".pdf", ".tiff"):
        path = base.with_suffix(suffix)
        if suffix in {".png", ".tiff"}:
            fig.savefig(path, dpi=400, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(base.with_suffix(".csv"), index=False)
    print(f"Wrote {base.with_suffix('.png')}")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.4,
            "axes.titlesize": 9,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
        }
    )


def collect_rows(core: pd.DataFrame, final_residual_ssl_cnn: pd.DataFrame) -> pd.DataFrame:
    final_metric_map = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC",
    }
    rows = []
    for contrast, row_type, label, color in MODELS:
        subset = core[(core["contrast"] == contrast) & (core["row_type"] == row_type)]
        if subset.empty:
            raise KeyError(f"Missing {contrast} {row_type} in core_ablation_10seed_summary.csv")
        row = subset.iloc[0]
        record = {"model": label, "color": color, "metric_source": "core_ablation_10seed_summary.csv"}
        for metric, _ in METRICS:
            record[metric] = float(row[metric])
        if label == "Residual-aware SSL-CNN":
            record["metric_source"] = str(FINAL_RESIDUAL_SSL_CNN_PATH.relative_to(PROJECT_ROOT))
            for metric, source_column in final_metric_map.items():
                record[metric] = float(final_residual_ssl_cnn[source_column].mean())
        rows.append(record)
    return pd.DataFrame(rows)


def plot_scorecard(ax: plt.Axes, data: pd.DataFrame) -> None:
    x = np.arange(len(METRICS))
    width = 0.145
    offsets = (np.arange(len(data)) - (len(data) - 1) / 2) * width
    for offset, (_, row) in zip(offsets, data.iterrows(), strict=True):
        values = [row[metric] for metric, _ in METRICS]
        bars = ax.bar(x + offset, values, width=width * 0.92, color=row["color"], label=row["model"])
        if row["model"] == "Residual-aware SSL-CNN":
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.014,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=5.6,
                    color=row["color"],
                )
    ax.set_xticks(x, [label for _, label in METRICS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Patient-level model scorecard")
    ax.text(-0.12, 1.05, "a", transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=9)
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3)


if __name__ == "__main__":
    main()
