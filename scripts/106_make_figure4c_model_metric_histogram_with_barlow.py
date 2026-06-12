from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures" / "revised_initial"
OUT_BASE = OUT_DIR / "figure4c_a_model_metric_histogram"
SOURCE_OUT = OUT_BASE.with_name(f"{OUT_BASE.name}_source_data.csv")

FINAL_GPU_ROOT = (
    ROOT
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

MODEL_ORDER = [
    "Logistic L1",
    "No-SSL CNN",
    "Barlow CNN",
    "Residual-aware CNN",
    "Residual-aware SSL-CNN",
]

COLORS = {
    "Logistic L1": "#4E79A7",
    "No-SSL CNN": "#F28E2B",
    "Barlow CNN": "#D95F59",
    "Residual-aware CNN": "#7B6AB0",
    "Residual-aware SSL-CNN": "#2EAD74",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.5,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _standardize(row: pd.Series, model: str, source: str) -> dict[str, float | str]:
    return {
        "model": model,
        "source": source,
        "accuracy": float(row["accuracy"]),
        "balanced_accuracy": float(row["balanced_accuracy"]),
        "sensitivity": float(row["sensitivity"]),
        "specificity": float(row["specificity"]),
        "roc_auc": float(row["roc_auc"]),
        "pr_auc": float(row["pr_auc"]),
    }


def load_scorecard_data() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    ml = pd.read_csv(ROOT / "results" / "tables" / "eeg_only_ml_three_line_table.csv")
    row = ml.loc[ml["模型"].eq("Logistic L1 (No selector)")].iloc[0]
    rows.append(
        {
            "model": "Logistic L1",
            "source": "eeg_only_ml_three_line_table.csv",
            "accuracy": float(row["Accuracy"]),
            "balanced_accuracy": float(row["Balanced accuracy"]),
            "sensitivity": float(row["Sensitivity"]),
            "specificity": float(row["Specificity"]),
            "roc_auc": float(row["ROC-AUC"]),
            "pr_auc": float(row["PR-AUC"]),
        }
    )

    no_ssl = pd.read_csv(ROOT / "results" / "metrics" / "updated_sub05_sub28_no_ssl_exact6cfg_standard10_summary.csv")
    rows.append(_standardize(no_ssl.loc[no_ssl["row_type"].eq("mean")].iloc[0], "No-SSL CNN", no_ssl.attrs.get("source", "updated_sub05_sub28_no_ssl_exact6cfg_standard10_summary.csv")))

    barlow = pd.read_csv(ROOT / "results" / "metrics" / "barlow_cnn_10seed_summary.csv")
    rows.append(_standardize(barlow.loc[barlow["row_type"].eq("mean")].iloc[0], "Barlow CNN", "barlow_cnn_10seed_summary.csv"))

    residual = pd.read_csv(ROOT / "results" / "metrics" / "no_ssl_residualaware_highrank_swa_clsalpha1_rerun_20260611_10seed_summary.csv")
    rows.append(_standardize(residual.loc[residual["row_type"].eq("mean")].iloc[0], "Residual-aware CNN", "no_ssl_residualaware_highrank_swa_clsalpha1_rerun_20260611_10seed_summary.csv"))

    final = pd.read_csv(FINAL_GPU_ROOT)
    final_mean = final.mean(numeric_only=True)
    rows.append(
        {
            "model": "Residual-aware SSL-CNN",
            "source": str(FINAL_GPU_ROOT.relative_to(ROOT)),
            "accuracy": float(final_mean["Accuracy"]),
            "balanced_accuracy": float(final_mean["Balanced accuracy"]),
            "sensitivity": float(final_mean["Sensitivity"]),
            "specificity": float(final_mean["Specificity"]),
            "roc_auc": float(final_mean["ROC AUC"]),
            "pr_auc": float(final_mean["PR AUC"]),
        }
    )

    data = pd.DataFrame(rows)
    data["order"] = data["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
    return data.sort_values("order").drop(columns="order").reset_index(drop=True)


def plot_scorecard(data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 3.95), constrained_layout=True)
    x = np.arange(len(METRICS))
    width = 0.132
    offsets = (np.arange(len(data)) - (len(data) - 1) / 2) * width

    for offset, (_, row) in zip(offsets, data.iterrows(), strict=True):
        values = [float(row[metric]) for metric, _ in METRICS]
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=COLORS[str(row["model"])],
            edgecolor="white",
            linewidth=0.45,
            label=str(row["model"]),
        )
        if row["model"] == "Residual-aware SSL-CNN":
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.012,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=5.5,
                    color="#1F2933",
                )

    ax.set_xticks(x, [label for _, label in METRICS])
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score", fontsize=8.2)
    ax.set_title("Patient-level model scorecard", loc="left", fontsize=10.0, pad=9)
    ax.text(-0.115, 1.03, "a", transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=9.5)
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        columnspacing=1.8,
        handlelength=1.1,
    )
    return fig


def save_pub(fig: plt.Figure, base: Path, dpi: int = 600) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_scorecard_data()
    data.to_csv(SOURCE_OUT, index=False, encoding="utf-8-sig")
    fig = plot_scorecard(data)
    save_pub(fig, OUT_BASE)
    plt.close(fig)
    print(f"saved={OUT_BASE}")
    print(f"source={SOURCE_OUT}")
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
