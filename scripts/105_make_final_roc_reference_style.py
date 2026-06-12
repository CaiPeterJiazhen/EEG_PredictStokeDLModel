from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "predictions"
    / "final_Residual_ssl_cnn_10seed_patient_predictions.csv"
)
OUT_BASE = PROJECT_ROOT / "results" / "figures" / "revised_initial" / "figure4a_final_model_roc"
SOURCE_OUT = OUT_BASE.with_name(f"{OUT_BASE.name}_source_data.csv")


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.5,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#1F2933",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save_pub(fig: plt.Figure, base: Path, *, dpi: int = 600) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def main() -> None:
    pred = pd.read_csv(PREDICTIONS)
    pred = (
        pred.groupby("subject_id", as_index=False)
        .agg(y_true=("y_true", "first"), y_score=("y_score", "mean"))
        .sort_values("subject_id")
    )

    y_true = pred["y_true"].astype(int).to_numpy()
    y_score = pred["y_score"].astype(float).to_numpy()
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    roc_auc = auc(fpr, tpr)

    pd.DataFrame(
        {
            "model_name": "final_residual_barlow_cnn_seedmean10",
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
            "roc_auc": roc_auc,
        }
    ).to_csv(SOURCE_OUT, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(4.35, 4.0), constrained_layout=True)
    purple = "#8E44AD"
    diagonal = "#EF6F8F"

    ax.plot([0, 1], [0, 1], color=diagonal, linewidth=0.8, linestyle=(0, (3.2, 2.3)), zorder=1)
    ax.plot(
        fpr,
        tpr,
        color=purple,
        linewidth=1.15,
        drawstyle="steps-post",
        label=f"AUC = {roc_auc:.2f}",
        zorder=3,
    )

    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=8.0)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=8.0)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=9.1, fontweight="bold", pad=7)
    ax.grid(True, color="#DADDE3", linestyle=":", linewidth=0.55, alpha=0.95)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#1F2933")

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor=purple,
        fontsize=7.4,
        handlelength=1.8,
        borderpad=0.6,
        labelspacing=0.3,
    )
    legend.get_frame().set_linewidth(0.7)

    save_pub(fig, OUT_BASE)
    plt.close(fig)
    print(f"saved={OUT_BASE}")
    print(f"source={SOURCE_OUT}")
    print(f"auc={roc_auc:.6f}")


if __name__ == "__main__":
    main()
