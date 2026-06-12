from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve


RUN_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = RUN_ROOT / "independent_train_gpu_main"
PREDICTIONS = (
    OUTPUT_ROOT
    / "results"
    / "predictions"
    / "final_ssl_cnn_feature_state_band_ablation_predictions.csv"
)
FIGURE_ROOT = OUTPUT_ROOT / "results" / "figures" / "revised_initial"
METRIC_ROOT = OUTPUT_ROOT / "results" / "metrics"
ABLATION_NAME = "full_psd_wpli"


def save_pub(fig: plt.Figure, base: Path, *, dpi: int = 600) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def load_full_predictions() -> pd.DataFrame:
    pred = pd.read_csv(PREDICTIONS)
    pred = pred[pred["ablation_name"].eq(ABLATION_NAME)].copy()
    if pred.empty:
        raise ValueError(f"No predictions found for ablation_name={ABLATION_NAME!r}.")
    return pred


def seedmean_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    averaged = (
        pred.groupby("test_subject_id", as_index=False)
        .agg(y_true=("y_true", "first"), y_score=("y_prob", "mean"))
        .sort_values("test_subject_id")
    )
    return averaged


def build_curve_source(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    seed_auc_rows = []

    for seed, frame in pred.groupby("seed", sort=True):
        y_true = frame["y_true"].astype(int).to_numpy()
        y_score = frame["y_prob"].astype(float).to_numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
        roc_auc = auc(fpr, tpr)
        seed_auc_rows.append({"curve": f"seed_{seed}", "seed": int(seed), "roc_auc": float(roc_auc)})
        rows.append(
            pd.DataFrame(
                {
                    "curve": f"seed_{seed}",
                    "seed": int(seed),
                    "fpr": fpr,
                    "tpr": tpr,
                    "threshold": thresholds,
                    "roc_auc": float(roc_auc),
                }
            )
        )

    averaged = seedmean_predictions(pred)
    y_true = averaged["y_true"].astype(int).to_numpy()
    y_score = averaged["y_score"].astype(float).to_numpy()
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    seedmean_auc = auc(fpr, tpr)
    seed_auc_rows.append({"curve": "seedmean10", "seed": np.nan, "roc_auc": float(seedmean_auc)})
    rows.append(
        pd.DataFrame(
            {
                "curve": "seedmean10",
                "seed": np.nan,
                "fpr": fpr,
                "tpr": tpr,
                "threshold": thresholds,
                "roc_auc": float(seedmean_auc),
            }
        )
    )

    return pd.concat(rows, ignore_index=True), pd.DataFrame(seed_auc_rows)


def plot_roc(curves: pd.DataFrame, auc_summary: pd.DataFrame, pred: pd.DataFrame) -> plt.Figure:
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
            "legend.frameon": False,
        }
    )

    seed_rows = auc_summary[auc_summary["curve"].str.startswith("seed_")]
    seedmean_auc = float(auc_summary.loc[auc_summary["curve"].eq("seedmean10"), "roc_auc"].iloc[0])
    seed_auc_mean = float(seed_rows["roc_auc"].mean())
    seed_auc_sd = float(seed_rows["roc_auc"].std(ddof=1))
    n_subjects = int(pred["test_subject_id"].nunique())
    n_seeds = int(pred["seed"].nunique())

    fig, ax = plt.subplots(figsize=(4.35, 4.0), constrained_layout=True)
    diagonal = "#C9CED6"
    seed_color = "#8EA9C4"
    main_color = "#245C91"
    accent = "#D95F02"

    ax.plot([0, 1], [0, 1], color=diagonal, linewidth=0.95, linestyle=(0, (3.2, 2.3)), zorder=1)
    for curve_name, frame in curves[curves["curve"].str.startswith("seed_")].groupby("curve"):
        ax.plot(
            frame["fpr"],
            frame["tpr"],
            color=seed_color,
            linewidth=0.75,
            alpha=0.34,
            drawstyle="steps-post",
            zorder=2,
        )

    seedmean = curves[curves["curve"].eq("seedmean10")]
    ax.plot(
        seedmean["fpr"],
        seedmean["tpr"],
        color=main_color,
        linewidth=1.8,
        drawstyle="steps-post",
        label=f"Seed-mean ROC-AUC = {seedmean_auc:.3f}",
        zorder=4,
    )

    ax.text(
        0.05,
        0.12,
        f"Per-seed ROC-AUC\n{seed_auc_mean:.3f} +/- {seed_auc_sd:.3f}\nn = {n_subjects}, seeds = {n_seeds}",
        transform=ax.transAxes,
        fontsize=7.1,
        color="#26323F",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#DADDE3", "linewidth": 0.55},
    )

    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("False positive rate (1 - specificity)", fontsize=8.0)
    ax.set_ylabel("True positive rate (sensitivity)", fontsize=8.0)
    ax.set_title("full_psd_wpli ROC curve", fontsize=9.1, fontweight="bold", pad=7)
    ax.grid(True, color="#DADDE3", linestyle=":", linewidth=0.55, alpha=0.95)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#1F2933")

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor=accent,
        fontsize=7.3,
        handlelength=1.9,
        borderpad=0.6,
        labelspacing=0.3,
    )
    legend.get_frame().set_linewidth(0.7)
    return fig


def main() -> None:
    pred = load_full_predictions()
    curves, auc_summary = build_curve_source(pred)

    METRIC_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    curve_path = METRIC_ROOT / "full_psd_wpli_roc_curve_source_data.csv"
    auc_path = METRIC_ROOT / "full_psd_wpli_roc_auc_summary.csv"
    curves.to_csv(curve_path, index=False)
    auc_summary.to_csv(auc_path, index=False)

    fig = plot_roc(curves, auc_summary, pred)
    base = FIGURE_ROOT / "figure_full_psd_wpli_roc_curve"
    save_pub(fig, base)
    plt.close(fig)

    seed_rows = auc_summary[auc_summary["curve"].str.startswith("seed_")]
    seedmean_auc = float(auc_summary.loc[auc_summary["curve"].eq("seedmean10"), "roc_auc"].iloc[0])
    print(f"Wrote figure base: {base}")
    print(f"Wrote curve source: {curve_path}")
    print(f"Wrote AUC summary: {auc_path}")
    print(f"seedmean_roc_auc={seedmean_auc:.6f}")
    print(f"per_seed_roc_auc_mean={seed_rows['roc_auc'].mean():.6f}")
    print(f"per_seed_roc_auc_sd={seed_rows['roc_auc'].std(ddof=1):.6f}")


if __name__ == "__main__":
    main()
