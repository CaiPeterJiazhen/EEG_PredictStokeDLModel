from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = RUN_ROOT / "independent_train_gpu_main"
LOSS_HISTORY_PATH = (
    OUTPUT_ROOT
    / "results"
    / "loss_history"
    / "final_ssl_cnn_feature_state_band_ablation_supervised_loss_history.csv"
)
FIGURE_ROOT = OUTPUT_ROOT / "results" / "figures" / "revised_initial"
METRIC_ROOT = OUTPUT_ROOT / "results" / "metrics"
ABLATION_NAME = "full_psd_wpli"
SWA_START_EPOCH = 50


def mean_sem(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        frame.groupby("epoch", as_index=False)[column]
        .agg(["mean", "sem"])
        .reset_index()
        .rename(columns={"mean": "value_mean", "sem": "value_sem"})
    )
    summary["value_sem"] = summary["value_sem"].fillna(0.0)
    return summary


def build_summary(loss_history: pd.DataFrame) -> pd.DataFrame:
    data = loss_history[loss_history["ablation_name"] == ABLATION_NAME].copy()
    if data.empty:
        raise ValueError(f"No rows found for ablation_name={ABLATION_NAME!r}.")

    for prefix in ("train", "val"):
        data[f"{prefix}_weighted_bce"] = data[f"{prefix}_loss_bce"]
        data[f"{prefix}_weighted_residual"] = (
            data["lambda_reg"] * data[f"{prefix}_loss_residual_regression"]
        )
        data[f"{prefix}_weighted_rank"] = (
            data["lambda_rank"] * data[f"{prefix}_loss_pairwise_ranking"]
        )
        data[f"{prefix}_weighted_soft"] = (
            data["lambda_soft"] * data[f"{prefix}_loss_soft_label"]
        )

    metric_columns = [
        "train_loss_total",
        "val_loss_total",
        "train_weighted_bce",
        "val_weighted_bce",
        "train_weighted_residual",
        "val_weighted_residual",
        "train_weighted_rank",
        "val_weighted_rank",
        "train_weighted_soft",
        "val_weighted_soft",
    ]
    parts = []
    for metric in metric_columns:
        part = mean_sem(data, metric)
        part["metric"] = metric
        parts.append(part)
    summary = pd.concat(parts, ignore_index=True)
    summary["ablation_name"] = ABLATION_NAME
    summary["n_seed_fold_runs"] = int(data.groupby(["seed", "fold_id"]).ngroups)
    summary["n_seeds"] = int(data["seed"].nunique())
    summary["n_folds"] = int(data["fold_id"].nunique())
    return summary


def plot_loss_curves(summary: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    colors = {
        "train_total": "#1f77b4",
        "val_total": "#ff7f0e",
        "bce": "#2b5c8a",
        "residual": "#d95f02",
        "ranking": "#7570b3",
        "soft": "#1b9e77",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)

    ax = axes[0]
    for metric, label, color in [
        ("train_loss_total", "Training total", colors["train_total"]),
        ("val_loss_total", "Validation total", colors["val_total"]),
    ]:
        series = summary[summary["metric"] == metric]
        x = series["epoch"].to_numpy(dtype=float)
        y = series["value_mean"].to_numpy(dtype=float)
        e = series["value_sem"].to_numpy(dtype=float)
        ax.plot(x, y, lw=1.6, color=color, label=label)
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.18, lw=0)
    ax.axvline(SWA_START_EPOCH, color="#777777", lw=0.8, ls="--")
    ax.text(
        SWA_START_EPOCH + 1.5,
        ax.get_ylim()[1] * 0.92,
        "SWA",
        color="#555555",
        fontsize=6.5,
    )
    ax.set_title("Total loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", fontsize=6.4)

    ax = axes[1]
    for metric, label, color_key in [
        ("train_weighted_bce", "BCE", "bce"),
        ("train_weighted_residual", "Residual", "residual"),
        ("train_weighted_rank", "Ranking", "ranking"),
        ("train_weighted_soft", "Soft label", "soft"),
    ]:
        series = summary[summary["metric"] == metric]
        ax.plot(series["epoch"], series["value_mean"], lw=1.25, color=colors[color_key], label=label)
    ax.axvline(SWA_START_EPOCH, color="#777777", lw=0.8, ls="--")
    ax.set_title("Weighted training components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted contribution")
    ax.legend(loc="upper right", fontsize=6.2)

    ax = axes[2]
    for metric, label, color_key in [
        ("val_weighted_bce", "BCE", "bce"),
        ("val_weighted_residual", "Residual", "residual"),
        ("val_weighted_rank", "Ranking", "ranking"),
        ("val_weighted_soft", "Soft label", "soft"),
    ]:
        series = summary[summary["metric"] == metric]
        ax.plot(series["epoch"], series["value_mean"], lw=1.25, color=colors[color_key], label=label)
    ax.axvline(SWA_START_EPOCH, color="#777777", lw=0.8, ls="--")
    ax.set_title("Weighted validation components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted contribution")
    ax.legend(loc="upper right", fontsize=6.2)

    for idx, ax in enumerate(axes):
        ax.text(
            -0.12,
            1.08,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
        )
        ax.grid(axis="y", color="#e5e5e5", lw=0.5)

    return fig


def main() -> None:
    loss_history = pd.read_csv(LOSS_HISTORY_PATH)
    summary = build_summary(loss_history)

    METRIC_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = METRIC_ROOT / "full_psd_wpli_loss_curve_summary.csv"
    summary.to_csv(summary_path, index=False)

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig = plot_loss_curves(summary)
    base = FIGURE_ROOT / "figure_full_psd_wpli_loss_curves"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote figure base: {base}")


if __name__ == "__main__":
    main()
