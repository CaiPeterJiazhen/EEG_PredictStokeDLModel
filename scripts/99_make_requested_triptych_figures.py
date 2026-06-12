from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import auc, brier_score_loss, confusion_matrix, roc_curve


OUT_DIR = PROJECT_ROOT / "results" / "figures" / "requested_triptychs"
FINAL_MODEL = "residual_aware_SSL_CNN_seedmean10"
LOGISTIC_L1 = "ML_EEG_updated_no_selector_logistic_l1"

COLORS = {
    "final": "#B84A39",
    "baseline": "#64748B",
    "cnn": "#3B78A3",
    "ssl": "#9A6A32",
    "residual": "#5B8C62",
    "metric1": "#4D7895",
    "metric2": "#B84A39",
    "metric3": "#D7A441",
    "metric4": "#7A6C9E",
    "psd": "#4E79A7",
    "wpli": "#B65A50",
    "modality": "#4E79A7",
    "state": "#5B8C62",
    "band": "#D7A441",
    "network": "#7A6C9E",
    "poor": "#6B7280",
    "prop": "#B84A39",
    "grid": "#D8DEE6",
    "ink": "#20252A",
}

DISPLAY = {
    FINAL_MODEL: "Residual-aware SSL-CNN",
    LOGISTIC_L1: "Logistic L1",
    "ML PSD+WPLI Logistic L1": "Logistic L1",
    "No-SSL CNN same architecture": "No-SSL CNN",
    "Patient-level Barlow CNN without residual-aware heads": "Barlow SSL-CNN\n(no residual heads)",
    "no-SSL CNN + residual-aware auxiliary heads": "Residual-aware CNN\n(no SSL)",
    "Patient-level Barlow SSL + residual-aware auxiliary heads": "Residual-aware\nSSL-CNN",
    "psd_only": "PSD only",
    "wpli_only": "WPLI only",
    "full_psd_wpli": "PSD + WPLI",
    "psd_wpli": "PSD + WPLI",
    "eo_only": "EO only",
    "ec_only": "EC only",
    "psd_eo_only": "PSD EO only",
    "wpli_ec_only": "WPLI EC only",
    "psd_eo_wpli_ec": "PSD EO +\nWPLI EC",
    "beta_medium_only": "Beta medium",
    "beta_high_only": "Beta high",
    "beta_medium_beta_high": "Beta medium +\nbeta high",
    "motor_wpli_edges_only": "Motor WPLI\nedges",
    "motor_wpli_edges": "Motor WPLI\nedges",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    manifest: list[dict[str, object]] = []
    manifest.append(make_final_performance_calibration())
    manifest.append(make_model_components_loss())
    manifest.append(make_feature_state_band_ablation())
    manifest.append(make_explainability_maps())
    manifest.append(make_top_feature_group_distributions())

    pd.DataFrame(manifest).to_csv(OUT_DIR / "requested_triptych_manifest.csv", index=False)
    write_captions(manifest)
    print(f"Wrote requested triptych figures to {OUT_DIR}")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.dpi": 160,
        }
    )


def save_pub(fig: plt.Figure, base: Path, *, dpi: int = 400) -> dict[str, str]:
    paths = {}
    for suffix in (".png", ".svg", ".pdf", ".tiff"):
        path = base.with_suffix(suffix)
        if suffix in {".png", ".tiff"}:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        paths[suffix.lstrip(".")] = str(path)
    return paths


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
    )


def style_axis(ax: plt.Axes, *, ygrid: bool = True) -> None:
    if ygrid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)


def make_final_performance_calibration() -> dict[str, object]:
    predictions = pd.read_csv(PROJECT_ROOT / "results" / "predictions" / "paper_locked_model_predictions.csv")
    perf = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "paper_locked_model_performance.csv")

    final = predictions[predictions["model_name"] == FINAL_MODEL].copy()
    logistic = predictions[predictions["model_name"] == LOGISTIC_L1].copy()
    if final.empty or logistic.empty:
        raise FileNotFoundError("Final model or Logistic L1 predictions are missing from paper_locked_model_predictions.csv")

    fig = plt.figure(figsize=(10.8, 3.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.15])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_label(ax_a, "A")
    plot_roc_panel(ax_a, final, logistic)

    panel_label(ax_b, "B")
    plot_confusion_panel(ax_b, final)

    panel_label(ax_c, "C")
    plot_reliability_panel(ax_c, final, logistic, perf)

    base = OUT_DIR / "figure1_final_performance_calibration_triptych"
    paths = save_pub(fig, base)
    plt.close(fig)
    return manifest_row(base, paths, "Final ROC, confusion matrix, and calibration")


def plot_roc_panel(ax: plt.Axes, final: pd.DataFrame, logistic: pd.DataFrame) -> None:
    for frame, color, label, lw in [
        (final, COLORS["final"], "Residual-aware SSL-CNN", 2.0),
        (logistic, COLORS["baseline"], "Logistic L1 baseline", 1.8),
    ]:
        y_true = frame["y_true"].astype(int).to_numpy()
        y_score = frame["y_score"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=lw, label=f"{label} (AUC {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="#111827", linestyle=(0, (3, 3)), linewidth=0.9, label="Random classifier")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Final model ROC curve")
    ax.legend(loc="lower right", frameon=False)
    style_axis(ax)


def plot_confusion_panel(ax: plt.Axes, final: pd.DataFrame) -> None:
    y_true = final["y_true"].astype(int).to_numpy()
    y_pred = final["y_pred"].astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(int(cm.max()), 1))
    ax.set_xticks([0, 1], ["Poor", "Proportional"])
    ax.set_yticks([0, 1], ["Poor", "Proportional"])
    ax.set_xlabel("Predicted recovery group")
    ax.set_ylabel("Observed recovery group")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.55 else COLORS["ink"]
            ax.text(j, i - 0.08, f"{cm[i, j]:d}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
            ax.text(j, i + 0.17, f"{row_pct[i, j]:.1f}%", ha="center", va="center", color=color, fontsize=7.2)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Subjects")


def plot_reliability_panel(ax: plt.Axes, final: pd.DataFrame, logistic: pd.DataFrame, perf: pd.DataFrame) -> None:
    ax.plot([0, 1], [0, 1], color="#111827", linestyle=(0, (3, 3)), linewidth=0.8)
    for frame, color, label in [
        (final, COLORS["final"], "Residual-aware SSL-CNN"),
        (logistic, COLORS["baseline"], "Logistic L1"),
    ]:
        centers, observed, counts = reliability_curve(frame["y_true"], frame["y_score"], n_bins=5)
        ax.plot(centers, observed, marker="o", color=color, linewidth=1.5, label=label)
        for x, y, n in zip(centers, observed, counts, strict=False):
            if n > 0:
                ax.text(x, y + 0.035, str(int(n)), ha="center", va="bottom", fontsize=5.8, color=color)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed proportional recovery")
    ax.set_title("Reliability and Brier score")
    ax.legend(loc="upper left", frameon=False)
    style_axis(ax)

    inset = inset_axes(ax, width="38%", height="35%", loc="lower right", borderpad=1.05)
    briers = []
    labels = []
    colors = []
    for name, color, label in [(LOGISTIC_L1, COLORS["baseline"], "L1"), (FINAL_MODEL, COLORS["final"], "Final")]:
        row = perf[perf["model_name"] == name]
        if row.empty:
            frame = final if name == FINAL_MODEL else logistic
            value = brier_score_loss(frame["y_true"], frame["y_score"])
        else:
            value = float(row["brier_score"].iloc[0])
        briers.append(value)
        labels.append(label)
        colors.append(color)
    inset.bar(range(len(briers)), briers, color=colors, width=0.62)
    inset.set_xticks(range(len(labels)), labels)
    inset.set_ylabel("Brier", fontsize=6)
    inset.tick_params(labelsize=5.8)
    inset.set_ylim(0, max(briers) * 1.25)
    inset.grid(axis="y", color=COLORS["grid"], linewidth=0.45)
    for i, value in enumerate(briers):
        inset.text(i, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8)


def reliability_curve(y_true: Iterable[float], y_score: Iterable[float], *, n_bins: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    observed = []
    counts = []
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        if high == 1.0:
            mask = (y_score_arr >= low) & (y_score_arr <= high)
        else:
            mask = (y_score_arr >= low) & (y_score_arr < high)
        if not mask.any():
            continue
        centers.append(float(y_score_arr[mask].mean()))
        observed.append(float(y_true_arr[mask].mean()))
        counts.append(int(mask.sum()))
    return np.asarray(centers), np.asarray(observed), np.asarray(counts)


def make_model_components_loss() -> dict[str, object]:
    core = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "core_ablation_10seed_summary.csv")
    loss = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "final_model_loss_curve_summary.csv")

    fig = plt.figure(figsize=(11.4, 3.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.16, 1.05, 1.18])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_label(ax_a, "A")
    plot_seed_average_metrics(ax_a, core)

    panel_label(ax_b, "B")
    plot_core_module_contribution(ax_b, core)

    panel_label(ax_c, "C")
    plot_loss_curves(ax_c, loss)

    base = OUT_DIR / "figure2_model_components_loss_triptych"
    paths = save_pub(fig, base)
    plt.close(fig)
    return manifest_row(base, paths, "Seed-level metrics, core module contribution, and loss curves")


def core_row(core: pd.DataFrame, contrast: str, row_type: str = "mean") -> pd.Series:
    subset = core[(core["contrast"] == contrast) & (core["row_type"] == row_type)]
    if subset.empty and row_type == "mean":
        subset = core[(core["contrast"] == contrast) & (core["row_type"] == "reported")]
    if subset.empty:
        raise KeyError(f"Missing {contrast} {row_type} in core ablation table")
    return subset.iloc[0]


def plot_seed_average_metrics(ax: plt.Axes, core: pd.DataFrame) -> None:
    rows = [
        core_row(core, "a_ml_psd_wpli_baseline", "reported"),
        core_row(core, "b_no_ssl_cnn_same_arch_10seed", "mean"),
        core_row(core, "d_no_ssl_cnn_residual_heads", "mean"),
        core_row(core, "e_patient_barlow_ssl_residual_heads", "mean"),
    ]
    std_rows = {
        "b_no_ssl_cnn_same_arch_10seed": core_row(core, "b_no_ssl_cnn_same_arch_10seed", "std")
        if not core[(core["contrast"] == "b_no_ssl_cnn_same_arch_10seed") & (core["row_type"] == "std")].empty
        else None,
        "d_no_ssl_cnn_residual_heads": core_row(core, "d_no_ssl_cnn_residual_heads", "std")
        if not core[(core["contrast"] == "d_no_ssl_cnn_residual_heads") & (core["row_type"] == "std")].empty
        else None,
        "e_patient_barlow_ssl_residual_heads": core_row(core, "e_patient_barlow_ssl_residual_heads", "std")
        if not core[(core["contrast"] == "e_patient_barlow_ssl_residual_heads") & (core["row_type"] == "std")].empty
        else None,
    }
    metrics = [("balanced_accuracy", "Balanced\naccuracy"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]
    labels = ["Logistic L1\n(reference)", "No-SSL\nCNN", "Residual-aware\nCNN", "Residual-aware\nSSL-CNN"]
    x = np.arange(len(labels))
    width = 0.23
    for offset, (metric, metric_label) in enumerate(metrics):
        values = [float(row[metric]) for row in rows]
        yerr = []
        for row in rows:
            std = std_rows.get(str(row["contrast"]))
            yerr.append(float(std[metric]) if std is not None and not pd.isna(std[metric]) else 0.0)
        ax.bar(
            x + (offset - 1) * width,
            values,
            width=width,
            color=[COLORS["metric1"], COLORS["metric2"], COLORS["metric3"]][offset],
            label=metric_label,
            yerr=yerr,
            error_kw={"elinewidth": 0.7, "capsize": 2, "capthick": 0.7},
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("10-seed mean metrics")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.03, 0.04, "Logistic L1 is a single LOSO reference.", transform=ax.transAxes, fontsize=6.2)
    style_axis(ax)


def plot_core_module_contribution(ax: plt.Axes, core: pd.DataFrame) -> None:
    contrasts = [
        "b_no_ssl_cnn_same_arch_10seed",
        "c_patient_barlow_ssl_no_residual_heads",
        "d_no_ssl_cnn_residual_heads",
        "e_patient_barlow_ssl_residual_heads",
    ]
    rows = [core_row(core, contrast, "mean") for contrast in contrasts]
    labels = [DISPLAY[str(row["model_key"])] for row in rows]
    y = np.arange(len(rows))
    bal = np.array([float(row["balanced_accuracy"]) for row in rows])
    roc = np.array([float(row["roc_auc"]) for row in rows])
    ax.barh(y - 0.17, bal, height=0.28, color=COLORS["metric1"], label="Balanced accuracy")
    ax.barh(y + 0.17, roc, height=0.28, color=COLORS["metric3"], label="ROC-AUC")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 0.96)
    ax.set_xlabel("Score")
    ax.set_title("Core module contribution")
    ax.legend(frameon=False, loc="lower right")
    for yi, value in zip(y, bal, strict=True):
        ax.text(value + 0.008, yi - 0.17, f"{value:.2f}", va="center", fontsize=6)
    for yi, value in zip(y, roc, strict=True):
        ax.text(value + 0.008, yi + 0.17, f"{value:.2f}", va="center", fontsize=6)
    style_axis(ax, ygrid=False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.75)


def plot_loss_curves(ax: plt.Axes, loss: pd.DataFrame) -> None:
    pivot = loss.pivot_table(index="epoch", columns="metric", values=["value_mean", "value_sem"])
    epochs = pivot.index.to_numpy()

    def get(metric: str) -> tuple[np.ndarray, np.ndarray]:
        if ("value_mean", metric) not in pivot:
            return np.full_like(epochs, np.nan, dtype=float), np.full_like(epochs, np.nan, dtype=float)
        mean = pivot[("value_mean", metric)].to_numpy(float)
        sem = pivot[("value_sem", metric)].fillna(0.0).to_numpy(float)
        return mean, sem

    train_total, train_total_sem = get("train_loss_total")
    val_total, val_total_sem = get("val_loss_total")
    train_bce, train_bce_sem = get("train_weighted_bce")
    val_bce, val_bce_sem = get("val_weighted_bce")

    train_aux = np.zeros_like(train_total)
    train_aux_sem_sq = np.zeros_like(train_total)
    val_aux = np.zeros_like(train_total)
    val_aux_sem_sq = np.zeros_like(train_total)
    for metric in ("residual", "rank", "soft"):
        mean, sem = get(f"train_weighted_{metric}")
        train_aux += np.nan_to_num(mean)
        train_aux_sem_sq += np.nan_to_num(sem) ** 2
        mean, sem = get(f"val_weighted_{metric}")
        val_aux += np.nan_to_num(mean)
        val_aux_sem_sq += np.nan_to_num(sem) ** 2

    curves = [
        (train_total, train_total_sem, COLORS["final"], "-", "Train total"),
        (val_total, val_total_sem, COLORS["final"], "--", "Validation total"),
        (train_bce, train_bce_sem, COLORS["metric1"], "-", "Train classification"),
        (val_bce, val_bce_sem, COLORS["metric1"], "--", "Validation classification"),
        (train_aux, np.sqrt(train_aux_sem_sq), COLORS["residual"], "-", "Train residual-aware aux."),
        (val_aux, np.sqrt(val_aux_sem_sq), COLORS["residual"], "--", "Validation residual-aware aux."),
    ]
    for mean, sem, color, linestyle, label in curves:
        ax.plot(epochs, mean, color=color, linestyle=linestyle, linewidth=1.35, label=label)
        ax.fill_between(epochs, mean - sem, mean + sem, color=color, alpha=0.10, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.set_xlim(float(epochs.min()), float(epochs.max()))
    ax.legend(frameon=False, ncol=1, loc="upper right")
    style_axis(ax)


def make_feature_state_band_ablation() -> dict[str, object]:
    final_metrics = pd.read_csv(
        PROJECT_ROOT
        / "final_model_ablation_explainability_results_20260610"
        / "results"
        / "metrics"
        / "final_ssl_cnn_feature_state_band_ablation_seedmean_metrics.csv"
    )
    legacy = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "modality_state_band_ablation.csv")

    fig = plt.figure(figsize=(10.8, 3.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.15, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_label(ax_a, "A")
    plot_ablation_block(ax_a, final_metrics, legacy, ["psd_only", "wpli_only", "full_psd_wpli"], "Feature modality ablation")

    panel_label(ax_b, "B")
    plot_ablation_block(
        ax_b,
        final_metrics,
        legacy,
        ["eo_only", "ec_only", "psd_eo_only", "wpli_ec_only", "psd_eo_wpli_ec"],
        "State and state-feature ablation",
    )

    panel_label(ax_c, "C")
    plot_ablation_block(
        ax_c,
        final_metrics,
        legacy,
        ["beta_medium_only", "beta_high_only", "beta_medium_beta_high", "motor_wpli_edges_only"],
        "Band and network constraint ablation",
    )

    base = OUT_DIR / "figure3_feature_state_band_ablation_triptych"
    paths = save_pub(fig, base)
    plt.close(fig)
    return manifest_row(base, paths, "Feature modality, state, and frequency/network ablations")


def ablation_records(final_metrics: pd.DataFrame, legacy: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    records = []
    for name in names:
        final_name = name
        if name == "motor_wpli_edges":
            final_name = "motor_wpli_edges_only"
        row = final_metrics[final_metrics["ablation_name"] == final_name]
        source = "Final SSL-CNN"
        if row.empty and name == "psd_eo_wpli_ec":
            row = legacy[legacy["ablation_name"] == name]
            source = "Legacy tabular Logistic L2"
        if row.empty and name == "motor_wpli_edges_only":
            row = legacy[legacy["ablation_name"] == "motor_wpli_edges"]
            source = "Legacy tabular Logistic L2"
        if row.empty:
            continue
        row0 = row.iloc[0]
        family = infer_ablation_family(name)
        records.append(
            {
                "ablation_name": name,
                "label": DISPLAY.get(name, name),
                "balanced_accuracy": float(row0["balanced_accuracy"]),
                "roc_auc": float(row0["roc_auc"]),
                "source": source,
                "family": family,
            }
        )
    return pd.DataFrame(records)


def infer_ablation_family(name: str) -> str:
    if name in {"psd_only", "wpli_only", "full_psd_wpli", "psd_wpli"}:
        return "modality"
    if "eo" in name or "ec" in name:
        return "state"
    if "motor" in name:
        return "network"
    return "band"


def plot_ablation_block(ax: plt.Axes, final_metrics: pd.DataFrame, legacy: pd.DataFrame, names: list[str], title: str) -> None:
    records = ablation_records(final_metrics, legacy, names)
    records = records.sort_values(["balanced_accuracy", "roc_auc"], ascending=True)
    y = np.arange(len(records))
    colors = [COLORS[row["family"]] for _, row in records.iterrows()]
    hatches = ["///" if row["source"].startswith("Legacy") else "" for _, row in records.iterrows()]
    bars = ax.barh(y, records["balanced_accuracy"], color=colors, height=0.58, edgecolor="white", linewidth=0.6)
    for bar, hatch in zip(bars, hatches, strict=False):
        bar.set_hatch(hatch)
        if hatch:
            bar.set_edgecolor("#374151")
            bar.set_linewidth(0.7)
    ax.scatter(records["roc_auc"], y, color="#111827", s=18, zorder=3, label="ROC-AUC")
    ax.set_yticks(y, records["label"])
    ax.set_xlim(0.35, 0.95)
    ax.set_xlabel("Balanced accuracy (bar), ROC-AUC (dot)")
    ax.set_title(title)
    for yi, (_, row) in zip(y, records.iterrows(), strict=True):
        value = float(row["balanced_accuracy"])
        if value > 0.48:
            ax.text(value - 0.012, yi, f"{value:.2f}", va="center", ha="right", fontsize=6.3, color="white")
        else:
            ax.text(value + 0.012, yi, f"{value:.2f}", va="center", ha="left", fontsize=6.3, color=COLORS["ink"])
    style_axis(ax, ygrid=False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    legend_handles = [
        Patch(facecolor=COLORS["modality"], label="Modality"),
        Patch(facecolor=COLORS["state"], label="State"),
        Patch(facecolor=COLORS["band"], label="Band"),
        Patch(facecolor=COLORS["network"], label="Network"),
        Patch(facecolor="white", edgecolor="#374151", hatch="///", label="Legacy tabular"),
    ]
    if title.startswith("State"):
        ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=5.8)


def make_explainability_maps() -> dict[str, object]:
    psd = pd.read_csv(PROJECT_ROOT / "results" / "explainability" / "psd_channel_band_importance.csv")
    wpli = pd.read_csv(PROJECT_ROOT / "results" / "explainability" / "wpli_top_edges.csv")

    fig = plt.figure(figsize=(13.2, 4.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, width_ratios=[1.55, 1.0, 1.0, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    axes_b = [fig.add_subplot(gs[row, col]) for row in range(2) for col in range(1, 3)]
    axes_c = [fig.add_subplot(gs[row, col]) for row in range(2) for col in range(3, 5)]

    panel_label(ax_a, "A")
    plot_global_feature_importance(ax_a, psd, wpli, top_n=24)

    panel_label(axes_b[0], "B")
    show_mne_image_grid(
        axes_b,
        top_psd_state_band_paths(psd, top_n=4),
        "PSD topomap",
    )

    panel_label(axes_c[0], "C")
    show_mne_image_grid(
        axes_c,
        top_wpli_state_band_paths(wpli, top_n=4),
        "WPLI connectivity",
    )

    base = OUT_DIR / "figure4_explainability_maps_triptych"
    paths = save_pub(fig, base)
    plt.close(fig)
    return manifest_row(base, paths, "Global feature importance with MNE PSD topomaps and WPLI connectivity")


def plot_global_feature_importance(ax: plt.Axes, psd: pd.DataFrame, wpli: pd.DataFrame, top_n: int = 30) -> None:
    psd_records = psd.assign(
        family="PSD",
        feature_label=psd.apply(lambda r: f"PSD {r['state']} {r['channel']} {r['band']}", axis=1),
    )[["family", "feature_label", "mean_abs_attribution", "mean_signed_attribution"]]
    wpli_records = wpli.assign(
        family="WPLI",
        feature_label=wpli.apply(lambda r: f"WPLI {r['state']} {r['channel_i']}-{r['channel_j']} {r['band']}", axis=1),
    )[["family", "feature_label", "mean_abs_attribution", "mean_signed_attribution"]]
    merged = pd.concat([psd_records, wpli_records], ignore_index=True)
    top = merged.sort_values("mean_abs_attribution", ascending=False).head(top_n).iloc[::-1]
    colors = [COLORS["psd"] if fam == "PSD" else COLORS["wpli"] for fam in top["family"]]
    ax.barh(np.arange(len(top)), top["mean_abs_attribution"], color=colors, height=0.72)
    ax.set_yticks(np.arange(len(top)), top["feature_label"], fontsize=5.5)
    ax.set_xlabel("Mean absolute attribution")
    ax.set_title(f"Global feature importance, top {top_n}")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    ax.legend(
        handles=[Patch(facecolor=COLORS["psd"], label="PSD"), Patch(facecolor=COLORS["wpli"], label="WPLI")],
        frameon=False,
        loc="lower right",
    )


def show_existing_image(ax: plt.Axes, path: Path, title: str) -> None:
    if not path.exists():
        ax.text(0.5, 0.5, f"Missing image:\n{path.name}", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title)
        return
    img = mpimg.imread(path)
    img = trim_image_array(img)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title)


def show_mne_image_grid(axes: list[plt.Axes], items: list[tuple[Path, str]], title_prefix: str) -> None:
    for ax, item in zip(axes, items, strict=False):
        path, title = item
        if not path.exists():
            ax.text(0.5, 0.5, f"Missing\n{path.name}", ha="center", va="center", fontsize=6.5)
            ax.set_axis_off()
            ax.set_title(title, fontsize=7)
            continue
        img = mpimg.imread(path)
        img = trim_image_array(img)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=7)
    for ax in axes[len(items) :]:
        ax.set_axis_off()
    axes[0].text(
        0.5,
        1.18,
        title_prefix,
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )


def top_psd_state_band_paths(psd: pd.DataFrame, *, top_n: int) -> list[tuple[Path, str]]:
    allowed = {"Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High"}
    grouped = (
        psd[psd["band"].isin(allowed)]
        .groupby(["state", "band"], as_index=False)["mean_abs_attribution"]
        .sum()
        .sort_values("mean_abs_attribution", ascending=False)
        .head(top_n)
    )
    base = PROJECT_ROOT / "results" / "figures" / "explainability" / "mne_topomaps"
    items = []
    for _, row in grouped.iterrows():
        state = str(row["state"]).lower()
        band = band_slug(row["band"])
        path = base / f"mne_psd_{state}_{band}_signed_attribution_topomap.png"
        items.append((path, f"{row['state']} {row['band']}"))
    return items


def top_wpli_state_band_paths(wpli: pd.DataFrame, *, top_n: int) -> list[tuple[Path, str]]:
    grouped = (
        wpli.groupby(["state", "band"], as_index=False)["mean_abs_attribution"]
        .sum()
        .sort_values("mean_abs_attribution", ascending=False)
        .head(top_n)
    )
    base = PROJECT_ROOT / "results" / "figures" / "explainability" / "mne_wpli_connectivity"
    items = []
    for _, row in grouped.iterrows():
        state = str(row["state"]).lower()
        band = band_slug(row["band"])
        path = base / f"mne_wpli_{state}_{band}_connectivity_top20.png"
        items.append((path, f"{row['state']} {row['band']}"))
    return items


def band_slug(value: object) -> str:
    return str(value).lower().replace(" ", "_").replace("-", "_")


def trim_image_array(img: np.ndarray) -> np.ndarray:
    arr = img
    if arr.ndim == 2:
        mask = arr < 0.98
    else:
        rgb = arr[..., :3]
        alpha = arr[..., 3] if arr.shape[-1] == 4 else np.ones(arr.shape[:2])
        mask = (alpha > 0.02) & (np.nanmin(rgb, axis=2) < 0.985)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    pad = 10
    r0, r1 = max(int(rows.min()) - pad, 0), min(int(rows.max()) + pad + 1, img.shape[0])
    c0, c1 = max(int(cols.min()) - pad, 0), min(int(cols.max()) + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def make_top_feature_group_distributions() -> dict[str, object]:
    labels = subject_labels()
    psd_validation = pd.read_csv(PROJECT_ROOT / "results" / "explainability" / "psd_biomarker_validation.csv")
    wpli_validation = pd.read_csv(PROJECT_ROOT / "results" / "explainability" / "wpli_biomarker_validation.csv")

    psd_top = psd_validation.sort_values("mean_abs_attribution", ascending=False).head(5).copy()
    wpli_top = wpli_validation.sort_values("mean_abs_attribution", ascending=False).head(5).copy()
    psd_values = extract_psd_feature_values(psd_top, labels)
    wpli_values = extract_wpli_feature_values(wpli_top, labels)
    summary = feature_effect_summary(psd_values, wpli_values, psd_top, wpli_top)
    summary.to_csv(OUT_DIR / "figure5_top_feature_effect_summary.csv", index=False)

    fig = plt.figure(figsize=(12.0, 4.05), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.05])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_label(ax_a, "A")
    plot_feature_violin_box(ax_a, psd_values, "Top PSD features", "Standardized log10 PSD")

    panel_label(ax_b, "B")
    plot_feature_violin_box(ax_b, wpli_values, "Top WPLI features", "Standardized WPLI")

    panel_label(ax_c, "C")
    plot_effect_summary(ax_c, summary)

    base = OUT_DIR / "figure5_top_feature_group_distributions_triptych"
    paths = save_pub(fig, base)
    plt.close(fig)
    return manifest_row(base, paths, "Top PSD/WPLI group distributions and effect-size summary")


def subject_labels() -> pd.DataFrame:
    predictions = pd.read_csv(PROJECT_ROOT / "results" / "predictions" / "paper_locked_model_predictions.csv")
    labels = (
        predictions[predictions["model_name"] == FINAL_MODEL][["subject_id", "y_true"]]
        .drop_duplicates("subject_id")
        .sort_values("subject_id")
        .copy()
    )
    labels["group"] = np.where(labels["y_true"].astype(int) == 1, "Proportional", "Poor")
    return labels


def extract_psd_feature_values(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    records = []
    feature_root = PROJECT_ROOT / "runs" / "baseline_rerun_20260529" / "data" / "features" / "psd"
    for feature_rank, (_, feature) in enumerate(features.iterrows(), start=1):
        state = str(feature["state"])
        channel = str(feature["channel"])
        channel_index = int(feature["channel_index"])
        frequency_bin = int(feature["frequency_bin"])
        frequency_hz = float(feature["frequency_hz"])
        label = f"{state} {channel} {frequency_hz:g} Hz"
        for _, subject in labels.iterrows():
            subject_id = str(subject["subject_id"])
            path = feature_root / f"{subject_id}_{state}_psd.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=True) as data:
                psd = np.asarray(data["psd"], dtype=float)
                value = float(psd[channel_index, frequency_bin])
            records.append(
                {
                    "family": "PSD",
                    "feature_rank": feature_rank,
                    "feature_label": label,
                    "subject_id": subject_id,
                    "group": subject["group"],
                    "raw_value": value,
                    "plot_value": math.log10(max(value, 0.0) + 1e-9),
                }
            )
    return standardize_by_feature(pd.DataFrame(records))


def extract_wpli_feature_values(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    records = []
    feature_root = PROJECT_ROOT / "runs" / "baseline_rerun_20260529" / "data" / "features" / "fc"
    for feature_rank, (_, feature) in enumerate(features.iterrows(), start=1):
        state = str(feature["state"])
        edge_index = int(feature["edge_index"])
        band_index = int(feature["band_index"])
        label = f"{state} {feature['channel_i']}-{feature['channel_j']} {feature['band']}"
        for _, subject in labels.iterrows():
            subject_id = str(subject["subject_id"])
            path = feature_root / f"{subject_id}_{state}_fc.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=True) as data:
                wpli = np.asarray(data["wpli"], dtype=float)
                value = float(wpli[edge_index, band_index])
            records.append(
                {
                    "family": "WPLI",
                    "feature_rank": feature_rank,
                    "feature_label": label,
                    "subject_id": subject_id,
                    "group": subject["group"],
                    "raw_value": value,
                    "plot_value": value,
                }
            )
    return standardize_by_feature(pd.DataFrame(records))


def standardize_by_feature(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(z_value=pd.Series(dtype=float))
    values = []
    for _, sub in frame.groupby("feature_label", sort=False):
        mean = float(sub["plot_value"].mean())
        std = float(sub["plot_value"].std(ddof=1))
        std = std if std > 1e-12 else 1.0
        values.append(sub.assign(z_value=(sub["plot_value"] - mean) / std))
    return pd.concat(values, ignore_index=True)


def plot_feature_violin_box(ax: plt.Axes, frame: pd.DataFrame, title: str, ylabel: str) -> None:
    if frame.empty:
        ax.text(0.5, 0.5, "No feature values found", ha="center", va="center")
        ax.set_axis_off()
        return
    labels = frame[["feature_rank", "feature_label"]].drop_duplicates().sort_values("feature_rank")
    x = np.arange(len(labels))
    offsets = {"Poor": -0.18, "Proportional": 0.18}
    colors = {"Poor": COLORS["poor"], "Proportional": COLORS["prop"]}
    rng = np.random.default_rng(20260611)

    for i, (_, row) in enumerate(labels.iterrows()):
        feature = row["feature_label"]
        for group in ["Poor", "Proportional"]:
            data = frame[(frame["feature_label"] == feature) & (frame["group"] == group)]["z_value"].to_numpy(float)
            if len(data) == 0:
                continue
            pos = i + offsets[group]
            parts = ax.violinplot([data], positions=[pos], widths=0.28, showmeans=False, showextrema=False, showmedians=False)
            for body in parts["bodies"]:
                body.set_facecolor(colors[group])
                body.set_edgecolor("none")
                body.set_alpha(0.28)
            ax.boxplot(
                [data],
                positions=[pos],
                widths=0.14,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": COLORS["ink"], "linewidth": 0.8},
                boxprops={"facecolor": "white", "edgecolor": colors[group], "linewidth": 0.8},
                whiskerprops={"color": colors[group], "linewidth": 0.7},
                capprops={"color": colors[group], "linewidth": 0.7},
            )
            jitter = rng.normal(0.0, 0.025, size=len(data))
            ax.scatter(np.full(len(data), pos) + jitter, data, s=13, color=colors[group], alpha=0.85, edgecolor="white", linewidth=0.25, zorder=3)

    ax.axhline(0, color="#111827", linewidth=0.7, linestyle=(0, (2, 2)))
    ax.set_xticks(x, [wrap_label(label, width=14) for label in labels["feature_label"]], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(
        handles=[Patch(facecolor=COLORS["poor"], alpha=0.35, label="Poor recovery"), Patch(facecolor=COLORS["prop"], alpha=0.35, label="Proportional recovery")],
        frameon=False,
        loc="upper right",
    )
    style_axis(ax)


def feature_effect_summary(psd_values: pd.DataFrame, wpli_values: pd.DataFrame, psd_top: pd.DataFrame, wpli_top: pd.DataFrame) -> pd.DataFrame:
    frames = []
    validation_rows = []
    psd_lookup = {}
    for _, row in psd_top.iterrows():
        label = f"{row['state']} {row['channel']} {float(row['frequency_hz']):g} Hz"
        psd_lookup[label] = row
        validation_rows.append(("PSD", label, float(row["label_group_permutation_p"])))
    wpli_lookup = {}
    for _, row in wpli_top.iterrows():
        label = f"{row['state']} {row['channel_i']}-{row['channel_j']} {row['band']}"
        wpli_lookup[label] = row
        validation_rows.append(("WPLI", label, float(row["label_group_permutation_p"])))

    p_table = pd.DataFrame(validation_rows, columns=["family", "feature_label", "raw_p"])
    p_table["fdr_p"] = bh_fdr(p_table["raw_p"].to_numpy(float))
    p_lookup = p_table.set_index("feature_label")

    all_values = pd.concat([psd_values, wpli_values], ignore_index=True)
    for (family, feature_label), sub in all_values.groupby(["family", "feature_label"], sort=False):
        prop = sub[sub["group"] == "Proportional"]["plot_value"].to_numpy(float)
        poor = sub[sub["group"] == "Poor"]["plot_value"].to_numpy(float)
        effect = cohen_d(prop, poor)
        raw_p = float(p_lookup.loc[feature_label, "raw_p"]) if feature_label in p_lookup.index else np.nan
        fdr_p = float(p_lookup.loc[feature_label, "fdr_p"]) if feature_label in p_lookup.index else np.nan
        frames.append(
            {
                "family": family,
                "feature_label": feature_label,
                "cohen_d_proportional_vs_poor": effect,
                "raw_p": raw_p,
                "fdr_p": fdr_p,
                "n_proportional": len(prop),
                "n_poor": len(poor),
            }
        )
    return pd.DataFrame(frames).sort_values("cohen_d_proportional_vs_poor", key=lambda s: s.abs(), ascending=False)


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    pooled_num = (len(group_a) - 1) * np.var(group_a, ddof=1) + (len(group_b) - 1) * np.var(group_b, ddof=1)
    pooled_den = len(group_a) + len(group_b) - 2
    pooled = math.sqrt(max(pooled_num / pooled_den, 1e-12))
    return float((np.mean(group_a) - np.mean(group_b)) / pooled)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    out[valid] = tmp
    return out


def plot_effect_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    if summary.empty:
        ax.text(0.5, 0.5, "No effect summary", ha="center", va="center")
        ax.set_axis_off()
        return
    plot = summary.copy().iloc[::-1]
    y = np.arange(len(plot))
    colors = [COLORS["psd"] if family == "PSD" else COLORS["wpli"] for family in plot["family"]]
    ax.barh(y, plot["cohen_d_proportional_vs_poor"], color=colors, height=0.68)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_yticks(y, [wrap_label(label, width=20) for label in plot["feature_label"]])
    ax.set_xlabel("Cohen's d\n(proportional - poor)")
    ax.set_title("Effect size and P values")
    xmin, xmax = ax.get_xlim()
    text_x = xmax + 0.05 * (xmax - xmin)
    for yi, (_, row) in zip(y, plot.iterrows(), strict=True):
        ax.text(
            text_x,
            yi,
            f"p={format_p(row['raw_p'])}, q={format_p(row['fdr_p'])}",
            va="center",
            ha="left",
            fontsize=5.8,
            clip_on=False,
        )
    ax.set_xlim(min(xmin, -2.8), max(xmax, 2.8))
    ax.legend(
        handles=[Patch(facecolor=COLORS["psd"], label="PSD"), Patch(facecolor=COLORS["wpli"], label="WPLI")],
        frameon=False,
        loc="lower left",
    )
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.75)


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def wrap_label(label: str, *, width: int) -> str:
    words = str(label).split()
    lines: list[str] = []
    line = ""
    for word in words:
        if len(line) + len(word) + 1 <= width:
            line = word if not line else f"{line} {word}"
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return "\n".join(lines)


def manifest_row(base: Path, paths: dict[str, str], description: str) -> dict[str, object]:
    png_path = Path(paths["png"])
    return {
        "figure": base.name,
        "description": description,
        "png": paths["png"],
        "svg": paths["svg"],
        "pdf": paths["pdf"],
        "tiff": paths["tiff"],
        "png_bytes": png_path.stat().st_size if png_path.exists() else np.nan,
    }


def write_captions(manifest: list[dict[str, object]]) -> None:
    text = """# Requested triptych figure captions

## Figure 1. Final model discrimination and calibration
A, Receiver operating characteristic curves for the residual-aware SSL-CNN and Logistic L1 baseline; the dashed diagonal is random classification. B, Confusion matrix for the final model; each cell reports count and row percentage by observed recovery group. C, Reliability curves with bin counts annotated; inset bars show Brier score, where lower values indicate better calibrated probabilities.

## Figure 2. Model components and optimization behavior
A, Mean 10-seed metrics for CNN-based models, with Logistic L1 shown as the single-LOSO reference. B, Core module contribution comparing the no-SSL CNN, Barlow SSL-CNN without residual-aware heads, residual-aware CNN without SSL, and residual-aware SSL-CNN. C, Mean training and validation losses across folds/seeds; residual-aware auxiliary loss is the weighted sum of residual regression, pairwise ranking, and soft-label losses.

## Figure 3. Final-model feature, state, and band ablations
A, Feature modality ablations. B, EO/EC and state-feature ablations. C, beta-band and motor-network WPLI constraints. Bars show balanced accuracy and black dots show ROC-AUC. Hatched bars mark the PSD EO + WPLI EC result that is available only from the legacy tabular Logistic L2 ablation table, not from the final SSL-CNN masked-inference run.

## Figure 4. Explainability maps
A, Global EEG feature importance ranked by mean absolute attribution. B, Existing MNE PSD topomaps: color encodes signed PSD attribution with a diverging scale centered at zero. C, Existing MNE WPLI connectivity maps: red edges indicate positive signed attribution, blue edges indicate negative signed attribution, and line width scales with mean absolute attribution.

## Figure 5. Top-feature group distributions
A, Top PSD features in proportional versus poor recovery groups; values are patient-level log10 PSD values standardized within each feature, with individual patients overlaid. B, Top WPLI features with patient-level WPLI values standardized within feature. C, Cohen's d compares proportional minus poor recovery groups; raw P values are the project label-group permutation P values, and q values are Benjamini-Hochberg FDR corrections across the displayed top PSD/WPLI features.
"""
    (OUT_DIR / "requested_triptych_captions.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
