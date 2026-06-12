from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence
import importlib.util

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import auc, confusion_matrix, roc_curve

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.config import load_path_config
from eeg_recovery.visualization.topomap import normalize_channel_name

PSD_BANDS = ("Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High", "Gamma")
WPLI_BANDS = ("Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High")
STATES = ("EO", "EC")
FINAL_PREDICTIONS = "seedmean_patient_barlow_residualaware_highrank_swa_clsalpha1_10seed.csv"


COLORS = {
    "ink": "#202020",
    "grid": "#d7dce2",
    "orange": "#ff8a00",
    "blue": "#2f6fc4",
    "red": "#c83b3b",
    "navy": "#1f4e79",
    "light": "#f3f6fb",
    "metric": "#9aa0b8",
    "accent": "#d84c6f",
}


def main() -> None:
    config = load_path_config(PROJECT_ROOT / "configs" / "paths.example.yaml")
    output_dir = config.output_root / "results" / "figures" / "revised_initial"
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_style()
    pred = pd.read_csv(config.output_root / "results" / "predictions" / FINAL_PREDICTIONS)
    make_final_roc(pred, output_dir / "figure4a_final_model_roc")
    make_final_confusion(pred, output_dir / "figure4b_final_model_confusion_matrix")

    topomap_helpers = _load_topomap_helpers()
    names, pos = topomap_helpers._load_ced_eeglab_topomap_layout(
        config.standard_1005_ced,
        CANONICAL_CHANNELS_62,
    )
    psd = pd.read_csv(config.output_root / "results" / "explainability" / "psd_channel_band_importance.csv")
    wpli = pd.read_csv(config.output_root / "results" / "explainability" / "wpli_top_edges.csv")
    make_psd_topomap_grid(psd, names, pos, output_dir / "figure6a_psd_topomap_bands")
    make_wpli_connectivity_grid(wpli, names, pos, output_dir / "figure6b_wpli_connectivity_bands")


def _load_topomap_helpers():
    script_path = PROJECT_ROOT / "scripts" / "45_make_mne_explainability_topomaps.py"
    spec = importlib.util.spec_from_file_location("mne_explainability_topomaps", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.linewidth": 0.8,
        }
    )


def save_pub(fig: plt.Figure, base: Path, *, dpi: int = 600) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def make_final_roc(pred: pd.DataFrame, base: Path) -> None:
    y_true = pred["y_true"].astype(int).to_numpy()
    y_score = pred["y_score"].astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=True)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4.35, 4.0), constrained_layout=True)
    purple = "#8E44AD"
    diagonal = "#EF6F8F"
    ax.plot([0, 1], [0, 1], linestyle=(0, (3.2, 2.3)), color=diagonal, linewidth=0.8)
    ax.plot(
        fpr,
        tpr,
        color=purple,
        linewidth=1.15,
        drawstyle="steps-post",
        label=f"AUC = {roc_auc:.2f}",
    )
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=8.0)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=8.0)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=9.1, fontweight="bold", pad=7)
    ax.grid(True, color="#DADDE3", linestyle=":", linewidth=0.55, alpha=0.95)
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
    save_pub(fig, base)
    plt.close(fig)


def make_final_confusion(pred: pd.DataFrame, base: Path) -> None:
    y_true = pred["y_true"].astype(int).to_numpy()
    y_pred = pred["y_pred"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    grid = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, tp, fn, sensitivity],
            [np.nan, fp, tn, specificity],
            [np.nan, precision, npv, accuracy],
        ],
        dtype=float,
    )
    count_max = max(tp, tn, fp, fn, 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.invert_yaxis()
    ax.axis("off")

    for row in range(4):
        for col in range(4):
            if row == 0 and col == 0:
                continue
            if row == 0 or col == 0:
                color = COLORS["navy"]
            elif row in (1, 2) and col in (1, 2):
                value = grid[row, col]
                intensity = 0.12 + 0.72 * value / count_max
                color = plt.cm.Blues(intensity)
            elif row == 3 and col == 3:
                color = COLORS["accent"]
            else:
                color = COLORS["metric"]
            rect = plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor="white", linewidth=0.8)
            ax.add_patch(rect)

    headers = {1: "Positive", 2: "Negative"}
    side = {1: "Positive", 2: "Negative"}
    for col, label in headers.items():
        ax.text(col + 0.5, 0.5, label, ha="center", va="center", color="white", fontsize=8)
    for row, label in side.items():
        ax.text(0.5, row + 0.5, label, ha="center", va="center", color="white", fontsize=8)

    count_labels = {(1, 1): tp, (1, 2): fn, (2, 1): fp, (2, 2): tn}
    for (row, col), value in count_labels.items():
        ax.text(col + 0.5, row + 0.5, str(int(value)), ha="center", va="center", color="white" if value >= count_max * 0.6 else COLORS["ink"], fontsize=12)

    metric_labels = {
        (1, 3): ("Sensitivity", sensitivity),
        (2, 3): ("Specificity", specificity),
        (3, 1): ("Precision", precision),
        (3, 2): ("Negative predictive\nvalue", npv),
        (3, 3): ("Accuracy", accuracy),
    }
    for (row, col), (label, value) in metric_labels.items():
        ax.text(col + 0.5, row + 0.38, label, ha="center", va="center", color="white", fontsize=8)
        ax.text(col + 0.5, row + 0.68, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)

    dashed = plt.Rectangle((1, 1), 2, 2, fill=False, edgecolor="#e6e000", linewidth=1.5, linestyle=(0, (4, 3)))
    ax.add_patch(dashed)
    ax.text(2.0, -0.18, "Predict Label", ha="center", va="bottom", fontsize=10)
    ax.text(-0.18, 2.0, "True Label", ha="right", va="center", rotation=90, fontsize=10)
    ax.plot([1.1, 1.1, 1.35, 1.55, 2.45, 2.65, 2.9, 2.9], [0.0, -0.05, -0.05, -0.18, -0.18, -0.05, -0.05, 0.0], color=COLORS["ink"], linewidth=0.7, clip_on=False)
    ax.plot([-0.05, -0.18, -0.18, -0.05], [1.1, 1.35, 2.65, 2.9], color=COLORS["ink"], linewidth=0.7, clip_on=False)

    mappable = plt.cm.ScalarMappable(cmap="Blues", norm=mpl.colors.Normalize(vmin=0, vmax=count_max))
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Count")
    cbar.set_ticks(range(0, count_max + 1, max(1, count_max // 4)))
    save_pub(fig, base)
    plt.close(fig)


def make_psd_topomap_grid(psd: pd.DataFrame, names: Sequence[str], pos: np.ndarray, base: Path) -> None:
    values = []
    for state in STATES:
        for band in PSD_BANDS:
            data = _psd_values_for_state_band(psd, state, band)
            values.append(data)
    pooled_abs = np.abs(np.concatenate(values))
    limit = max(float(np.nanpercentile(pooled_abs, 95)), 1e-12)

    fig, axes = plt.subplots(2, len(PSD_BANDS), figsize=(11.1, 3.9), constrained_layout=True)
    last_image = None
    for row, state in enumerate(STATES):
        for col, band in enumerate(PSD_BANDS):
            ax = axes[row, col]
            data = _psd_values_for_state_band(psd, state, band)
            image, _ = mne.viz.plot_topomap(
                data,
                pos,
                axes=ax,
                show=False,
                sensors=True,
                contours=5,
                outlines="head",
                sphere=(0.0, 0.0, 0.0, 1.0),
                extrapolate="head",
                border="mean",
                image_interp="cubic",
                cmap="RdBu_r",
                vlim=(-limit, limit),
            )
            last_image = image
            if row == 0:
                ax.set_title(band.replace("Medium", "Mid"), fontsize=9)
            if col == 0:
                ax.text(-0.18, 0.5, state, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=10, fontweight="bold")
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes, fraction=0.018, pad=0.012)
        cbar.set_ticks([-limit, 0.0, limit])
        cbar.set_ticklabels(["Min", "0", "Max"])
        cbar.set_label("Signed PSD attribution", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle("PSD attribution topomaps by state and frequency band", fontsize=11)
    save_pub(fig, base)
    plt.close(fig)


def make_wpli_connectivity_grid(wpli: pd.DataFrame, names: Sequence[str], pos: np.ndarray, base: Path, *, top_n: int = 20) -> None:
    fig, axes = plt.subplots(2, len(WPLI_BANDS), figsize=(9.8, 4.2), constrained_layout=True)
    max_abs = max(float(wpli["mean_abs_attribution"].max()), 1e-12)
    name_to_index = {normalize_channel_name(name): index for index, name in enumerate(names)}

    for row, state in enumerate(STATES):
        for col, band in enumerate(WPLI_BANDS):
            ax = axes[row, col]
            subset = (
                wpli[(wpli["state"] == state) & (wpli["band"] == band)]
                .sort_values("mean_abs_attribution", ascending=False)
                .head(top_n)
            )
            _draw_connectivity_panel(ax, subset, names, pos, name_to_index, max_abs=max_abs)
            if row == 0:
                ax.set_title(band.replace("Medium", "Mid"), fontsize=9)
            if col == 0:
                ax.text(-0.18, 0.5, state, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=10, fontweight="bold")

    red = plt.Line2D([0], [0], color=COLORS["red"], lw=2, label="positive signed attribution")
    blue = plt.Line2D([0], [0], color=COLORS["blue"], lw=2, label="negative signed attribution")
    fig.legend(handles=[red, blue], loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("WPLI connectivity attribution by state and frequency band", fontsize=11)
    save_pub(fig, base)
    plt.close(fig)


def _draw_connectivity_panel(
    ax: plt.Axes,
    edges: pd.DataFrame,
    names: Sequence[str],
    pos: np.ndarray,
    name_to_index: dict[str, int],
    *,
    max_abs: float,
) -> None:
    image, _ = mne.viz.plot_topomap(
        np.zeros(len(names), dtype=float),
        pos,
        axes=ax,
        show=False,
        sensors=False,
        contours=0,
        outlines="head",
        sphere=(0.0, 0.0, 0.0, 1.0),
        extrapolate="head",
        border="mean",
        cmap="Greys",
        vlim=(-1.0, 1.0),
    )
    image.set_alpha(0.0)
    for _, edge in edges.iterrows():
        source = normalize_channel_name(edge["channel_i"])
        target = normalize_channel_name(edge["channel_j"])
        if source not in name_to_index or target not in name_to_index:
            continue
        start = pos[name_to_index[source]]
        end = pos[name_to_index[target]]
        signed_value = float(edge.get("mean_signed_attribution", 0.0))
        magnitude = float(edge["mean_abs_attribution"])
        color = COLORS["red"] if signed_value >= 0 else COLORS["blue"]
        linewidth = 0.35 + 2.2 * magnitude / max_abs
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth, alpha=0.68, solid_capstyle="round", zorder=2)
    ax.scatter(pos[:, 0], pos[:, 1], s=5, c=COLORS["ink"], linewidths=0, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")


def _psd_values_for_state_band(psd: pd.DataFrame, state: str, band: str) -> np.ndarray:
    frame = psd[(psd["state"] == state) & (psd["band"] == band)].copy()
    frame["channel_norm"] = frame["channel"].map(normalize_channel_name)
    values = frame.groupby("channel_norm", as_index=True)["mean_signed_attribution"].mean().to_dict()
    return np.asarray([float(values.get(normalize_channel_name(channel), 0.0)) for channel in CANONICAL_CHANNELS_62], dtype=float)


if __name__ == "__main__":
    main()
