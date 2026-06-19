from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "paper_panels"
DESKTOP_FIG_DIR = Path(r"C:\Users\HPGZZ\Desktop\小论文\论文图片\3.3")
OUT_STEM = OUT_DIR / "figure5_abc_feature_state_band_ablation_native_composite"
DESKTOP_OUT_STEM = DESKTOP_FIG_DIR / "figure5_abc_feature_state_band_ablation_native_composite"
SOURCE_OUT = OUT_DIR / "figure5_abc_feature_state_band_ablation_native_composite_source_data.csv"

FEATURE_SCRIPT = PROJECT_ROOT / "scripts" / "110_make_feature_modality_ablation_plot.py"
STATE_SCRIPT = PROJECT_ROOT / "scripts" / "111_make_state_ablation_plot.py"
BAND_SCRIPT = (
    PROJECT_ROOT
    / "final_model_ablation_explainability_results_20260610"
    / "scripts"
    / "46_plot_band_only_leave_band_out_ranking.py"
)


def load_script(path: Path, name: str) -> ModuleType:
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plotting script: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


feature_plot = load_script(FEATURE_SCRIPT, "figure5a_feature_modality_ablation_plot")
state_plot = load_script(STATE_SCRIPT, "figure5b_state_ablation_plot")
band_plot = load_script(BAND_SCRIPT, "figure5c_band_only_leave_band_out_ranking")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#263238",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.045) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color="black",
    )


def plot_grouped_metric_bars(
    ax: plt.Axes,
    selected: pd.DataFrame,
    *,
    order: list[str],
    labels: dict[str, str],
    title: str,
    show_ylabel: bool,
) -> None:
    x = np.arange(len(order), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, len(feature_plot.METRICS))

    for offset, (mean_col, sd_col, metric_label, color) in zip(offsets, feature_plot.METRICS, strict=True):
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
            capsize=2.4,
            error_kw={
                "elinewidth": 0.75,
                "capthick": 0.75,
                "ecolor": "#263238",
            },
            zorder=3,
            label=metric_label,
        )

    for metric_label, (value, color) in feature_plot.FINAL_REFERENCES.items():
        ax.axhline(value, color=color, linestyle=(0, (3.2, 2.3)), linewidth=0.9, alpha=0.86, zorder=1)

    label_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ref_text = [
        ("Final ROC-AUC", "ROC-AUC", 0.006, "bottom", feature_plot.FINAL_REFERENCES["ROC-AUC"][1]),
        (
            "Final balanced acc.",
            "Balanced accuracy",
            -0.006,
            "top",
            feature_plot.FINAL_REFERENCES["Balanced accuracy"][1],
        ),
        ("Final Brier", "Brier score", 0.006, "bottom", "#7A7F87"),
    ]
    for text, metric_label, offset, va, color in ref_text:
        ax.text(
            1.012,
            feature_plot.FINAL_REFERENCES[metric_label][0] + offset,
            text,
            transform=label_transform,
            color=color,
            ha="left",
            va=va,
            fontsize=5.8,
            clip_on=False,
        )

    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[name] for name in order])
    ax.set_ylabel("Score" if show_ylabel else "")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.grid(axis="y", color="#D7DEE8", linewidth=0.55, zorder=0)
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)


def plot_band_ranking(ax: plt.Axes, plot_data: pd.DataFrame) -> None:
    y_positions = np.arange(len(plot_data))[::-1]
    balanced_accuracy = plot_data["balanced_accuracy_mean"].to_numpy(dtype=float)
    roc_auc = plot_data["roc_auc_mean"].to_numpy(dtype=float)

    for y, bal, roc in zip(y_positions, balanced_accuracy, roc_auc, strict=True):
        ax.hlines(y, min(bal, roc), max(bal, roc), color="#C7CDD4", linewidth=1.15, zorder=1)

    ax.scatter(
        balanced_accuracy,
        y_positions,
        s=35,
        color=band_plot.BALANCED_ACCURACY_COLOR,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
        label="Balanced accuracy",
    )
    ax.scatter(
        roc_auc,
        y_positions,
        s=35,
        color=band_plot.ROC_AUC_COLOR,
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
        label="ROC-AUC",
    )

    ax.axvline(
        band_plot.FINAL_BALANCED_ACCURACY,
        color="#7C838A",
        linestyle=(0, (3.0, 2.5)),
        linewidth=0.82,
        alpha=0.58,
        zorder=0,
    )
    ax.axvline(
        band_plot.FINAL_ROC_AUC,
        color="#7C838A",
        linestyle=(0, (1.0, 2.2)),
        linewidth=0.82,
        alpha=0.58,
        zorder=0,
    )

    separator_y = (y_positions[7] + y_positions[8]) / 2
    ax.axhline(separator_y, color="#D3D8DE", linewidth=0.85, zorder=0)

    label_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.015,
        float(np.mean(y_positions[:8])),
        "Leave-band-out",
        transform=label_transform,
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#5A626B",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    ax.text(
        0.015,
        float(np.mean(y_positions[8:])),
        "Band-only",
        transform=label_transform,
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#5A626B",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    ax.set_title("Band-only and leave-band-out ablation ranking", loc="left", pad=8, fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_xlim(0.2, 1.0)
    ax.set_xticks(np.arange(0.2, 1.01, 0.1))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["display_label"])
    ax.set_ylim(-0.65, len(plot_data) - 0.35)
    ax.grid(axis="x", color="#E3E7EB", linewidth=0.55, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def make_global_legend(fig: plt.Figure) -> None:
    balanced_handle = Patch(facecolor=feature_plot.METRICS[0][3], edgecolor="white", label="Balanced accuracy")
    roc_handle = (
        Patch(facecolor=feature_plot.METRICS[1][3], edgecolor="white"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=band_plot.ROC_AUC_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=5.2,
        ),
    )
    brier_handle = Patch(facecolor=feature_plot.METRICS[2][3], edgecolor="white", label="Brier score")
    final_handle = Line2D(
        [0],
        [0],
        color="#7A7F87",
        linestyle=(0, (3.2, 2.3)),
        linewidth=0.95,
        label="Final model reference",
    )
    fig.legend(
        handles=[balanced_handle, roc_handle, brier_handle, final_handle],
        labels=["Balanced accuracy", "ROC-AUC", "Brier score", "Final model reference"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower center",
        bbox_to_anchor=(0.5, 0.062),
        ncol=4,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.5,
    )


def save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")


def write_source_data(feature_source: pd.DataFrame, state_source: pd.DataFrame, band_source: pd.DataFrame) -> None:
    feature_long = feature_plot.make_long_source(feature_source)
    feature_long.insert(0, "panel", "A")

    state_long = state_plot.make_long_source(state_source)
    state_long.insert(0, "panel", "B")

    band_long = band_source[
        [
            "ablation_name",
            "display_label",
            "block",
            "balanced_accuracy_mean",
            "roc_auc_mean",
        ]
    ].copy()
    band_long = band_long.rename(columns={"display_label": "display"})
    band_long["final_balanced_accuracy"] = band_plot.FINAL_BALANCED_ACCURACY
    band_long["final_roc_auc"] = band_plot.FINAL_ROC_AUC
    band_long.insert(0, "panel", "C")

    source = pd.concat([feature_long, state_long, band_long], ignore_index=True, sort=False)
    SOURCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(SOURCE_OUT, index=False, encoding="utf-8-sig")


def main() -> None:
    configure_matplotlib()
    feature_source = feature_plot.load_source()
    state_source = state_plot.load_source()
    band_source = band_plot.load_and_prepare_plot_data()

    fig = plt.figure(figsize=(8.8, 9.6), constrained_layout=False)
    ax_a = fig.add_axes([0.075, 0.645, 0.36, 0.285])
    ax_b = fig.add_axes([0.575, 0.645, 0.36, 0.285])
    ax_c = fig.add_axes([0.265, 0.16, 0.70, 0.405])

    plot_grouped_metric_bars(
        ax_a,
        feature_source,
        order=feature_plot.ABLATION_ORDER,
        labels=feature_plot.ABLATION_LABELS,
        title="Feature modality ablation",
        show_ylabel=True,
    )
    plot_grouped_metric_bars(
        ax_b,
        state_source,
        order=state_plot.ABLATION_ORDER,
        labels=state_plot.ABLATION_LABELS,
        title="State ablation",
        show_ylabel=False,
    )
    plot_band_ranking(ax_c, band_source)

    panel_label(ax_a, "A", x=-0.18, y=1.06)
    panel_label(ax_b, "B", x=-0.18, y=1.06)
    panel_label(ax_c, "C", x=-0.13, y=1.035)

    make_global_legend(fig)

    save_all(fig, OUT_STEM)
    save_all(fig, DESKTOP_OUT_STEM)
    write_source_data(feature_source, state_source, band_source)
    plt.close(fig)

    print(f"png={OUT_STEM.with_suffix('.png')}")
    print(f"tiff={OUT_STEM.with_suffix('.tiff')}")
    print(f"pdf={OUT_STEM.with_suffix('.pdf')}")
    print(f"svg={OUT_STEM.with_suffix('.svg')}")
    print(f"desktop_png={DESKTOP_OUT_STEM.with_suffix('.png')}")
    print(f"desktop_tiff={DESKTOP_OUT_STEM.with_suffix('.tiff')}")
    print(f"source={SOURCE_OUT}")


if __name__ == "__main__":
    main()
