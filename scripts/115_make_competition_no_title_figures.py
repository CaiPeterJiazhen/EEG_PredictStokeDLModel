from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import transforms
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(r"C:\Users\HPGZZ\Desktop\2026生物医学工程竞赛\比赛图")

FIG_REVISED = ROOT / "results" / "figures" / "revised_initial"
FIG_PAPER = ROOT / "results" / "figures" / "paper_panels"
METRIC_DIR = ROOT / "results" / "metrics"

ROC_SOURCE = FIG_REVISED / "figure4a_final_model_roc_source_data.csv"
CM_SOURCE = FIG_REVISED / "figure4b_final_model_confusion_matrix_traditional_style.csv"
LOSS_SOURCE = METRIC_DIR / "final_model_loss_curve_summary.csv"
SCORECARD_SOURCE = FIG_REVISED / "figure4c_a_model_metric_histogram_source_data.csv"
SEED_SOURCE = FIG_REVISED / "figure5a_seed_stability_source_data.csv"
BRIER_SOURCE = FIG_REVISED / "figure4c_b_brier_calibration_source_data.csv"

ML_SCRIPT = ROOT / "scripts" / "98_make_ml_baseline_two_panel.py"
FEATURE_SCRIPT = ROOT / "scripts" / "110_make_feature_modality_ablation_plot.py"
STATE_SCRIPT = ROOT / "scripts" / "111_make_state_ablation_plot.py"
BAND_SCRIPT = (
    ROOT
    / "final_model_ablation_explainability_results_20260610"
    / "scripts"
    / "46_plot_band_only_leave_band_out_ranking.py"
)


PALETTE = {
    "purple": "#8E44AD",
    "pink": "#F06C91",
    "blue": "#2F5E8C",
    "orange": "#E66A00",
    "teal": "#1B9E77",
    "gold": "#D9A43A",
    "green": "#2EAD74",
    "red": "#D66A5C",
    "ink": "#1F2D3A",
    "grid": "#DCE4EE",
}

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
SCORECARD_METRICS = [
    ("accuracy", "Mean\naccuracy"),
    ("balanced_accuracy", "Balanced\naccuracy"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]

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
CN_FONT: font_manager.FontProperties | None = None


def load_script(path: Path, name: str) -> ModuleType:
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plotting script: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_matplotlib() -> None:
    global CN_FONT
    chinese_font_paths = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for font_path in chinese_font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    for font_path in chinese_font_paths:
        if font_path.exists():
            CN_FONT = font_manager.FontProperties(fname=str(font_path))
            break
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans SC", "Microsoft YaHei", "SimHei", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "axes.edgecolor": PALETTE["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


def remove_stem(stem: Path) -> None:
    for suffix in [".png", ".tiff", ".pdf", ".svg"]:
        path = stem.with_suffix(suffix)
        if path.exists():
            path.unlink()


def plot_roc(ax: plt.Axes) -> None:
    roc = pd.read_csv(ROC_SOURCE)
    auc = float(roc["roc_auc"].iloc[0])
    ax.step(
        roc["fpr"],
        roc["tpr"],
        where="post",
        color=PALETTE["purple"],
        lw=1.8,
        label=f"AUC = {auc:.2f}",
    )
    ax.plot([0, 1], [0, 1], color=PALETTE["pink"], lw=1.0, ls=(0, (3, 2)))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=9.2, fontweight="bold", pad=7)
    ax.grid(True, color=PALETTE["grid"], lw=0.7, ls=":")
    leg = ax.legend(loc="lower right", frameon=True)
    leg.get_frame().set_edgecolor(PALETTE["purple"])
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(1.0)


def plot_confusion(ax: plt.Axes, fig: plt.Figure) -> None:
    frame = pd.read_csv(CM_SOURCE)
    labels = ["Positive", "Negative"]
    counts = np.zeros((2, 2), dtype=float)
    percents = np.zeros((2, 2), dtype=float)
    cells = np.empty((2, 2), dtype=object)
    for _, row in frame.iterrows():
        i = labels.index(row["true_label"])
        j = labels.index(row["predicted_label"])
        counts[i, j] = row["count"]
        percents[i, j] = row["percent_of_total"]
        cells[i, j] = row["cell"]

    im = ax.imshow(counts, cmap=mpl.colormaps["RdBu_r"], norm=mcolors.TwoSlopeNorm(vmin=0, vcenter=5, vmax=10))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, rotation=90, va="center")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=2)
    ax.set_xlabel("Predicted Label", fontsize=9.0, fontweight="bold", labelpad=9)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("True label", fontsize=9.0, fontweight="bold", labelpad=9)

    for i in range(2):
        for j in range(2):
            value = counts[i, j]
            text_color = "white" if value >= 7 or value <= 1 else PALETTE["ink"]
            label_color = "white" if value >= 7 or value <= 1 else PALETTE["ink"]
            ax.text(j - 0.42, i - 0.35, str(cells[i, j]), ha="left", va="top", fontsize=9.0, fontweight="bold", color=label_color)
            ax.text(j, i - 0.03, f"{int(value)}", ha="center", va="center", fontsize=17.0, fontweight="bold", color=text_color)
            ax.text(j, i + 0.35, f"{percents[i, j] * 100:.1f}% of total", ha="center", va="center", fontsize=6.6, color=text_color)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color(PALETTE["ink"])
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Count", rotation=90, labelpad=8)
    cbar.set_ticks([0, 2, 4, 6, 8, 10])


def plot_loss_curves(axes: list[plt.Axes]) -> None:
    summary = pd.read_csv(LOSS_SOURCE)
    colors = {
        "train_loss_total": "#1F77B4",
        "val_loss_total": "#FF7F0E",
        "bce": PALETTE["blue"],
        "residual": PALETTE["orange"],
        "ranking": "#7A6FB3",
        "soft": PALETTE["teal"],
    }

    ax = axes[0]
    for metric, label, color in [
        ("train_loss_total", "Training total", colors["train_loss_total"]),
        ("val_loss_total", "Validation total", colors["val_loss_total"]),
    ]:
        s = summary[summary["metric"].eq(metric)]
        x = s["epoch"].to_numpy(dtype=float)
        y = s["value_mean"].to_numpy(dtype=float)
        e = s["value_sem"].to_numpy(dtype=float)
        ax.plot(x, y, lw=1.45, color=color, label=label)
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.16, lw=0)
    ax.set_ylabel("Loss")
    ax.set_title("总损失", fontproperties=CN_FONT, fontsize=9.0, pad=6)
    ax.legend(loc="upper right")

    for ax, metrics, ylabel, title in [
        (
            axes[1],
            [
                ("train_weighted_bce", "BCE", "bce"),
                ("train_weighted_residual", "Residual", "residual"),
                ("train_weighted_rank", "Ranking", "ranking"),
                ("train_weighted_soft", "Soft label", "soft"),
            ],
            "Training weighted contribution",
            "加权训练损失分量",
        ),
        (
            axes[2],
            [
                ("val_weighted_bce", "BCE", "bce"),
                ("val_weighted_residual", "Residual", "residual"),
                ("val_weighted_rank", "Ranking", "ranking"),
                ("val_weighted_soft", "Soft label", "soft"),
            ],
            "Validation weighted contribution",
            "加权验证损失分量",
        ),
    ]:
        for metric, label, color_key in metrics:
            s = summary[summary["metric"].eq(metric)]
            ax.plot(s["epoch"], s["value_mean"], lw=1.25, color=colors[color_key], label=label)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontproperties=CN_FONT, fontsize=9.0, pad=6)
        ax.legend(loc="upper right")

    for ax in axes:
        ax.axvline(50, color="#7A7A7A", lw=0.9, ls="--")
        ax.text(48, ax.get_ylim()[1] * 0.94, "SWA", color="#555555", fontsize=7.0, ha="right")
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 105)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)


def plot_scorecard(ax: plt.Axes) -> None:
    data = pd.read_csv(SCORECARD_SOURCE).copy()
    data["model"] = pd.Categorical(data["model"], categories=SCORECARD_ORDER, ordered=True)
    data = data.sort_values("model").reset_index(drop=True)

    x = np.arange(len(SCORECARD_METRICS), dtype=float)
    width = 0.13
    offsets = (np.arange(len(data)) - (len(data) - 1) / 2) * width

    for offset, (_, row) in zip(offsets, data.iterrows(), strict=True):
        model = str(row["model"])
        values = [float(row[metric]) for metric, _ in SCORECARD_METRICS]
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

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in SCORECARD_METRICS])
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.58, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False, handlelength=1.1, columnspacing=1.35)


def plot_seed_stability(ax: plt.Axes) -> None:
    source = pd.read_csv(SEED_SOURCE)
    positions = np.arange(1, len(SEED_ORDER) + 1)
    grouped = [source.loc[source["model"].eq(label), "accuracy"].astype(float).to_numpy() for label in SEED_ORDER]

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
        ax.scatter(np.full_like(values, pos, dtype=float) + jitter, values, s=17, color=color, edgecolor="white", linewidth=0.42, zorder=3)
        ax.scatter([pos], [np.median(values)], marker="D", s=31, facecolor="white", edgecolor=color, linewidth=1.0, zorder=4)

    ax.set_ylabel("Seed-level accuracy")
    ax.set_xticks(positions)
    ax.set_xticklabels(SEED_XLABELS)
    ax.set_ylim(0.50, 0.93)
    ax.set_yticks(np.arange(0.50, 0.95, 0.05))
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.56, zorder=0)
    ax.text(0.02, 0.02, "Diamonds mark medians", transform=ax.transAxes, fontsize=6.0, color="#667085")


def plot_brier(ax: plt.Axes) -> None:
    data = pd.read_csv(BRIER_SOURCE).sort_values("brier_score", ascending=True).reset_index(drop=True)
    y = np.arange(len(data))
    values = data["brier_score"].astype(float).to_numpy()
    labels = data["display"].tolist()
    colors = [BRIER_COLORS.get(label, "#4E79A7") for label in labels]
    ax.hlines(y, 0, values, color="#D8DEE6", linewidth=1.0, zorder=1)
    ax.scatter(values, y, s=42, color=colors, edgecolor="white", linewidth=0.55, zorder=3)
    for yi, value in zip(y, values, strict=True):
        ax.text(value + 0.008, yi, f"{value:.3f}", va="center", ha="left", fontsize=7.0, color="#263238")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.245, float(values.max()) + 0.035))
    ax.set_xticks(np.arange(0, 0.26, 0.05))
    ax.set_xlabel("Brier score")
    ax.grid(axis="x", color="#D8DEE6", linewidth=0.52, zorder=0)
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.52, zorder=0)


def plot_grouped_ablation_bars(ax: plt.Axes, selected: pd.DataFrame, module: ModuleType, show_ylabel: bool) -> None:
    x = np.arange(len(module.ABLATION_ORDER), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, len(module.METRICS))

    for offset, (mean_col, sd_col, metric_label, color) in zip(offsets, module.METRICS, strict=True):
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
            error_kw={"elinewidth": 0.75, "capthick": 0.75, "ecolor": "#263238"},
            zorder=3,
            label=metric_label,
        )
    for _, (value, color) in module.FINAL_REFERENCES.items():
        ax.axhline(value, color=color, linestyle=(0, (3.2, 2.3)), linewidth=0.9, alpha=0.86, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([module.ABLATION_LABELS[name] for name in module.ABLATION_ORDER])
    ax.set_ylabel("Score" if show_ylabel else "")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_xlim(-0.5, len(module.ABLATION_ORDER) - 0.5)
    ax.grid(axis="y", color="#D7DEE8", linewidth=0.55, zorder=0)
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)


def plot_band_ranking(ax: plt.Axes, band_plot: ModuleType, plot_data: pd.DataFrame) -> None:
    y_positions = np.arange(len(plot_data))[::-1]
    balanced_accuracy = plot_data["balanced_accuracy_mean"].to_numpy(dtype=float)
    roc_auc = plot_data["roc_auc_mean"].to_numpy(dtype=float)
    for y, bal, roc in zip(y_positions, balanced_accuracy, roc_auc, strict=True):
        ax.hlines(y, min(bal, roc), max(bal, roc), color="#C7CDD4", linewidth=1.15, zorder=1)
    ax.scatter(balanced_accuracy, y_positions, s=35, color=band_plot.BALANCED_ACCURACY_COLOR, edgecolor="white", linewidth=0.55, zorder=3, label="Balanced accuracy")
    ax.scatter(roc_auc, y_positions, s=35, color=band_plot.ROC_AUC_COLOR, edgecolor="white", linewidth=0.55, zorder=4, label="ROC-AUC")
    ax.axvline(band_plot.FINAL_BALANCED_ACCURACY, color="#7C838A", linestyle=(0, (3.0, 2.5)), linewidth=0.82, alpha=0.58, zorder=0)
    ax.axvline(band_plot.FINAL_ROC_AUC, color="#7C838A", linestyle=(0, (1.0, 2.2)), linewidth=0.82, alpha=0.58, zorder=0)
    separator_y = (y_positions[7] + y_positions[8]) / 2
    ax.axhline(separator_y, color="#D3D8DE", linewidth=0.85, zorder=0)
    label_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.015, float(np.mean(y_positions[:8])), "Leave-band-out", transform=label_transform, ha="left", va="center", fontsize=7.0, fontweight="bold", color="#5A626B", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5})
    ax.text(0.015, float(np.mean(y_positions[8:])), "Band-only", transform=label_transform, ha="left", va="center", fontsize=7.0, fontweight="bold", color="#5A626B", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5})
    ax.set_xlabel("Score")
    ax.set_xlim(0.2, 1.0)
    ax.set_xticks(np.arange(0.2, 1.01, 0.1))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["display_label"])
    ax.set_ylim(-0.65, len(plot_data) - 0.35)
    ax.grid(axis="x", color="#E3E7EB", linewidth=0.55, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)


def make_ml_baseline_two_panel() -> None:
    ml_plot = load_script(ML_SCRIPT, "ml_baseline_two_panel")
    polar_source = pd.read_csv(ml_plot.POLAR_SOURCE)
    baseline = ml_plot.load_key_baselines()
    fig = plt.figure(figsize=(7.7, 3.95), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.055, right=0.972, top=0.94, bottom=0.15, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0], polar=True)
    ml_plot.draw_polar_panel(ax_a, polar_source)
    ax_b = ml_plot.draw_baseline_panel(fig, gs[0, 1], baseline)
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    ax_b.set_position([pos_b.x0, pos_a.y0, pos_b.width * 0.88, pos_a.height])
    save_all(fig, OUT_ROOT / "3.1" / "figure_ml_baseline_two_panel")
    plt.close(fig)


def make_final_model_two_and_loss() -> None:
    fig = plt.figure(figsize=(7.6, 3.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.965, top=0.91, bottom=0.16, wspace=0.35)
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_cm = fig.add_subplot(gs[0, 1])
    plot_roc(ax_roc)
    plot_confusion(ax_cm, fig)
    save_all(fig, OUT_ROOT / "3.2" / "figure4_final_model_roc_confusion_two_panel")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.45), constrained_layout=True)
    plot_loss_curves(list(axes))
    save_all(fig, OUT_ROOT / "3.2" / "figure4_final_model_loss_curves")
    plt.close(fig)


def make_model_score_seed_and_brier() -> None:
    fig, ax_score = plt.subplots(figsize=(8.2, 3.35), constrained_layout=True)
    plot_scorecard(ax_score)
    save_all(fig, OUT_ROOT / "3.2" / "figure5_model_scorecard_single")
    plt.close(fig)

    fig = plt.figure(figsize=(7.8, 3.35), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.06, right=0.985, top=0.965, bottom=0.16, wspace=0.38)
    ax_seed = fig.add_subplot(gs[0, 0])
    ax_brier = fig.add_subplot(gs[0, 1])
    plot_seed_stability(ax_seed)
    plot_brier(ax_brier)
    save_all(fig, OUT_ROOT / "3.2" / "figure5_seed_stability_brier_two_panel")
    plt.close(fig)


def make_ablation_two_and_single() -> None:
    feature_plot = load_script(FEATURE_SCRIPT, "feature_modality_ablation_plot_competition")
    state_plot = load_script(STATE_SCRIPT, "state_ablation_plot_competition")
    band_plot = load_script(BAND_SCRIPT, "band_ablation_ranking_competition")
    feature_source = feature_plot.load_source()
    state_source = state_plot.load_source()
    band_source = band_plot.load_and_prepare_plot_data()

    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.55), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.255, wspace=0.035)
    plot_grouped_ablation_bars(axes[0], feature_source, feature_plot, show_ylabel=True)
    plot_grouped_ablation_bars(axes[1], state_source, state_plot, show_ylabel=False)
    handles = [Patch(facecolor=color, edgecolor="white", label=label) for _, _, label, color in feature_plot.METRICS]
    handles.append(Line2D([0], [0], color="#7A7F87", linestyle=(0, (3.2, 2.3)), linewidth=0.95, label="Final model reference"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=4, frameon=False, handlelength=1.55)
    save_all(fig, OUT_ROOT / "3.3" / "figure5_feature_state_ablation_two_panel")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    plot_band_ranking(ax, band_plot, band_source)
    save_all(fig, OUT_ROOT / "3.3" / "figure5_band_ablation_single")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    remove_stem(OUT_ROOT / "3.2" / "figure5_model_scorecard_seed_two_panel")
    remove_stem(OUT_ROOT / "3.2" / "figure5_brier_score_single")
    make_ml_baseline_two_panel()
    make_final_model_two_and_loss()
    make_model_score_seed_and_brier()
    make_ablation_two_and_single()
    print(f"competition_dir={OUT_ROOT}")


if __name__ == "__main__":
    main()
