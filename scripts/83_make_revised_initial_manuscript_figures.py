from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw
from sklearn.metrics import auc, precision_recall_curve, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRIMARY_MODELS = [
    "ML_EEG_updated_no_selector_logistic_l1",
    "ML_EEG_updated_no_selector_logistic_l2",
    "ML_EEG_updated_selectk100_svm_rbf",
    "no_SSL_CNN_updated_sub05_sub28_seedensemble10",
    "residual_aware_SSL_CNN_seedmean10",
]

MODEL_LABELS = {
    "ML_EEG_updated_no_selector_logistic_l1": "Logistic L1",
    "ML_EEG_updated_no_selector_logistic_l2": "Logistic L2",
    "ML_EEG_updated_selectk100_svm_rbf": "SVM RBF",
    "no_SSL_CNN_updated_sub05_sub28_seedensemble10": "CNN\nno SSL",
    "residual_aware_SSL_CNN_seedmean10": "Residual-aware\nSSL-CNN",
}

CURVE_MODELS = [
    "ML_EEG_updated_no_selector_logistic_l1",
    "no_SSL_CNN_updated_sub05_sub28_seedensemble10",
    "residual_aware_SSL_CNN_seedmean10",
]

CURVE_LABELS = {
    "ML_EEG_updated_no_selector_logistic_l1": "Logistic L1",
    "no_SSL_CNN_updated_sub05_sub28_seedensemble10": "CNN no SSL",
    "residual_aware_SSL_CNN_seedmean10": "Residual-aware SSL-CNN",
}

COLORS = {
    "ink": "#263238",
    "muted": "#6B7280",
    "grid": "#D8DEE2",
    "panel": "#F6F8F9",
    "blue": "#3F6F8F",
    "teal": "#4E8D7C",
    "red": "#B85C4A",
    "gold": "#C79B3B",
    "green": "#688E5A",
    "violet": "#756B9B",
    "pale_blue": "#E9F1F4",
    "pale_red": "#F6EDEB",
    "pale_gold": "#FAF4E4",
    "pale_green": "#EEF4EA",
    "pale_violet": "#F0EEF6",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build revised manuscript figures aligned to the new outline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default=PROJECT_ROOT)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = root / "results" / "figures" / "revised_initial"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    rows: list[dict[str, str]] = []
    rows.append(make_figure1(root, output_dir))
    rows.append(make_figure2(root, output_dir))
    rows.append(make_figure3(root, output_dir))
    rows.append(make_figure4(root, output_dir))
    rows.append(make_figure5(root, output_dir))
    rows.append(make_figure6(root, output_dir))
    pd.DataFrame(rows).to_csv(output_dir / "figure_manifest.csv", index=False)
    write_contact_sheet(output_dir, rows)
    print(f"Wrote revised manuscript figures to {output_dir}")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.dpi": 150,
        }
    )


def make_figure1(root: Path, output_dir: Path) -> dict[str, str]:
    flow = pd.read_csv(root / "results" / "tables" / "participant_flow_safety_source_notes.csv")
    counts = dict(zip(flow["category"], flow["n"], strict=False))
    n_records = get_count(counts, "M1 patient records in clinical source workbook")
    n_eeg = get_count(counts, "Current EEG-indexed M1 patient pool")
    n_labelled = get_count(counts, "Final labeled supervised cohort")
    n_unlabelled = get_count(counts, "EEG-indexed patients not used for supervised labels")
    n_pos = get_count(counts, "Proportional-recovery label")
    n_neg = get_count(counts, "Poor-recovery label")

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.1, 1.4], hspace=0.34, wspace=0.28)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    ax_a.set_axis_off()
    panel_label(ax_a, "a")
    ax_a.set_title("Cohort and endpoint", loc="left")
    box(ax_a, 0.50, 0.88, f"Clinical source workbook\nM1 stroke records, n={n_records}", 0.76, 0.12, COLORS["panel"])
    arrow(ax_a, 0.50, 0.80, 0.50, 0.72)
    box(ax_a, 0.50, 0.66, f"EEG-indexed pool\nbaseline EEG, n={n_eeg}", 0.76, 0.12, COLORS["pale_blue"])
    arrow(ax_a, 0.50, 0.58, 0.34, 0.49)
    arrow(ax_a, 0.50, 0.58, 0.66, 0.49)
    box(ax_a, 0.32, 0.42, f"Supervised labels\nn={n_labelled}", 0.46, 0.13, COLORS["pale_gold"])
    box(ax_a, 0.69, 0.42, f"SSL-only EEG\nn={n_unlabelled}", 0.42, 0.13, COLORS["pale_green"])
    box(ax_a, 0.32, 0.25, f"Proportional\nn={n_pos}", 0.30, 0.12, COLORS["pale_red"])
    box(ax_a, 0.32, 0.10, f"Poor recovery\nn={n_neg}", 0.30, 0.12, COLORS["panel"])
    ax_a.text(
        0.69,
        0.24,
        "Incomplete follow-up or\npartial EEG time points;\nused for representation\nlearning only.",
        transform=ax_a.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color=COLORS["ink"],
    )

    ax_b.set_axis_off()
    panel_label(ax_b, "b")
    ax_b.set_title("Clinical acquisition", loc="left")
    stages = [
        ("Baseline EEG\nSynAmps2\n64 channels", 0.12, COLORS["pale_blue"]),
        ("EEGLAB\n0.5-45 Hz\nnotch, ICA", 0.36, COLORS["panel"]),
        ("tACS\n14 sessions\n<30 kOhm", 0.62, COLORS["pale_green"]),
        ("FMA-UE, MBI\npre-treatment\npost-14", 0.88, COLORS["pale_gold"]),
    ]
    for i, (label, x, face) in enumerate(stages):
        box(ax_b, x, 0.54, label, 0.19, 0.33, face, fontsize=6.2)
        if i < len(stages) - 1:
            arrow(ax_b, x + 0.10, 0.54, stages[i + 1][1] - 0.10, 0.54)

    ax_c.set_axis_off()
    panel_label(ax_c, "c")
    ax_c.set_title("Prediction workflow", loc="left")
    flow_boxes = [
        ("Side\nalignment", 0.09, 0.70, COLORS["panel"]),
        ("PSD + WPLI\nEO/EC", 0.31, 0.70, COLORS["pale_blue"]),
        ("Traditional\nML", 0.54, 0.70, COLORS["panel"]),
        ("SSL-CNN", 0.77, 0.70, COLORS["pale_green"]),
        ("Patient-level\nprediction", 0.77, 0.30, COLORS["pale_red"]),
        ("IG + SmoothGrad\nocclusion + MNE", 0.29, 0.30, COLORS["pale_violet"]),
    ]
    for label, x, y, face in flow_boxes:
        width = 0.24 if "SmoothGrad" in label else 0.18
        box(ax_c, x, y, label, width, 0.20, face, fontsize=6.6)
    for x1, y1, x2, y2 in [(0.18, 0.70, 0.22, 0.70), (0.40, 0.70, 0.45, 0.70), (0.63, 0.70, 0.68, 0.70), (0.77, 0.60, 0.77, 0.40), (0.68, 0.30, 0.43, 0.30)]:
        arrow(ax_c, x1, y1, x2, y2)

    base = output_dir / "figure1_overall_framework"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(base, "Overall study, acquisition, label and analysis workflow", "participant flow table and methods specification")


def make_figure2(root: Path, output_dir: Path) -> dict[str, str]:
    flow = pd.read_csv(root / "results" / "tables" / "participant_flow_safety_source_notes.csv")
    counts = dict(zip(flow["category"], flow["n"], strict=False))
    n_labelled = get_count(counts, "Final labeled supervised cohort")
    n_unlabelled = get_count(counts, "EEG-indexed patients not used for supervised labels")

    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.set_axis_off()

    panel_label(ax_a, "a")
    ax_a.set_title("Unlabelled EEG use", loc="left")
    box(ax_a, 0.18, 0.76, f"Labelled EEG\nn={n_labelled}", 0.27, 0.18, COLORS["pale_gold"])
    box(ax_a, 0.18, 0.36, f"SSL-only EEG\nn={n_unlabelled}", 0.27, 0.18, COLORS["pale_green"])
    ssl_sources = [
        "baseline only",
        "baseline + immediate",
        "baseline + immediate + final",
    ]
    for idx, label in enumerate(ssl_sources):
        y = 0.76 - idx * 0.25
        box(ax_a, 0.66, y, label, 0.36, 0.15, COLORS["panel"], fontsize=6.8)
        arrow(ax_a, 0.31, 0.36, 0.48, y)
    ax_a.text(0.60, 0.10, "Used without outcome labels during SSL pretraining", transform=ax_a.transAxes, ha="center", fontsize=6.8)

    panel_label(ax_b, "b")
    ax_b.set_title("Two-view augmentation", loc="left")
    box(ax_b, 0.12, 0.55, "EEG feature tensor\nPSD + WPLI", 0.24, 0.24, COLORS["pale_blue"])
    arrow(ax_b, 0.24, 0.60, 0.40, 0.76)
    arrow(ax_b, 0.24, 0.50, 0.40, 0.32)
    box(ax_b, 0.52, 0.76, "View A\nfeature dropout\nband masking", 0.30, 0.22, COLORS["panel"])
    box(ax_b, 0.52, 0.32, "View B\nnoise jitter\nstate masking", 0.30, 0.22, COLORS["panel"])
    arrow(ax_b, 0.67, 0.76, 0.82, 0.62)
    arrow(ax_b, 0.67, 0.32, 0.82, 0.48)
    box(ax_b, 0.88, 0.55, "shared\nencoder", 0.18, 0.28, COLORS["pale_green"])

    panel_label(ax_c, "c")
    ax_c.set_title("Redundancy reduction", loc="left")
    matrix = np.full((8, 8), 0.22)
    np.fill_diagonal(matrix, 0.92)
    ax_c.imshow(matrix, cmap="Blues", vmin=0, vmax=1, extent=(0.08, 0.48, 0.18, 0.78), transform=ax_c.transAxes)
    ax_c.add_patch(Rectangle((0.08, 0.18), 0.40, 0.60, transform=ax_c.transAxes, fill=False, lw=0.8, ec=COLORS["ink"]))
    ax_c.text(0.28, 0.10, "cross-correlation C", transform=ax_c.transAxes, ha="center", fontsize=7)
    ax_c.text(
        0.72,
        0.54,
        r"$L_{BT}=\sum_i(1-C_{ii})^2+\lambda\sum_{i\ne j}C_{ij}^2$",
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=9,
    )
    ax_c.text(0.72, 0.30, "diagonal invariance\nplus off-diagonal\nredundancy penalty", transform=ax_c.transAxes, ha="center", fontsize=7)

    panel_label(ax_d, "d")
    ax_d.set_title("Downstream transfer", loc="left")
    box(ax_d, 0.16, 0.60, "pretrained\nencoder", 0.25, 0.22, COLORS["pale_green"])
    arrow(ax_d, 0.29, 0.60, 0.42, 0.60)
    box(ax_d, 0.55, 0.60, "LOSO fine-tuning\nn=19 labelled", 0.30, 0.22, COLORS["pale_gold"])
    arrow(ax_d, 0.70, 0.60, 0.77, 0.60)
    box(ax_d, 0.86, 0.60, "proportional\nrecovery", 0.20, 0.22, COLORS["pale_red"])
    ax_d.text(
        0.50,
        0.26,
        "Pretraining expands representation learning beyond patients with complete supervised labels.",
        transform=ax_d.transAxes,
        ha="center",
        fontsize=7,
    )

    base = output_dir / "figure2_ssl_framework"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(base, "Self-supervised learning uses otherwise unusable EEG to enlarge representation learning", "participant flow table and SSL methods")


def make_figure3(root: Path, output_dir: Path) -> dict[str, str]:
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.05, 0.95], hspace=0.32, wspace=0.28)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    for ax in [ax_a, ax_b, ax_c]:
        ax.set_axis_off()

    panel_label(ax_a, "a")
    ax_a.set_title("Multibranch CNN", loc="left")
    branch_specs = [
        ("PSD EO", 0.12, 0.78, COLORS["pale_blue"]),
        ("PSD EC", 0.12, 0.58, COLORS["pale_blue"]),
        ("WPLI EO", 0.12, 0.38, COLORS["pale_violet"]),
        ("WPLI EC", 0.12, 0.18, COLORS["pale_violet"]),
    ]
    for label, x, y, face in branch_specs:
        box(ax_a, x, y, label, 0.18, 0.13, face)
        arrow(ax_a, x + 0.09, y, 0.30, y)
        box(ax_a, 0.38, y, "1D CNN\nencoder", 0.18, 0.13, COLORS["panel"])
        arrow(ax_a, 0.47, y, 0.58, 0.50)
    box(ax_a, 0.66, 0.50, "gated fusion", 0.22, 0.18, COLORS["pale_green"])
    arrow(ax_a, 0.77, 0.50, 0.88, 0.50)
    box(ax_a, 0.90, 0.50, "shared\nembedding z", 0.15, 0.18, COLORS["pale_gold"])

    panel_label(ax_b, "b")
    ax_b.set_title("Residual-aware heads", loc="left")
    box(ax_b, 0.16, 0.55, "embedding z", 0.22, 0.20, COLORS["pale_gold"])
    heads = [
        ("binary head\np(y=1)", 0.67, 0.78, COLORS["pale_red"]),
        ("residual-distance\nhead", 0.67, 0.55, COLORS["panel"]),
        ("pairwise ranking\nhead", 0.67, 0.32, COLORS["panel"]),
    ]
    for label, x, y, face in heads:
        arrow(ax_b, 0.27, 0.55, 0.52, y)
        box(ax_b, x, y, label, 0.34, 0.15, face)
    ax_b.text(0.66, 0.10, "Only the binary head is used at test time.", transform=ax_b.transAxes, ha="center", fontsize=7)

    panel_label(ax_c, "c")
    ax_c.set_title("Training target", loc="left")
    ax_c.text(
        0.50,
        0.68,
        r"$L=L_{cls}+\alpha L_{res}+\beta L_{rank}+\gamma L_{soft}$",
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )
    box(ax_c, 0.22, 0.34, "signed residual\nfrom proportional\nrecovery", 0.34, 0.25, COLORS["panel"])
    box(ax_c, 0.65, 0.34, "fold-local train\nthresholding\ninside LOSO", 0.34, 0.25, COLORS["pale_green"])
    arrow(ax_c, 0.39, 0.34, 0.48, 0.34)
    ax_c.text(0.50, 0.08, "Auxiliary objectives regularize a scarce supervised cohort.", transform=ax_c.transAxes, ha="center", fontsize=7)

    base = output_dir / "figure3_cnn_residual_aware"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(base, "CNN architecture separates modality/state branches and residual-aware training heads", "model architecture and training objective")


def make_figure4(root: Path, output_dir: Path) -> dict[str, str]:
    perf = pd.read_csv(root / "results" / "tables" / "table2_main_model_performance.csv")
    ci = pd.read_csv(root / "results" / "statistics" / "model_metric_confidence_intervals.csv")
    pred = pd.read_csv(root / "results" / "predictions" / "paper_locked_model_predictions.csv")
    perf = perf[perf["model_name"].isin(PRIMARY_MODELS)].copy()
    perf["order"] = perf["model_name"].map({name: idx for idx, name in enumerate(PRIMARY_MODELS)})
    perf = perf.sort_values("order")

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    panel_label(ax_a, "a")
    plot_metric_bars(ax_a, perf)

    panel_label(ax_b, "b")
    plot_roc(ax_b, pred)

    panel_label(ax_c, "c")
    plot_pr(ax_c, pred)

    panel_label(ax_d, "d")
    plot_confusion(ax_d, pred, "residual_aware_SSL_CNN_seedmean10")

    panel_label(ax_e, "e")
    plot_calibration(ax_e, pred)

    panel_label(ax_f, "f")
    plot_ci(ax_f, ci)

    base = output_dir / "figure4_model_performance"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(base, "Residual-aware SSL-CNN improves discrimination and reports conservative uncertainty", "main performance, locked predictions and bootstrap intervals")


def make_figure5(root: Path, output_dir: Path) -> dict[str, str]:
    core = pd.read_csv(root / "results" / "metrics" / "core_ablation_10seed_summary.csv")
    mod = pd.read_csv(root / "results" / "tables" / "table3_ablation.csv")
    core = core[core["availability"].isin(["available"])]

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_label(ax_a, "a")
    plot_core_accuracy(ax_a, core)

    panel_label(ax_b, "b")
    plot_core_auc(ax_b, core)

    panel_label(ax_c, "c")
    plot_feature_state_ablation(ax_c, mod)

    panel_label(ax_d, "d")
    plot_band_ablation(ax_d, mod)

    base = output_dir / "figure5_stability_ablation"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(base, "Stability and ablation analyses show the contribution of CNN, residual-aware heads and EEG feature blocks", "core ablation and modality/state/band ablation tables")


def make_figure6(root: Path, output_dir: Path) -> dict[str, str]:
    occlusion = pd.read_csv(root / "results" / "explainability" / "occlusion_branch_state.csv")
    psd = pd.read_csv(root / "results" / "explainability" / "psd_channel_band_importance.csv")
    wpli = pd.read_csv(root / "results" / "explainability" / "wpli_top_edges.csv")

    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.05], hspace=0.38, wspace=0.56)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    panel_label(ax_a, "a")
    plot_occlusion(ax_a, occlusion)

    panel_label(ax_b, "b")
    plot_psd_heatmap(ax_b, psd)

    panel_label(ax_c, "c")
    plot_wpli_edges(ax_c, wpli)

    panel_label(ax_d, "d")
    show_image_panel(
        ax_d,
        root / "results" / "figures" / "explainability" / "mne_topomaps" / "mne_psd_eo_beta_high_signed_attribution_topomap.png",
        "MNE PSD topomap\nEO beta high",
    )

    panel_label(ax_e, "e")
    show_image_panel(
        ax_e,
        root / "results" / "figures" / "explainability" / "mne_wpli_connectivity" / "mne_wpli_ec_beta_high_connectivity_top20.png",
        "MNE WPLI connectivity\nEC beta high",
    )

    panel_label(ax_f, "f")
    show_image_panel(
        ax_f,
        root / "results" / "figures" / "explainability" / "paper_attribution_stability.png",
        "Attribution stability",
    )

    base = output_dir / "figure6_eeg_explainability"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(base, "Model explanations localize state, spectral and connectivity contributions as hypothesis-generating evidence", "occlusion, attribution and MNE-rendered explainability outputs")


def plot_metric_bars(ax: plt.Axes, perf: pd.DataFrame) -> None:
    metrics = ["accuracy", "roc_auc", "pr_auc"]
    labels = ["Accuracy", "ROC-AUC", "PR-AUC"]
    colors = [COLORS["blue"], COLORS["red"], COLORS["gold"]]
    x = np.arange(len(perf))
    width = 0.23
    for idx, (metric, label, color) in enumerate(zip(metrics, labels, colors, strict=True)):
        ax.bar(x + (idx - 1) * width, perf[metric].to_numpy(float), width=width, color=color, label=label)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[name] for name in perf["model_name"]], rotation=35, ha="right")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5)


def plot_roc(ax: plt.Axes, pred: pd.DataFrame) -> None:
    curve_colors = [COLORS["muted"], COLORS["blue"], COLORS["red"]]
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#222222")
    for model, color in zip(CURVE_MODELS, curve_colors, strict=True):
        group = subject_predictions(pred, model)
        if group.empty:
            continue
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_score"])
        ax.plot(fpr, tpr, lw=1.2, color=color, label=f"{CURVE_LABELS[model]} ({auc(fpr, tpr):.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("ROC")
    ax.legend(frameon=False, loc="lower right")


def plot_pr(ax: plt.Axes, pred: pd.DataFrame) -> None:
    curve_colors = [COLORS["muted"], COLORS["blue"], COLORS["red"]]
    baseline_group = subject_predictions(pred, CURVE_MODELS[-1])
    prevalence = float(baseline_group["y_true"].mean()) if not baseline_group.empty else 0.5
    ax.axhline(prevalence, ls="--", lw=0.8, color="#222222")
    for model, color in zip(CURVE_MODELS, curve_colors, strict=True):
        group = subject_predictions(pred, model)
        if group.empty:
            continue
        precision, recall, _ = precision_recall_curve(group["y_true"], group["y_score"])
        ax.plot(recall, precision, lw=1.2, color=color, label=f"{CURVE_LABELS[model]} ({auc(recall, precision):.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall")
    ax.legend(frameon=False, loc="lower left")


def plot_confusion(ax: plt.Axes, pred: pd.DataFrame, model_name: str) -> None:
    group = subject_predictions(pred, model_name)
    tn = int(((group["y_true"] == 0) & (group["y_pred"] == 0)).sum())
    fp = int(((group["y_true"] == 0) & (group["y_pred"] == 1)).sum())
    fn = int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum())
    tp = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).sum())
    mat = np.array([[tn, fp], [fn, tp]])
    ax.imshow(mat, cmap="Reds", vmin=0, vmax=max(1, int(mat.max())))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Poor", "Prop."])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Poor", "Prop."])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("Final confusion matrix")
    for i in range(2):
        for j in range(2):
            color = "white" if mat[i, j] > mat.max() * 0.55 else COLORS["ink"]
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)


def plot_calibration(ax: plt.Axes, pred: pd.DataFrame) -> None:
    curve_colors = [COLORS["muted"], COLORS["blue"], COLORS["red"]]
    ax.plot([0, 1], [0, 1], ls="--", color="#222222", lw=0.8)
    for model, color in zip(CURVE_MODELS, curve_colors, strict=True):
        group = subject_predictions(pred, model)
        if group.empty:
            continue
        bins = np.linspace(0, 1, 6)
        xs: list[float] = []
        ys: list[float] = []
        for low, high in zip(bins[:-1], bins[1:], strict=True):
            mask = (group["y_score"] >= low) & (group["y_score"] <= high if high == 1 else group["y_score"] < high)
            if np.any(mask):
                xs.append(float(group.loc[mask, "y_score"].mean()))
                ys.append(float(group.loc[mask, "y_true"].mean()))
        ax.plot(xs, ys, marker="o", ms=3, lw=1.1, color=color, label=CURVE_LABELS[model])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_title("Calibration")
    ax.legend(frameon=False, loc="lower right")


def plot_ci(ax: plt.Axes, ci: pd.DataFrame) -> None:
    rows = ci[ci["model_name"].isin(CURVE_MODELS)].copy()
    rows["order"] = rows["model_name"].map({name: idx for idx, name in enumerate(CURVE_MODELS)})
    rows = rows.sort_values("order")
    y = np.arange(len(rows))
    auc_values = rows["roc_auc"].to_numpy(float)
    low = rows["roc_auc_low_ci"].to_numpy(float)
    high = rows["roc_auc_high_ci"].to_numpy(float)
    ax.errorbar(auc_values, y, xerr=np.vstack([auc_values - low, high - auc_values]), fmt="o", color=COLORS["ink"], ecolor=COLORS["muted"], capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([CURVE_LABELS[name] for name in rows["model_name"]])
    ax.set_xlim(0.35, 1.02)
    ax.set_xlabel("ROC-AUC, bootstrap 95% CI")
    ax.set_title("Subject-level uncertainty")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5)


def plot_core_accuracy(ax: plt.Axes, core: pd.DataFrame) -> None:
    keys = [
        "a_ml_psd_wpli_baseline",
        "b_no_ssl_cnn_same_arch_10seed",
        "c_patient_barlow_ssl_no_residual_heads",
        "d_no_ssl_cnn_residual_heads",
        "e_patient_barlow_ssl_residual_heads",
    ]
    labels = ["ML\nPSD+WPLI", "CNN\nno SSL", "Barlow SSL\nno residual", "Residual heads\nno SSL", "Final\nSSL-CNN"]
    means = value_by_row(core, keys, "mean", "reported", "accuracy")
    stds = value_by_row(core, keys, "std", None, "accuracy")
    mins = value_by_row(core, keys, "min", None, "accuracy")
    x = np.arange(len(keys))
    ax.bar(x, means, color=[COLORS["muted"], COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["red"]])
    for xi, mean, std in zip(x, means, stds, strict=True):
        if not math.isnan(std):
            ax.errorbar(xi, mean, yerr=std, fmt="none", ecolor=COLORS["ink"], capsize=3, lw=0.8)
    for xi, mn in zip(x, mins, strict=True):
        if not math.isnan(mn):
            ax.scatter([xi], [mn], marker="v", color=COLORS["ink"], s=18, zorder=3)
    ax.set_ylim(0.55, 0.92)
    ax.set_ylabel("Accuracy")
    ax.set_title("Ten-seed accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
    ax.text(0.02, 0.04, "error bars: SD; triangles: minimum", transform=ax.transAxes, fontsize=6)


def plot_core_auc(ax: plt.Axes, core: pd.DataFrame) -> None:
    keys = [
        "a_ml_psd_wpli_baseline",
        "b_no_ssl_cnn_same_arch_10seed",
        "c_patient_barlow_ssl_no_residual_heads",
        "d_no_ssl_cnn_residual_heads",
        "e_patient_barlow_ssl_residual_heads",
    ]
    labels = ["ML", "CNN", "SSL", "Residual", "Final"]
    roc = value_by_row(core, keys, "mean", "reported", "roc_auc")
    pr = value_by_row(core, keys, "mean", "reported", "pr_auc")
    x = np.arange(len(keys))
    ax.plot(x, roc, marker="o", color=COLORS["blue"], lw=1.2, label="ROC-AUC")
    ax.plot(x, pr, marker="s", color=COLORS["red"], lw=1.2, label="PR-AUC")
    ax.set_ylim(0.65, 0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Discrimination ablation")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
    ax.legend(frameon=False, loc="lower right")


def plot_feature_state_ablation(ax: plt.Axes, mod: pd.DataFrame) -> None:
    selection = {
        "psd_only": "PSD only",
        "wpli_only": "WPLI only",
        "psd_wpli": "PSD+WPLI",
        "eo_only": "EO only",
        "ec_only": "EC only",
        "motor_wpli_edges": "Motor WPLI",
    }
    rows = mod[mod["ablation_name"].isin(selection)].copy()
    rows["label"] = rows["ablation_name"].map(selection)
    rows = rows.set_index("ablation_name").loc[list(selection)].reset_index()
    x = np.arange(len(rows))
    ax.bar(x, rows["accuracy"], color=[COLORS["blue"], COLORS["violet"], COLORS["red"], COLORS["gold"], COLORS["green"], COLORS["muted"]])
    ax.set_ylim(0.25, 0.86)
    ax.set_ylabel("Accuracy")
    ax.set_title("Feature and state ablation")
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"], rotation=35, ha="right")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5)


def plot_band_ablation(ax: plt.Axes, mod: pd.DataFrame) -> None:
    selection = {
        "beta_medium_only": "Beta medium",
        "beta_high_only": "Beta high",
        "beta_medium_beta_high": "Beta med+high",
        "psd_eo_only": "PSD EO",
        "psd_eo_wpli_ec": "PSD EO+\nWPLI EC",
    }
    rows = mod[mod["ablation_name"].isin(selection)].copy()
    rows["label"] = rows["ablation_name"].map(selection)
    rows = rows.set_index("ablation_name").loc[list(selection)].reset_index()
    x = np.arange(len(rows))
    ax.bar(x, rows["roc_auc"], color=COLORS["teal"], alpha=0.85, label="ROC-AUC")
    ax.plot(x, rows["pr_auc"], marker="o", color=COLORS["red"], lw=1.2, label="PR-AUC")
    ax.set_ylim(0.45, 0.88)
    ax.set_ylabel("Score")
    ax.set_title("Band-specific ablation")
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"], rotation=35, ha="right")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
    ax.legend(frameon=False, loc="lower right")


def plot_occlusion(ax: plt.Axes, occlusion: pd.DataFrame) -> None:
    grouped = occlusion.groupby("group_name", as_index=False).agg(delta_loss=("delta_loss", "mean"))
    grouped = grouped.sort_values("delta_loss", ascending=True)
    colors = [COLORS["blue"] if "PSD" in name or "EO" in name else COLORS["violet"] for name in grouped["group_name"]]
    ax.barh(grouped["group_name"], grouped["delta_loss"], color=colors)
    ax.axvline(0, color=COLORS["ink"], lw=0.8)
    ax.set_xlabel("Mean loss increase after occlusion")
    ax.set_title("Branch/state occlusion")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5)


def plot_psd_heatmap(ax: plt.Axes, psd: pd.DataFrame) -> None:
    channel_order = (
        psd.groupby("channel")["mean_abs_attribution"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    band_order = ["Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High", "Gamma"]
    pivot = (
        psd[psd["channel"].isin(channel_order)]
        .pivot_table(index="channel", columns="band", values="mean_abs_attribution", aggfunc="mean")
        .reindex(index=channel_order, columns=band_order)
    )
    values = pivot.to_numpy(float)
    denom = np.nanmax(values)
    if denom > 0:
        values = values / denom
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(band_order)))
    ax.set_xticklabels([wrap_label(label, 8) for label in band_order], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(channel_order)))
    ax.set_yticklabels(channel_order)
    ax.set_title("PSD attribution")
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6)


def plot_wpli_edges(ax: plt.Axes, wpli: pd.DataFrame) -> None:
    rows = wpli.sort_values("mean_abs_attribution", ascending=False).head(6).copy()
    rows["edge"] = rows["state"] + " " + rows["channel_i"] + "-" + rows["channel_j"]
    rows = rows.iloc[::-1]
    colors = [COLORS["red"] if value >= 0 else COLORS["blue"] for value in rows["mean_signed_attribution"]]
    ax.barh(rows["edge"], rows["mean_abs_attribution"], color=colors)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlabel("Mean abs attribution")
    ax.set_title("Top WPLI edges")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5)


def show_image_panel(ax: plt.Axes, path: Path, title: str) -> None:
    ax.set_axis_off()
    if path.exists():
        image = mpimg.imread(path)
        ax.imshow(image)
    else:
        ax.text(0.5, 0.5, f"Missing image\n{path.name}", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title, loc="left")


def subject_predictions(pred: pd.DataFrame, model_name: str) -> pd.DataFrame:
    group = pred[pred["model_name"] == model_name].copy()
    if group.empty:
        return group
    return group.groupby("subject_id", as_index=False).agg(
        y_true=("y_true", "first"),
        y_score=("y_score", "mean"),
        y_pred=("y_pred", lambda value: int(np.mean(value) >= 0.5)),
    )


def value_by_row(core: pd.DataFrame, keys: list[str], row_type: str, fallback_row_type: str | None, column: str) -> np.ndarray:
    values: list[float] = []
    for key in keys:
        row = core[(core["contrast"] == key) & (core["row_type"] == row_type)]
        if row.empty and fallback_row_type:
            row = core[(core["contrast"] == key) & (core["row_type"] == fallback_row_type)]
        if row.empty or pd.isna(row[column].iloc[0]):
            values.append(float("nan"))
        else:
            values.append(float(row[column].iloc[0]))
    return np.array(values, dtype=float)


def box(ax: plt.Axes, x: float, y: float, text: str, width: float, height: float, face: str, *, fontsize: float = 7.0) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        edgecolor=COLORS["ink"],
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center", fontsize=fontsize)


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.8,
        color=COLORS["ink"],
        shrinkA=1,
        shrinkB=1,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.07, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def get_count(counts: dict[str, object], key: str) -> int:
    return int(counts.get(key, 0))


def wrap_label(label: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width))


def save_pub(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def manifest(base: Path, conclusion: str, source_data: str) -> dict[str, str]:
    return {
        "figure": base.name,
        "conclusion": conclusion,
        "source_data": source_data,
        "png": str(base.with_suffix(".png")),
        "svg": str(base.with_suffix(".svg")),
        "pdf": str(base.with_suffix(".pdf")),
        "tiff": str(base.with_suffix(".tiff")),
    }


def write_contact_sheet(output_dir: Path, rows: list[dict[str, str]]) -> None:
    images: list[tuple[str, Image.Image]] = []
    for row in rows:
        path = Path(row["png"])
        if not path.exists():
            continue
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            image.thumbnail((900, 620), Image.LANCZOS)
            images.append((row["figure"], image.copy()))
    if not images:
        return
    cell_w, cell_h = 960, 690
    columns = 2
    rows_needed = int(math.ceil(len(images) / columns))
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows_needed), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        col = index % columns
        row = index // columns
        x = col * cell_w + (cell_w - image.width) // 2
        y = row * cell_h + 44
        draw.text((col * cell_w + 24, row * cell_h + 14), label, fill=(32, 40, 45))
        sheet.paste(image, (x, y))
    sheet.save(output_dir / "revised_initial_figures_contact_sheet.png")


if __name__ == "__main__":
    main()
