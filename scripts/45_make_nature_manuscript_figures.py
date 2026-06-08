from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_LABELS = {
    "ML_EEG_updated_no_selector_logistic_l1": "Logistic EEG",
    "no_SSL_CNN_updated_sub05_sub28_seedensemble10": "CNN",
    "residual_aware_SSL_CNN_seedmean10": "Residual-aware\nSSL-CNN",
}

COLORS = {
    "ml": "#6B7280",
    "cnn": "#2F6C99",
    "ssl": "#B84A39",
    "clinical": "#5B8C62",
    "accent": "#D7A441",
    "light": "#F5F7F8",
    "line": "#2F3A3F",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build publication-oriented manuscript figures from locked paper outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default=PROJECT_ROOT)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = root / "results" / "figures" / "nature"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    manifest_rows = []
    manifest_rows.append(make_figure1(root, output_dir))
    manifest_rows.append(make_figure2(root, output_dir))
    manifest_rows.append(make_figure3(root, output_dir))
    manifest_rows.append(make_figure4(root, output_dir))
    manifest_rows.append(make_supplementary_error_figure(root, output_dir))
    manifest_rows.append(make_supplementary_performance_precision_figure(root, output_dir))
    manifest_rows.append(make_supplementary_clinical_incremental_figure(root, output_dir))
    pd.DataFrame(manifest_rows).to_csv(output_dir / "figure_manifest.csv", index=False)
    write_contact_sheet(output_dir, manifest_rows)
    print(f"Wrote publication figure bundle to {output_dir}")


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
    table1 = pd.read_csv(root / "results" / "tables" / "table1_cohort_characteristics.csv")
    flow = pd.read_csv(root / "results" / "tables" / "participant_flow_safety_source_notes.csv")
    n_positive = parse_count(table1.loc[table1["variable"] == "label=1", "All"].iloc[0])
    n_negative = parse_count(table1.loc[table1["variable"] == "label=0", "All"].iloc[0])
    n_total = n_positive + n_negative
    flow_counts = dict(zip(flow["category"], flow["n"], strict=False))

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.45], height_ratios=[0.85, 1.15], hspace=0.44, wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_label(ax_a, "a")
    ax_a.set_axis_off()
    ax_a.set_xlim(0, n_total)
    ax_a.set_ylim(0, 1)
    add_box(
        ax_a,
        0.50,
        0.84,
        f"M1 patient records\nn={flow_counts['M1 patient records in clinical source workbook']}",
        width=0.78,
        height=0.19,
        face="#EEF2F4",
    )
    add_arrow(ax_a, 0.50, 0.74, 0.50, 0.66)
    add_box(
        ax_a,
        0.50,
        0.57,
        "EEG-indexed pool\n"
        f"n={flow_counts['Current EEG-indexed M1 patient pool']} "
        f"({flow_counts['Clinical workbook entries without current indexed EEG']} not indexed)",
        width=0.78,
        height=0.19,
        face="#EAF1F3",
    )
    add_arrow(ax_a, 0.50, 0.47, 0.50, 0.39)
    add_box(
        ax_a,
        0.50,
        0.30,
        f"Supervised LOSO\nn={n_total} "
        f"({flow_counts['EEG-indexed patients not used for supervised labels']} outside labels)",
        width=0.78,
        height=0.19,
        face="#FAF3DF",
    )
    ax_a.barh([0.08], [n_positive], color=COLORS["ssl"], height=0.08)
    ax_a.barh([0.08], [n_negative], left=[n_positive], color=COLORS["ml"], height=0.08)
    ax_a.text(n_positive / 2, 0.08, f"{n_positive} proportional", ha="center", va="center", color="white", fontsize=6)
    ax_a.text(n_positive + n_negative / 2, 0.08, f"{n_negative} poor", ha="center", va="center", color="white", fontsize=6)
    ax_a.set_title("Participant flow and endpoint")

    panel_label(ax_b, "b")
    ax_b.set_axis_off()
    workflow = [
        ("Baseline\nresting EEG", 0.08),
        ("PSD + WPLI\nEO/EC", 0.29),
        ("Patient-level\nLOSO", 0.50),
        ("ML, CNN,\nSSL-CNN", 0.70),
        ("Statistics +\nbiomarkers", 0.91),
    ]
    for idx, (label, xpos) in enumerate(workflow):
        add_box(ax_b, xpos, 0.54, label, width=0.15, height=0.22, face=COLORS["light"])
        if idx < len(workflow) - 1:
            add_arrow(ax_b, xpos + 0.075, 0.54, workflow[idx + 1][1] - 0.075, 0.54)
    ax_b.set_title("Analysis workflow")

    panel_label(ax_c, "c")
    ax_c.set_axis_off()
    branch_labels = [
        ("PSD EO", 0.08, 0.72),
        ("PSD EC", 0.08, 0.48),
        ("WPLI EO", 0.08, 0.24),
        ("WPLI EC", 0.08, 0.00),
    ]
    for label, x, y in branch_labels:
        add_box(ax_c, x, y + 0.16, label, width=0.12, height=0.14, face="#EAF1F3")
        add_arrow(ax_c, x + 0.06, y + 0.16, 0.30, 0.50)
    add_box(ax_c, 0.37, 0.50, "CNN encoders\n+ gated fusion", width=0.16, height=0.20, face="#EEF4EA")
    add_arrow(ax_c, 0.45, 0.50, 0.57, 0.50)
    add_box(ax_c, 0.64, 0.50, "32-d embedding", width=0.13, height=0.16, face="#FAF3DF")
    for y, label in [(0.72, "binary\nclassification"), (0.50, "fold-local\nsigned residual"), (0.28, "pairwise\nranking")]:
        add_arrow(ax_c, 0.705, 0.50, 0.80, y)
        add_box(ax_c, 0.87, y, label, width=0.15, height=0.14, face="#F2E7E4")
    ax_c.text(0.86, 0.08, "Inference uses the classification head;\nresidual/ranking heads regularize training.", ha="center", va="center", fontsize=7)
    ax_c.set_title("Residual-aware multimodal SSL-CNN")

    base = output_dir / "figure1_study_design_model"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(base, "Participant flow, study design, and residual-aware SSL-CNN architecture", "participant-flow source data, labels, methods schematic")


def make_figure2(root: Path, output_dir: Path) -> dict[str, str]:
    perf = pd.read_csv(root / "results" / "tables" / "table2_main_model_performance.csv")
    ci = pd.read_csv(root / "results" / "statistics" / "model_metric_confidence_intervals.csv")
    predictions = pd.read_csv(root / "results" / "predictions" / "paper_locked_model_predictions.csv")
    primary_names = list(MODEL_LABELS)
    perf = perf[perf["model_name"].isin(primary_names)].copy()
    ci = ci[ci["model_name"].isin(primary_names)].copy()

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_label(ax_a, "a")
    metrics = ["accuracy", "roc_auc", "pr_auc"]
    x = np.arange(len(perf))
    width = 0.23
    metric_colors = ["#4D7895", "#B84A39", "#D7A441"]
    for offset, (metric, color) in enumerate(zip(metrics, metric_colors, strict=True)):
        ax_a.bar(x + (offset - 1) * width, perf[metric], width=width, color=color, label=metric.replace("_", "-").upper())
    ax_a.set_ylim(0, 1.02)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([MODEL_LABELS[name] for name in perf["model_name"]])
    ax_a.set_ylabel("Score")
    ax_a.set_title("Primary subject-level performance")
    ax_a.legend(frameon=False, ncol=1, loc="upper left")

    panel_label(ax_b, "b")
    ordered = ci.set_index("model_name").loc[primary_names].reset_index()
    x = np.arange(len(ordered))
    y = ordered["roc_auc"].to_numpy(float)
    yerr = np.vstack([y - ordered["roc_auc_low_ci"], ordered["roc_auc_high_ci"] - y])
    ax_b.errorbar(x, y, yerr=yerr, fmt="o", color=COLORS["line"], capsize=3)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([MODEL_LABELS[name] for name in ordered["model_name"]])
    ax_b.set_ylim(0.35, 1.02)
    ax_b.set_ylabel("ROC-AUC, bootstrap 95% CI")
    ax_b.set_title("Uncertainty over subjects")

    panel_label(ax_c, "c")
    brier = perf.set_index("model_name").loc[primary_names, "brier_score"]
    ax_c.bar(range(len(brier)), brier.values, color=[COLORS["ml"], COLORS["cnn"], COLORS["ssl"]])
    ax_c.set_xticks(range(len(brier)))
    ax_c.set_xticklabels([MODEL_LABELS[name] for name in brier.index])
    ax_c.set_ylim(0, max(0.25, float(brier.max()) * 1.25))
    ax_c.set_ylabel("Brier score (lower is better)")
    ax_c.set_title("Calibration-oriented score")
    for idx, value in enumerate(brier.values):
        ax_c.text(idx, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=6)

    panel_label(ax_d, "d")
    plot_calibration(ax_d, predictions[predictions["model_name"].isin(primary_names)])
    ax_d.set_title("Reliability curves")

    base = output_dir / "figure2_performance_calibration"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(base, "Primary performance, uncertainty, and calibration", "locked predictions and statistics")


def make_figure3(root: Path, output_dir: Path) -> dict[str, str]:
    seed = pd.read_csv(root / "results" / "tables" / "seed_stability_table.csv")
    table3 = pd.read_csv(root / "results" / "tables" / "table3_ablation.csv")
    sensitivity = pd.read_csv(root / "results" / "metrics" / "residual_threshold_sensitivity.csv")

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_label(ax_a, "a")
    x = np.arange(len(seed))
    ax_a.bar(x, seed["mean_accuracy"], color=["#6B7280", "#839A7C", COLORS["ssl"]], alpha=0.92)
    ax_a.errorbar(x, seed["mean_accuracy"], yerr=seed["std_accuracy"], fmt="none", ecolor="black", capsize=3, lw=0.8)
    ax_a.scatter(x, seed["min_accuracy"], color="black", marker="D", s=16, label="minimum seed")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["CNN", "Barlow SSL", "Residual-aware\nSSL-CNN"])
    ax_a.set_ylim(0.45, 0.95)
    ax_a.set_ylabel("Accuracy across 10 seeds")
    ax_a.set_title("Seed stability")
    ax_a.legend(frameon=False, loc="lower right")

    panel_label(ax_b, "b")
    core = table3[(table3["ablation_block"] == "core") & (table3["row_type"].isin(["mean", "seedmean10", "ensemble10", "reported"]))].copy()
    core = core.dropna(subset=["roc_auc"])
    priority = {"reported": 0, "seedmean10": 1, "ensemble10": 2, "mean": 3}
    core["row_priority"] = core["row_type"].map(priority).fillna(9)
    core = core.sort_values(["ablation_name", "row_priority"]).groupby("ablation_name", as_index=False).head(1)
    labels = [shorten_ablation(v) for v in core["ablation_name"]]
    y = np.arange(len(core))
    ax_b.barh(y, core["roc_auc"], color="#4D7895")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels)
    ax_b.set_xlim(0.55, 0.96)
    ax_b.set_xlabel("ROC-AUC")
    ax_b.set_title("Core ablation")

    panel_label(ax_c, "c")
    modality = table3[table3["ablation_block"] == "modality_state_band"].dropna(subset=["roc_auc"]).copy()
    keep = ["psd_only", "wpli_only", "eo_only", "ec_only", "beta_medium_only", "motor_wpli_edges", "psd_wpli"]
    modality = modality[modality["ablation_name"].isin(keep)]
    y = np.arange(len(modality))
    ax_c.barh(y, modality["roc_auc"], color="#5B8C62")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([v.replace("_", " ") for v in modality["ablation_name"]])
    ax_c.set_xlim(0.2, 0.9)
    ax_c.set_xlabel("ROC-AUC")
    ax_c.set_title("Modality, state, and band probes")

    panel_label(ax_d, "d")
    model_column = "model_name" if "model_name" in sensitivity.columns else "model"
    final = sensitivity[sensitivity[model_column].astype(str).str.contains("residual_aware", case=False, na=False)].copy()
    if "threshold_type" in final.columns:
        final = final[final["threshold_type"] == "fixed"]
    if {"residual_threshold", "roc_auc"}.issubset(final.columns) and not final.empty:
        ax_d.plot(final["residual_threshold"], final["roc_auc"], marker="o", color=COLORS["ssl"], label="ROC-AUC")
        ax_d.plot(final["residual_threshold"], final["pr_auc"], marker="s", color=COLORS["accent"], label="PR-AUC")
        ax_d.set_xlabel("Residual threshold")
        ax_d.set_ylabel("Score")
        ax_d.set_ylim(0, 1)
        ax_d.legend(frameon=False)
    else:
        ax_d.text(0.5, 0.5, "Threshold sensitivity table unavailable", ha="center", va="center")
        ax_d.set_axis_off()
    ax_d.set_title("Outcome-threshold sensitivity")

    base = output_dir / "figure3_robustness_ablation"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(base, "Seed stability, ablation, and threshold sensitivity", "seed, ablation, and sensitivity tables")


def make_figure4(root: Path, output_dir: Path) -> dict[str, str]:
    figure_root = root / "results" / "figures"
    panels = [
        ("a", "PSD attribution topomaps", figure_root / "paper" / "figure4_psd_attribution_topomaps.png"),
        ("b", "EC beta-medium WPLI connectome", figure_root / "paper" / "figure5_wpli_beta_connectome.png"),
        ("c", "Branch/state occlusion", figure_root / "paper" / "figure6_branch_state_occlusion.png"),
        ("d", "Attribution stability", figure_root / "explainability" / "paper_attribution_stability.png"),
    ]
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.12)
    for idx, (label, title, path) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        panel_label(ax, label)
        ax.set_title(title, pad=2)
        ax.set_axis_off()
        if path.exists():
            image = mpimg.imread(path)
            ax.imshow(image)
        else:
            ax.text(0.5, 0.5, f"Missing:\n{path.name}", ha="center", va="center")
    base = output_dir / "figure4_explainability_neurophysiology"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(base, "Model explanation and EEG biomarker localization", "explainability images and topomaps")


def make_supplementary_error_figure(root: Path, output_dir: Path) -> dict[str, str]:
    paths = [
        ("a", "Repeated error probabilities", root / "results" / "figures" / "paper" / "error_subject_probability_distribution.png"),
        ("b", "Feature outlier profile", root / "results" / "figures" / "paper" / "error_subject_feature_outlier_heatmap.png"),
    ]
    fig = plt.figure(figsize=(7.2, 3.2))
    gs = fig.add_gridspec(1, 2, wspace=0.16)
    for idx, (label, title, path) in enumerate(paths):
        ax = fig.add_subplot(gs[0, idx])
        panel_label(ax, label)
        ax.set_title(title, pad=2)
        ax.set_axis_off()
        if path.exists():
            ax.imshow(mpimg.imread(path))
    base = output_dir / "supplementary_error_subjects"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(base, "Post-hoc repeated-error subject analysis", "error analysis figures")


def make_supplementary_performance_precision_figure(root: Path, output_dir: Path) -> dict[str, str]:
    ci = pd.read_csv(root / "results" / "statistics" / "model_metric_confidence_intervals.csv")
    pairwise = pd.read_csv(root / "results" / "statistics" / "model_pairwise_comparisons.csv")
    precision_audit = pd.read_csv(root / "results" / "tables" / "performance_precision_audit.csv")
    final = ci.loc[ci["model_name"] == "residual_aware_SSL_CNN_seedmean10"].iloc[0]

    fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82], width_ratios=[1.15, 1.0], hspace=0.46, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_label(ax_a, "a")
    metric_rows = [
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Brier score", "brier_score"),
    ]
    y = np.arange(len(metric_rows))[::-1]
    values = np.array([final[key] for _, key in metric_rows], dtype=float)
    low = np.array([final[f"{key}_low_ci"] for _, key in metric_rows], dtype=float)
    high = np.array([final[f"{key}_high_ci"] for _, key in metric_rows], dtype=float)
    xerr = np.vstack([values - low, high - values])
    colors = [COLORS["ssl"]] * 4 + [COLORS["accent"]]
    ax_a.errorbar(values, y, xerr=xerr, fmt="none", ecolor="#4B5563", elinewidth=1.0, capsize=3, zorder=1)
    ax_a.scatter(values, y, s=32, color=colors, edgecolor="white", linewidth=0.6, zorder=2)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([label for label, _ in metric_rows])
    ax_a.set_xlim(0, 1.04)
    ax_a.set_xlabel("Metric value with bootstrap interval")
    ax_a.set_title("Final model precision")
    ax_a.grid(axis="x", color="#E5E7EB", lw=0.6)

    panel_label(ax_b, "b")
    comparisons = []
    metric_short = {
        "accuracy": "accuracy",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "brier_score": "Brier",
    }
    for reference, display in [
        ("no_SSL_CNN_updated_sub05_sub28_seedensemble10", "CNN"),
        ("ML_EEG_updated_no_selector_logistic_l1", "Logistic EEG"),
    ]:
        for metric in ["accuracy", "roc_auc", "pr_auc", "brier_score"]:
            row = pairwise[
                (pairwise["model_a"] == reference)
                & (pairwise["model_b"] == "residual_aware_SSL_CNN_seedmean10")
                & (pairwise["metric"] == metric)
            ].iloc[0]
            comparisons.append(
                {
                    "label": f"{display}, {metric_short[metric]}",
                    "difference": float(row["difference"]),
                    "low": float(row["ci_low"]),
                    "high": float(row["ci_high"]),
                    "metric": metric,
                }
            )
    yb = np.arange(len(comparisons))[::-1]
    diffs = np.array([row["difference"] for row in comparisons])
    lows = np.array([row["low"] for row in comparisons])
    highs = np.array([row["high"] for row in comparisons])
    ax_b.axvline(0, color="#111827", lw=0.8, ls="--")
    ax_b.errorbar(diffs, yb, xerr=np.vstack([diffs - lows, highs - diffs]), fmt="none", ecolor="#4B5563", elinewidth=0.9, capsize=2.5)
    point_colors = [COLORS["accent"] if row["metric"] == "brier_score" else COLORS["ssl"] for row in comparisons]
    ax_b.scatter(diffs, yb, s=24, color=point_colors, edgecolor="white", linewidth=0.5, zorder=2)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(["\n".join(textwrap.wrap(row["label"], width=18)) for row in comparisons])
    ax_b.set_xlabel("Difference (final model - reference)")
    ax_b.set_title("Paired comparison boundaries")
    ax_b.grid(axis="x", color="#E5E7EB", lw=0.6)

    panel_label(ax_c, "c")
    resolution_items = [
        ("Accuracy\none subject", "Accuracy one-subject resolution"),
        ("Sensitivity\none positive case", "Sensitivity one-case resolution"),
        ("Specificity\none negative case", "Specificity one-case resolution"),
    ]
    values_c = []
    for _, item in resolution_items:
        row = precision_audit.loc[precision_audit["item"] == item].iloc[0]
        values_c.append(float(row["observed_value"]))
    bars = ax_c.bar(np.arange(len(values_c)), np.array(values_c) * 100, color=["#718096", COLORS["ssl"], COLORS["cnn"]], width=0.55)
    ax_c.set_xticks(np.arange(len(values_c)))
    ax_c.set_xticklabels([label for label, _ in resolution_items])
    ax_c.set_ylabel("Metric movement per single case (%)")
    ax_c.set_ylim(0, max(values_c) * 100 + 4)
    ax_c.set_title("Coarse operating-characteristic resolution")
    ax_c.grid(axis="y", color="#E5E7EB", lw=0.6)
    for bar, value in zip(bars, values_c, strict=True):
        ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{value * 100:.1f}%", ha="center", va="bottom", fontsize=7)
    ax_c.text(
        0.99,
        0.92,
        "n=19 internal LOSO cohort;\nno external validation.",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#F9FAFB", "edgecolor": "#CBD5E1", "linewidth": 0.6},
    )

    base = output_dir / "supplementary_performance_precision"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(
        base,
        "Performance precision boundaries for the 19-patient internal LOSO analysis",
        "model metric confidence intervals, paired comparisons, and performance precision audit",
    )


def make_supplementary_clinical_incremental_figure(root: Path, output_dir: Path) -> dict[str, str]:
    perf = pd.read_csv(root / "results" / "tables" / "table2_main_model_performance.csv")
    incremental = pd.read_csv(root / "results" / "statistics" / "clinical_incremental_paired_bootstrap_comparison.csv")

    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    row_by_model = perf.set_index("model_name")
    selected_models = [
        ("Clinical-only", "clinical_only_best_available_clinical_only_logistic_l2", COLORS["clinical"]),
        ("EEG+clinical L1", "eeg_clinical_best_available_eeg_clinical_logistic_l1_selectk100", COLORS["accent"]),
        ("SSL-CNN", "residual_aware_SSL_CNN_seedmean10", COLORS["ssl"]),
    ]
    metric_rows = [("Accuracy", "accuracy"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Brier", "brier_score")]

    panel_label(ax_a, "a")
    x = np.arange(len(metric_rows))
    width = 0.22
    for idx, (label, model_name, color) in enumerate(selected_models):
        values = [float(row_by_model.loc[model_name, metric]) for _, metric in metric_rows]
        ax_a.bar(x + (idx - 1) * width, values, width=width, color=color, label=label)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([label for label, _ in metric_rows])
    ax_a.set_ylim(0, 1.18)
    ax_a.set_ylabel("Subject-level score")
    ax_a.set_title("Clinical context for EEG prediction", pad=22)
    ax_a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=6, ncol=3, handlelength=1.5, columnspacing=0.8)
    ax_a.text(0.98, 0.16, "Brier lower\nis better", transform=ax_a.transAxes, ha="right", va="bottom", fontsize=6)

    panel_label(ax_b, "b")
    candidate_labels = {
        "eeg_clinical_logistic_l1_selectk100": "EEG+clinical\nlogistic L1",
        "eeg_clinical_logistic_l2_selectk100": "EEG+clinical\nlogistic L2",
        "eeg_clinical_svm_rbf_selectk100": "EEG+clinical\nSVM RBF",
    }
    metric_specs = [
        ("accuracy", "Acc.", "#4D7895"),
        ("roc_auc", "ROC", COLORS["ssl"]),
        ("brier_score", "Brier", COLORS["accent"]),
    ]
    rows = []
    for candidate, candidate_label in candidate_labels.items():
        for metric, metric_label, color in metric_specs:
            row = incremental[(incremental["candidate_model"] == candidate) & (incremental["metric"] == metric)].iloc[0]
            rows.append(
                {
                    "label": f"{candidate_label}, {metric_label}",
                    "difference": float(row["difference"]),
                    "low": float(row["ci_low"]),
                    "high": float(row["ci_high"]),
                    "color": color,
                }
            )
    y = np.arange(len(rows))[::-1]
    diffs = np.array([row["difference"] for row in rows])
    lows = np.array([row["low"] for row in rows])
    highs = np.array([row["high"] for row in rows])
    ax_b.axvline(0, color="#111827", lw=0.8, ls="--")
    ax_b.errorbar(diffs, y, xerr=np.vstack([diffs - lows, highs - diffs]), fmt="none", ecolor="#4B5563", elinewidth=0.9, capsize=2.4)
    ax_b.scatter(diffs, y, color=[row["color"] for row in rows], s=22, edgecolor="white", linewidth=0.5, zorder=2)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(["\n".join(textwrap.wrap(row["label"], width=18)) for row in rows])
    ax_b.set_xlabel("Candidate - clinical-only")
    ax_b.set_title("Paired incremental comparisons")
    ax_b.grid(axis="x", color="#E5E7EB", lw=0.6)
    ax_b.text(0.98, 0.06, "Positive Brier\ndifference is worse", transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6)

    panel_label(ax_c, "c")
    metric_direction = {
        "accuracy": ("Accuracy", True),
        "roc_auc": ("ROC-AUC", True),
        "brier_score": ("Brier", False),
    }
    counts = []
    for metric, (display, higher_is_better) in metric_direction.items():
        group = incremental[incremental["metric"] == metric].copy()
        diff = group["difference"].astype(float).to_numpy()
        if higher_is_better:
            benefit = diff
            benefit_low = group["ci_low"].astype(float).to_numpy()
            benefit_high = group["ci_high"].astype(float).to_numpy()
        else:
            benefit = -diff
            benefit_low = -group["ci_high"].astype(float).to_numpy()
            benefit_high = -group["ci_low"].astype(float).to_numpy()
        counts.append(
            {
                "metric": display,
                "favorable_point": int(np.sum(benefit > 0)),
                "unfavorable_point": int(np.sum(benefit < 0)),
                "stable_favorable": int(np.sum(benefit_low > 0)),
                "stable_unfavorable": int(np.sum(benefit_high < 0)),
                "total": len(group),
            }
        )
    xc = np.arange(len(counts))
    fav = np.array([row["favorable_point"] for row in counts])
    unfav = -np.array([row["unfavorable_point"] for row in counts])
    ax_c.axhline(0, color="#111827", lw=0.8)
    ax_c.bar(xc, fav, color=COLORS["clinical"], width=0.54, label="candidate better")
    ax_c.bar(xc, unfav, color="#C9A097", width=0.54, label="clinical-only better")
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([row["metric"] for row in counts])
    ax_c.set_ylabel("Candidate count")
    ax_c.set_title("Direction across exploratory candidates")
    ax_c.set_ylim(min(unfav) - 1.5, max(fav) + 2.5)
    ax_c.legend(frameon=False, loc="upper right")
    for idx, row in enumerate(counts):
        ax_c.text(
            idx,
            max(fav[idx], 0) + 0.35,
            f"stable gain: {row['stable_favorable']}/{row['total']}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    panel_label(ax_d, "d")
    pivot = incremental.pivot_table(index="candidate_model", columns="metric", values="difference", aggfunc="first")
    roc_benefit = pivot["roc_auc"].astype(float)
    brier_benefit = -pivot["brier_score"].astype(float)
    categories = ["EEG+clinical" if str(name).startswith("eeg_clinical_") else "EEG-only" for name in pivot.index]
    colors = [COLORS["accent"] if category == "EEG+clinical" else COLORS["ml"] for category in categories]
    ax_d.axvline(0, color="#111827", lw=0.8, ls="--")
    ax_d.axhline(0, color="#111827", lw=0.8, ls="--")
    ax_d.scatter(roc_benefit, brier_benefit, s=30, color=colors, edgecolor="white", linewidth=0.5)
    highlight = "eeg_clinical_logistic_l1_selectk100"
    if highlight in pivot.index:
        ax_d.scatter(roc_benefit.loc[highlight], brier_benefit.loc[highlight], s=54, color=COLORS["ssl"], edgecolor="#111827", linewidth=0.6, zorder=3)
        ax_d.annotate("best EEG+clinical L1", (roc_benefit.loc[highlight], brier_benefit.loc[highlight]), xytext=(-84, 8), textcoords="offset points", fontsize=6)
    ax_d.set_xlabel("ROC-AUC benefit vs clinical-only")
    ax_d.set_ylabel("Brier benefit vs clinical-only")
    ax_d.set_title("Ranking vs calibration benefit")
    ax_d.grid(color="#E5E7EB", lw=0.6)

    base = output_dir / "supplementary_clinical_incremental_value"
    save_pub(fig, base)
    plt.close(fig)
    return manifest(
        base,
        "Clinical-only variables are strong and EEG-plus-clinical candidates do not show stable incremental gain.",
        "main performance table and clinical incremental paired bootstrap comparisons",
    )


def plot_calibration(ax: plt.Axes, predictions: pd.DataFrame) -> None:
    ax.plot([0, 1], [0, 1], ls="--", color="#222222", lw=0.8, label="Ideal")
    colors = {
        "ML_EEG_updated_no_selector_logistic_l1": COLORS["ml"],
        "no_SSL_CNN_updated_sub05_sub28_seedensemble10": COLORS["cnn"],
        "residual_aware_SSL_CNN_seedmean10": COLORS["ssl"],
    }
    for model_name, group in predictions.groupby("model_name", sort=False):
        subject = group.groupby("subject_id", as_index=False).agg(y_true=("y_true", "first"), y_score=("y_score", "mean"))
        y_true = subject["y_true"].to_numpy(float)
        y_score = subject["y_score"].to_numpy(float)
        bins = np.linspace(0, 1, 6)
        xs = []
        ys = []
        for low, high in zip(bins[:-1], bins[1:], strict=True):
            mask = (y_score >= low) & (y_score <= high if high == 1 else y_score < high)
            if np.any(mask):
                xs.append(float(np.mean(y_score[mask])))
                ys.append(float(np.mean(y_true[mask])))
        ax.plot(xs, ys, marker="o", lw=1.1, ms=3.0, color=colors.get(model_name, "#444444"), label=MODEL_LABELS.get(model_name, model_name))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.legend(frameon=False, loc="lower right")


def add_box(ax: plt.Axes, x: float, y: float, text: str, *, width: float, height: float, face: str) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.8,
        edgecolor=COLORS["line"],
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=7, transform=ax.transAxes)


def add_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.8,
        color=COLORS["line"],
        shrinkA=1,
        shrinkB=1,
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.06, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def shorten_ablation(value: str) -> str:
    value = str(value)
    replacements = {
        "b_no_ssl_cnn_same_arch_10seed": "CNN, no SSL",
        "c_patient_barlow_ssl_no_residual_heads": "Barlow SSL",
        "d_no_ssl_cnn_residual_heads": "Residual heads,\nno SSL",
        "e_patient_barlow_ssl_residual_heads": "Residual-aware\nSSL-CNN",
    }
    return replacements.get(value, "\n".join(textwrap.wrap(value.replace("_", " "), width=16)))


def parse_count(value: object) -> int:
    return int(str(value).split("(", maxsplit=1)[0].strip())


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
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((900, 560), Image.LANCZOS)
            images.append((row["figure"], image.copy()))
    if not images:
        return
    cell_w, cell_h = 960, 630
    columns = 2
    rows_needed = int(np.ceil(len(images) / columns))
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows_needed), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        col = index % columns
        row = index // columns
        x = col * cell_w + (cell_w - image.width) // 2
        y = row * cell_h + 34
        draw.text((col * cell_w + 24, row * cell_h + 12), label, fill=(32, 40, 45))
        sheet.paste(image, (x, y))
    sheet.save(output_dir / "nature_figures_contact_sheet.png")


if __name__ == "__main__":
    main()
