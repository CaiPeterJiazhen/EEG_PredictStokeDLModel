from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table


MAIN_MODEL = "residualaware_highrank_swa_clsalpha1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create manuscript-ready tables and lightweight figures.")
    parser.add_argument("--config", default="configs/paths.example.yaml")
    args = parser.parse_args()

    config = load_path_config(args.config)
    output_root = config.output_root
    tables_dir = output_root / "results" / "tables"
    figures_dir = output_root / "results" / "figures" / "paper"
    docs_assets = output_root / "docs" / "paper_assets"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    docs_assets.mkdir(parents=True, exist_ok=True)

    labels = load_supervised_label_table(config)
    patient_table = _make_patient_characteristics(labels)
    patient_table.to_csv(tables_dir / "patient_characteristics_table.csv", index=False)

    performance = _make_model_performance_table(output_root)
    performance.to_csv(tables_dir / "model_performance_main_table.csv", index=False)

    stability = _make_seed_stability_table(output_root)
    stability.to_csv(tables_dir / "seed_stability_table.csv", index=False)

    findings = _make_explainability_key_findings(output_root)
    findings.to_csv(tables_dir / "explainability_key_findings_table.csv", index=False)

    figure_plan = _make_figure_plan()
    figure_plan.to_csv(docs_assets / "figure_plan.csv", index=False)

    _plot_pipeline(figures_dir / "figure1_cohort_pipeline.png")
    _plot_architecture(figures_dir / "figure2_model_architecture.png")
    _plot_metric_bars(performance, figures_dir / "figure3_model_performance.png")
    _plot_stability(stability, figures_dir / "figure3_seed_stability.png")
    _plot_topomap_panel(
        output_root / "results" / "figures" / "explainability" / "topomaps",
        figures_dir / "figure4_psd_attribution_topomaps.png",
    )
    _copy_or_placeholder(
        output_root / "results" / "figures" / "explainability" / "wpli_connectome_top20_beta_medium.png",
        figures_dir / "figure5_wpli_beta_connectome.png",
        title="WPLI beta-band connectome placeholder",
    )
    _copy_or_placeholder(
        output_root / "results" / "figures" / "explainability" / "branch_state_occlusion_barplot.png",
        figures_dir / "figure6_branch_state_occlusion.png",
        title="Branch/state occlusion placeholder",
    )
    _plot_subject_errors(
        output_root / "results" / "metrics" / "patient_barlow_residualaware_highrank_swa_clsalpha1_subject_error_frequency.csv",
        figures_dir / "figure7_subject_error_frequency.png",
    )
    _write_assets_readme(docs_assets / "README.md")
    print(f"Wrote paper tables to {tables_dir}")
    print(f"Wrote paper figures to {figures_dir}")


def _make_patient_characteristics(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "all": labels,
        "proportional_label1": labels[labels["label"] == 1],
        "poor_recovery_label0": labels[labels["label"] == 0],
    }
    numeric = ["age", "duration", "FMA_pre", "FMA_post", "Delta_FMA_obs", "Residual", "MBI_pre", "MBI_post"]
    for variable in numeric:
        row = {"variable": variable, "type": "mean_sd"}
        for name, frame in groups.items():
            row[name] = f"{frame[variable].mean():.2f} ({frame[variable].std(ddof=1):.2f})"
        rows.append(row)
    for variable in ["sex", "affected_hand", "label"]:
        levels = sorted(labels[variable].dropna().astype(str).unique())
        for level in levels:
            row = {"variable": f"{variable}={level}", "type": "n"}
            for name, frame in groups.items():
                row[name] = int((frame[variable].astype(str) == level).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _make_model_performance_table(output_root: Path) -> pd.DataFrame:
    source = pd.read_csv(output_root / "results" / "metrics" / "clinical_baseline_model_comparison.csv")
    order = [
        ("ML_EEG_logistic_l1", "Logistic L1"),
        ("ML_EEG_logistic_l2", "Logistic L2"),
        ("baseline_clinical_only_logistic", "Clinical-only logistic"),
        ("qEEG_only_logistic", "qEEG-only logistic"),
        ("no_SSL_CNN_seedmean10", "no-SSL CNN"),
        ("residual_aware_SSL_CNN_seedmean10", "Residual-aware SSL-CNN"),
    ]
    rows = []
    for model_id, display in order:
        match = source[source["model"] == model_id]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        rows.append(
            {
                "model": display,
                "source_model": model_id,
                "accuracy": row.get("accuracy"),
                "accuracy_95ci": _format_ci(row, "accuracy"),
                "balanced_accuracy": row.get("balanced_accuracy"),
                "sensitivity": row.get("sensitivity"),
                "specificity": row.get("specificity"),
                "roc_auc": row.get("roc_auc"),
                "roc_auc_95ci": _format_ci(row, "roc_auc"),
                "pr_auc": row.get("pr_auc"),
                "pr_auc_95ci": _format_ci(row, "pr_auc"),
                "brier_score": row.get("brier_score"),
                "brier_95ci": _format_ci(row, "brier_score"),
            }
        )
    return pd.DataFrame(rows)


def _make_seed_stability_table(output_root: Path) -> pd.DataFrame:
    rows = []
    rows.append(
        _stability_from_summary(
            output_root / "results" / "metrics" / "no_ssl_psdfcwpli_gated_cnn_rerun_20260531_10seed_summary.csv",
            "no-SSL CNN",
            model_column=None,
        )
    )
    rows.append(
        _stability_from_method_summary(
            output_root / "results" / "metrics" / "updated_sub05_sub28_10seed_no_ssl_barlow_cnn_summary.csv",
            "barlow_ssl",
            "Patient-level Barlow baseline",
        )
    )
    rows.append(
        _stability_from_summary(
            output_root / "results" / "metrics" / "patient_barlow_residualaware_highrank_swa_clsalpha1_10seed_summary.csv",
            "Residual-aware SSL-CNN",
            model_column="model_group",
        )
    )
    return pd.DataFrame(rows)


def _make_explainability_key_findings(output_root: Path) -> pd.DataFrame:
    rows = []
    psd = pd.read_csv(output_root / "results" / "explainability" / "psd_channel_band_importance.csv")
    for _, row in psd.sort_values("mean_abs_attribution", ascending=False).head(5).iterrows():
        rows.append(
            {
                "finding_type": "PSD channel-band",
                "finding": f"{row['state']} {row['channel']} {row['band']}",
                "metric": "mean_abs_attribution",
                "value": row["mean_abs_attribution"],
                "stability_or_validation": "SmoothGrad-IG summary",
            }
        )
    wpli = pd.read_csv(output_root / "results" / "explainability" / "network_level_biomarker_validation.csv")
    for _, row in wpli.sort_values("spearman_signed_distance_fdr_p").head(5).iterrows():
        rows.append(
            {
                "finding_type": "Network/ROI validation",
                "finding": row["summary_name"],
                "metric": "rho_vs_signed_distance_fdr",
                "value": row["spearman_signed_distance_r"],
                "stability_or_validation": f"q={row['spearman_signed_distance_fdr_p']:.4f}; {row['effect_direction']}",
            }
        )
    sanity = pd.read_csv(output_root / "results" / "explainability" / "sanity_check_summary.csv")
    for _, row in sanity.head(3).iterrows():
        rows.append(
            {
                "finding_type": "Sanity check",
                "finding": row.get("check", row.get("sanity_check", "sanity")),
                "metric": row.get("metric", "correlation"),
                "value": row.get("value", np.nan),
                "stability_or_validation": "Existing explainability sanity check",
            }
        )
    return pd.DataFrame(rows)


def _stability_from_summary(path: Path, display_name: str, *, model_column: str | None) -> dict[str, object]:
    frame = pd.read_csv(path)
    row_type_col = "row_type"
    mean = frame[frame[row_type_col] == "mean"].iloc[0]
    std = frame[frame[row_type_col] == "std"].iloc[0]
    min_row = frame[frame[row_type_col] == "min"].iloc[0]
    return {
        "model": display_name,
        "mean_accuracy": mean["accuracy"],
        "std_accuracy": std["accuracy"],
        "min_accuracy": min_row["accuracy"],
        "mean_roc_auc": mean["roc_auc"],
        "mean_pr_auc": mean["pr_auc"],
        "mean_brier": mean["brier_score"],
    }


def _stability_from_method_summary(path: Path, method: str, display_name: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    subset = frame[frame["method"] == method]
    mean = subset[subset["row_type"] == "mean"].iloc[0]
    std = subset[subset["row_type"] == "std"].iloc[0]
    min_row = subset[subset["row_type"] == "min"].iloc[0]
    return {
        "model": display_name,
        "mean_accuracy": mean["accuracy"],
        "std_accuracy": std["accuracy"],
        "min_accuracy": min_row["accuracy"],
        "mean_roc_auc": mean["roc_auc"],
        "mean_pr_auc": mean["pr_auc"],
        "mean_brier": np.nan,
    }


def _make_figure_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"figure": "Figure 1", "asset": "figure1_cohort_pipeline.png", "purpose": "Cohort and analysis pipeline"},
            {"figure": "Figure 2", "asset": "figure2_model_architecture.png", "purpose": "Locked residual-aware SSL-CNN architecture"},
            {"figure": "Figure 3", "asset": "figure3_seed_stability.png", "purpose": "10-seed performance and stability"},
            {"figure": "Figure 4", "asset": "figure4_psd_attribution_topomaps.png", "purpose": "PSD attribution topomaps"},
            {"figure": "Figure 5", "asset": "figure5_wpli_beta_connectome.png", "purpose": "WPLI beta-band connectome"},
            {"figure": "Figure 6", "asset": "figure6_branch_state_occlusion.png", "purpose": "Branch/state occlusion"},
            {"figure": "Figure 7", "asset": "figure7_subject_error_frequency.png", "purpose": "Post-hoc subject error frequency"},
        ]
    )


def _format_ci(row: dict[str, object], metric: str) -> str:
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if pd.isna(low) or pd.isna(high):
        return ""
    return f"{float(low):.3f}-{float(high):.3f}"


def _plot_pipeline(path: Path) -> None:
    _box_figure(
        path,
        "Cohort / Pipeline",
        ["19 baseline tACS stroke EEG", "PSD EO/EC + WPLI EO/EC", "Patient-level LOSO", "10-seed stability", "Locked final model + explainability"],
    )


def _plot_architecture(path: Path) -> None:
    _box_figure(
        path,
        "Residual-aware SSL-CNN",
        ["Patient-level Barlow encoder", "PSD branch + WPLI branch", "Gated fusion CNN embedding", "Classification head for inference", "Residual/ranking/soft heads for training only"],
    )


def _box_figure(path: Path, title: str, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_axis_off()
    xs = np.linspace(0.1, 0.9, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels, strict=True)):
        ax.text(x, 0.55, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#f5f7fb", ec="#2f4057"))
        if index < len(labels) - 1:
            ax.annotate("", xy=(xs[index + 1] - 0.08, 0.55), xytext=(x + 0.08, 0.55), arrowprops=dict(arrowstyle="->", color="#2f4057"))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric_bars(performance: pd.DataFrame, path: Path) -> None:
    subset = performance[performance["model"].isin(["Clinical-only logistic", "qEEG-only logistic", "no-SSL CNN", "Residual-aware SSL-CNN"])]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(subset))
    ax.bar(x - 0.2, subset["accuracy"], width=0.2, label="Accuracy")
    ax.bar(x, subset["roc_auc"], width=0.2, label="ROC AUC")
    ax.bar(x + 0.2, subset["pr_auc"], width=0.2, label="PR AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(subset["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Main model performance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_stability(stability: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(stability))
    ax.bar(x, stability["mean_accuracy"], yerr=stability["std_accuracy"], capsize=4)
    ax.scatter(x, stability["min_accuracy"], color="black", label="Min accuracy", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(stability["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("10-seed accuracy stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_topomap_panel(topomap_dir: Path, path: Path) -> None:
    names = [
        "psd_eo_delta_signed_attribution_topomap.png",
        "psd_eo_beta_high_signed_attribution_topomap.png",
        "psd_ec_alpha_signed_attribution_topomap.png",
        "psd_ec_beta_medium_signed_attribution_topomap.png",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    for ax, name in zip(axes.ravel(), names, strict=True):
        source = topomap_dir / name
        if source.exists():
            ax.imshow(mpimg.imread(source))
            ax.set_title(name.replace("_signed_attribution_topomap.png", "").replace("_", " "))
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _copy_or_placeholder(source: Path, target: Path, *, title: str) -> None:
    if source.exists():
        shutil.copyfile(source, target)
        return
    _box_figure(target, title, ["Source figure not available", "See CSV summaries"])


def _plot_subject_errors(source: Path, target: Path) -> None:
    frame = pd.read_csv(source).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(frame["subject_id"], frame["error_rate"], color="#5b7c99")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Error rate across 10 seeds")
    ax.set_title("Post-hoc repeated-error subjects")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def _write_assets_readme(path: Path) -> None:
    path.write_text(
        "# Paper Assets\n\n"
        "This directory indexes manuscript-ready table and figure assets generated from locked results. "
        "The assets do not change the final model selection.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
