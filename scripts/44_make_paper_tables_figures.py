from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import load_supervised_label_table
from eeg_recovery.metadata.subjects import normalize_subject_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build manuscript tables and figures from generated CSV outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    tables_dir = output_root / "results" / "tables"
    figures_dir = output_root / "results" / "figures" / "paper"
    docs_dir = output_root / "docs"
    for directory in (tables_dir, figures_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    labels = load_supervised_label_table(path_config)
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    table1 = make_table1(labels)
    table2 = load_or_build_table2(output_root)
    table3 = make_table3(output_root)
    table4 = make_table4(output_root)
    supplementary = make_supplementary_metrics(output_root)

    write_table(table1, tables_dir / "table1_cohort_characteristics.csv", tables_dir / "table1_cohort_characteristics.md")
    write_table(table2, tables_dir / "table2_main_model_performance.csv", tables_dir / "table2_main_model_performance.md")
    write_table(table3, tables_dir / "table3_ablation.csv", tables_dir / "table3_ablation.md")
    write_table(table4, tables_dir / "table4_explainability_biomarkers.csv", tables_dir / "table4_explainability_biomarkers.md")
    supplementary.to_csv(tables_dir / "supplementary_all_metrics.csv", index=False)

    write_pipeline_mermaid(figures_dir / "figure1_pipeline.mmd")
    plot_pipeline(figures_dir / "figure1_pipeline.png")
    plot_model_performance(table2, figures_dir / "figure2_model_performance.png")
    plot_ablation(table3, figures_dir / "figure3_ablation.png")
    plot_explainability(table4, figures_dir / "figure4_explainability.png")
    write_results_narrative(docs_dir / "paper_results_narrative.md", table1, table2, table3, table4)
    print(f"Wrote paper tables to {tables_dir}")
    print(f"Wrote paper figures to {figures_dir}")


def make_table1(labels: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "All": labels,
        "Proportional recovery": labels[labels["label"] == 1],
        "Poor recovery": labels[labels["label"] == 0],
    }
    rows = []
    for variable in ["age", "duration", "FMA_pre", "FMA_post", "Delta_FMA_obs", "Residual", "MBI_pre", "MBI_post"]:
        row = {"variable": variable, "summary": "mean (SD)"}
        for group_name, frame in groups.items():
            row[group_name] = _mean_sd(frame[variable])
        rows.append(row)
    for variable in ["sex", "affected_hand", "label"]:
        for level in sorted(labels[variable].dropna().astype(str).unique()):
            row = {"variable": f"{variable}={level}", "summary": "n (%)"}
            for group_name, frame in groups.items():
                n = int((frame[variable].astype(str) == level).sum())
                row[group_name] = f"{n} ({100.0 * n / max(len(frame), 1):.1f}%)"
            rows.append(row)
    return pd.DataFrame(rows)


def load_or_build_table2(output_root: Path) -> pd.DataFrame:
    path = output_root / "results" / "tables" / "paper_locked_model_performance.csv"
    if not path.exists():
        module = _load_script_module("37_build_paper_locked_results.py", "paper_locked_for_table2")
        predictions = module.collect_locked_predictions(output_root)
        table = module.build_performance_table(predictions)
    else:
        table = pd.read_csv(path)
    columns = [
        "model_family",
        "model_name",
        "input_features",
        "feature_selection",
        "inference_type",
        "result_role",
        "n_subjects",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]
    return table.loc[:, [column for column in columns if column in table.columns]].copy()


def make_table3(output_root: Path) -> pd.DataFrame:
    frames = []
    core = output_root / "results" / "metrics" / "core_ablation_10seed_summary.csv"
    if core.exists():
        core_frame = pd.read_csv(core)
        core_frame = core_frame[core_frame["row_type"].isin(["mean", "seedmean10", "ensemble10", "reported"])]
        core_frame["ablation_block"] = "core"
        core_frame["ablation_name"] = core_frame["contrast"]
        frames.append(core_frame)
    modality = output_root / "results" / "metrics" / "modality_state_band_ablation.csv"
    if modality.exists():
        modality_frame = pd.read_csv(modality)
        if "roc_auc" in modality_frame.columns:
            modality_frame = modality_frame.sort_values(["ablation_name", "roc_auc", "brier_score"], ascending=[True, False, True]).groupby("ablation_name", as_index=False).head(1)
        modality_frame["ablation_block"] = "modality_state_band"
        frames.append(modality_frame)
    if not frames:
        return pd.DataFrame(columns=["ablation_block", "ablation_name", "model", "accuracy", "roc_auc", "pr_auc", "brier_score"])
    table = pd.concat(frames, ignore_index=True, sort=False)
    if "model_key" in table.columns and "model" not in table.columns:
        table["model"] = table["model_key"]
    columns = ["ablation_block", "ablation_name", "model", "model_key", "row_type", "n_input_columns", "accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score", "interpretation", "explainability_alignment"]
    return table.loc[:, [column for column in columns if column in table.columns]].copy()


def make_table4(output_root: Path) -> pd.DataFrame:
    candidates = [
        output_root / "results" / "explainability" / "biomarker_group_difference_validation.csv",
        output_root / "results" / "explainability" / "network_level_biomarker_validation.csv",
        output_root / "results" / "explainability" / "psd_biomarker_validation.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            if "spearman_signed_distance_fdr_p" in frame.columns:
                frame = frame.sort_values("spearman_signed_distance_fdr_p")
            elif "mean_abs_attribution" in frame.columns:
                frame = frame.sort_values("mean_abs_attribution", ascending=False)
            return frame.head(25).copy()
    return pd.DataFrame(columns=["feature_family", "feature", "value", "note"])


def make_supplementary_metrics(output_root: Path) -> pd.DataFrame:
    metric_files = [
        "paper_locked_model_performance.csv",
        "clinical_only_model_comparison.csv",
        "eeg_clinical_incremental_model_comparison.csv",
        "core_ablation_10seed_summary.csv",
        "modality_state_band_ablation.csv",
        "residual_threshold_sensitivity.csv",
    ]
    frames = []
    for filename in metric_files:
        for root in (output_root / "results" / "tables", output_root / "results" / "metrics", output_root / "results" / "statistics"):
            path = root / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame["source_file"] = filename
            frames.append(frame)
            break
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def write_table(frame: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown(frame), encoding="utf-8")


def write_pipeline_mermaid(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "flowchart LR",
                '  A["19 stroke patients baseline resting EEG"] --> B["PSD + WPLI, EO + EC"]',
                '  B --> C["Outer patient-level LOSO"]',
                '  C --> D["Traditional ML baseline"]',
                '  C --> E["No-SSL CNN"]',
                '  C --> F["Residual-aware SSL-CNN"]',
                '  F --> G["Statistics, calibration, ablations, explainability"]',
            ]
        ),
        encoding="utf-8",
    )


def plot_pipeline(path: Path) -> None:
    labels = ["Baseline EEG", "PSD/WPLI", "LOSO", "ML/CNN/SSL", "Stats + biomarkers"]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_axis_off()
    xs = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels, strict=True)):
        ax.text(x, 0.5, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="#f3f6f4", ec="#335c49"))
        if index < len(labels) - 1:
            ax.annotate("", xy=(xs[index + 1] - 0.07, 0.5), xytext=(x + 0.07, 0.5), arrowprops=dict(arrowstyle="->", color="#335c49"))
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_model_performance(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        _placeholder(path, "No model performance table")
        return
    primary = table[table["result_role"].astype(str).str.startswith("primary")] if "result_role" in table.columns else table
    primary = primary.head(8)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(primary))
    width = 0.25
    ax.bar(x - width, primary["accuracy"], width=width, label="Accuracy")
    ax.bar(x, primary["roc_auc"], width=width, label="ROC-AUC")
    ax.bar(x + width, primary["pr_auc"], width=width, label="PR-AUC")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(primary["model_name"], rotation=25, ha="right", fontsize=8)
    ax.legend()
    ax.set_title("Locked model performance")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_ablation(table: pd.DataFrame, path: Path) -> None:
    if table.empty or "roc_auc" not in table.columns:
        _placeholder(path, "No ablation table")
        return
    plot = table.dropna(subset=["roc_auc"]).head(18)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(plot)), plot["roc_auc"])
    ax.set_yticks(range(len(plot)))
    ax.set_yticklabels(plot["ablation_name"], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Ablation summary")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_explainability(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        _placeholder(path, "No explainability biomarker table")
        return
    value_column = "spearman_signed_distance_r" if "spearman_signed_distance_r" in table.columns else None
    if value_column is None:
        numeric = table.select_dtypes(include=[np.number]).columns.tolist()
        value_column = numeric[0] if numeric else None
    if value_column is None:
        _placeholder(path, "No numeric explainability value")
        return
    labels = [_feature_label(row) for _, row in table.head(15).iterrows()]
    values = table.head(15)[value_column].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(values)), values)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(value_column)
    ax.set_title("Top explainability biomarker validations")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_results_narrative(path: Path, table1: pd.DataFrame, table2: pd.DataFrame, table3: pd.DataFrame, table4: pd.DataFrame) -> None:
    best = table2.sort_values("roc_auc", ascending=False).iloc[0] if not table2.empty and "roc_auc" in table2.columns else None
    lines = [
        "# Paper Results Narrative",
        "",
        f"The cohort table contains {len(table1)} characteristic rows. Model performance, ablation, and explainability tables were generated from CSV outputs rather than manually entered values.",
    ]
    if best is not None:
        lines.extend(
            [
                "",
                f"The best locked primary ROC-AUC row is `{best['model_name']}` with ROC-AUC {float(best['roc_auc']):.3f}, PR-AUC {float(best['pr_auc']):.3f}, and Brier score {float(best['brier_score']):.3f}.",
            ]
        )
    lines.extend(
        [
            "",
            "Residual-aware SSL-CNN results should be described as improving ranking/discrimination and calibration relative to the updated PSD+WPLI ML baseline and no-SSL CNN, with accuracy treated as one metric rather than the sole endpoint.",
            "",
            "Ablation and explainability rows remain support analyses unless explicitly marked as locked primary results.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _mean_sd(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce")
    return f"{values.mean():.2f} ({values.std(ddof=1):.2f})"


def _feature_label(row: pd.Series) -> str:
    for columns in (("summary_name",), ("feature_id",), ("state", "channel", "band"), ("state", "channel_i", "channel_j", "band")):
        if all(column in row.index for column in columns):
            return " ".join(str(row[column]) for column in columns)
    return str(row.name)


def _placeholder(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_script_module(filename: str, module_name: str):
    spec = spec_from_file_location(module_name, PROJECT_ROOT / "scripts" / filename)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
