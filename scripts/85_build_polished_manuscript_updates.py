from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.metadata.labels import read_supervised_subject_ids
from eeg_recovery.metadata.subjects import normalize_subject_id


COLORS = {
    "ink": "#263238",
    "muted": "#667085",
    "grid": "#D7DEE8",
    "blue": "#476A8E",
    "teal": "#4E8D7C",
    "green": "#6C8E63",
    "gold": "#C79B3B",
    "red": "#B85C4A",
    "violet": "#7869A6",
    "light": "#F5F7FA",
}

MODEL_LABELS = {
    "ML_EEG_updated_no_selector_logistic_l1": "Logistic L1",
    "ML_EEG_updated_no_selector_logistic_l2": "Logistic L2",
    "ML_EEG_updated_selectk100_svm_rbf": "SVM RBF",
    "no_SSL_CNN_updated_sub05_sub28_seedensemble10": "No-SSL CNN",
    "residual_aware_SSL_CNN_seedmean10": "Residual-aware\nSSL-CNN",
}

CORE_LABELS = {
    "a_ml_psd_wpli_baseline": "ML PSD+WPLI",
    "b_no_ssl_cnn_same_arch_10seed": "No-SSL CNN",
    "c_patient_barlow_ssl_no_residual_heads": "Barlow CNN",
    "d_no_ssl_cnn_residual_heads": "No-SSL residual-aware",
    "e_patient_barlow_ssl_residual_heads": "Residual-aware SSL-CNN",
}

STANDARD_SEEDS = "0,1,2,3,4,5,7,13,21,42"
STANDARD_SEED_VALUES = [int(seed) for seed in STANDARD_SEEDS.split(",")]

MODEL_COMPARISON_COLORS = {
    "Logistic L1": "#4C78A8",
    "Logistic L2": "#72B7B2",
    "SVM RBF": "#E15759",
    "No-SSL CNN": "#F28E2B",
    "Residual-aware CNN": "#7B6CB5",
    "Barlow CNN": "#7B6CB5",
    "Residual-aware\nSSL-CNN": "#2AA876",
}

ABLATION_LABELS = {
    "psd_only": "PSD only",
    "wpli_only": "WPLI only",
    "psd_wpli": "PSD + WPLI",
    "eo_only": "EO only",
    "ec_only": "EC only",
    "psd_eo_only": "PSD EO",
    "wpli_ec_only": "WPLI EC",
    "psd_eo_wpli_ec": "PSD EO + WPLI EC",
    "beta_medium_only": "Beta medium",
    "beta_high_only": "Beta high",
    "beta_medium_beta_high": "Beta medium+high",
    "motor_wpli_edges": "Motor WPLI edges",
}


def main() -> None:
    configure_matplotlib()
    results_tables = PROJECT_ROOT / "results" / "tables"
    fig_dir = PROJECT_ROOT / "results" / "figures" / "revised_initial"
    results_tables.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    config = load_path_config(PROJECT_ROOT / "configs" / "paths.example.yaml")
    table1 = build_patients_information_table(config)
    write_table(table1, results_tables / "table1_cohort_characteristics.csv", results_tables / "table1_cohort_characteristics.md")

    standardize_core_ablation_summary()
    table3 = rebuild_table3()
    write_table(table3, results_tables / "table3_ablation.csv", results_tables / "table3_ablation.md")

    make_model_comparison_figure(fig_dir)
    make_redesigned_ablation_figure(fig_dir)
    print("Updated cohort statistics, ablation table, Figure 4C and Figure 5.")


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
        }
    )


def build_patients_information_table(config) -> pd.DataFrame:
    clinical = read_patient_clinical_source(config.patient_info_clinical_xlsx)
    supervised_ids = set(read_supervised_subject_ids(config.patient_info_integrity_xlsx))
    eeg_ids = discover_patient_eeg_subjects(config.patient_eeg_root)

    clinical["is_eeg_indexed"] = clinical["subject_id"].isin(eeg_ids)
    clinical["is_supervised"] = clinical["subject_id"].isin(supervised_ids)
    clinical["is_ssl_extra"] = clinical["is_eeg_indexed"] & ~clinical["is_supervised"]
    clinical["complete_post_fma"] = clinical["FMA_pre"].notna() & clinical["FMA_post"].notna()
    clinical["complete_supervised_label"] = clinical["is_supervised"]
    clinical["Delta_FMA_obs"] = clinical["FMA_post"] - clinical["FMA_pre"]

    groups = {
        "All clinical records": clinical,
        "Supervised labelled cohort": clinical[clinical["is_supervised"]],
        "Additional EEG-indexed SSL pool": clinical[clinical["is_ssl_extra"]],
    }
    comparison_a = groups["Supervised labelled cohort"]
    comparison_b = groups["Additional EEG-indexed SSL pool"]

    rows: list[dict[str, object]] = []

    def add_category(name: str) -> None:
        rows.append(
            {
                "section": name,
                "item": name,
                "All clinical records": "",
                "Supervised labelled cohort": "",
                "Additional EEG-indexed SSL pool": "",
                "P": "",
                "notes": "",
            }
        )

    def add_row(item: str, values: dict[str, str], p_value: str = "", notes: str = "", section: str = "") -> None:
        row = {"section": section, "item": item, "P": p_value, "notes": notes}
        for group_name in groups:
            row[group_name] = values.get(group_name, "")
        rows.append(row)

    add_row(
        "Subject",
        {group_name: str(len(frame)) for group_name, frame in groups.items()},
        notes="One M1 clinical source record had no current EEG index and is included only in all clinical records.",
    )

    add_category("Demographics")
    add_percent_row(rows, groups, "Gender (woman)", lambda df: df["sex"].astype(str).str.contains("女", na=False), p_categorical(comparison_a, comparison_b, lambda df: df["sex"].astype(str).str.contains("女", na=False)))
    add_continuous_row(rows, groups, "Age, years", "age", p_continuous(comparison_a["age"], comparison_b["age"]))
    add_continuous_row(rows, groups, "Course of disease, months", "duration_months", p_continuous(comparison_a["duration_months"], comparison_b["duration_months"]), notes="One value recorded in days was converted to months by days/30.")

    add_category("Clinical measurements")
    add_percent_row(rows, groups, "Affected upper limb, left", lambda df: df["affected_hand"].astype(str).str.contains("左", na=False), p_categorical(comparison_a, comparison_b, lambda df: df["affected_hand"].astype(str).str.contains("左", na=False)))
    add_percent_row(rows, groups, "Affected upper limb, right", lambda df: df["affected_hand"].astype(str).str.contains("右", na=False))
    add_continuous_row(rows, groups, "FMA-UE before treatment", "FMA_pre", p_continuous(comparison_a["FMA_pre"], comparison_b["FMA_pre"]))
    add_continuous_row(rows, groups, "FMA-UE after 14 sessions", "FMA_post", p_continuous(comparison_a["FMA_post"], comparison_b["FMA_post"]))
    add_continuous_row(rows, groups, "Observed FMA-UE improvement", "Delta_FMA_obs", p_continuous(comparison_a["Delta_FMA_obs"], comparison_b["Delta_FMA_obs"]))
    add_continuous_row(rows, groups, "MBI before treatment", "MBI_pre", p_continuous(comparison_a["MBI_pre"], comparison_b["MBI_pre"]))
    add_continuous_row(rows, groups, "MBI after 14 sessions", "MBI_post", p_continuous(comparison_a["MBI_post"], comparison_b["MBI_post"]))

    add_category("Data availability")
    add_percent_row(rows, groups, "Baseline EEG indexed", lambda df: df["is_eeg_indexed"])
    add_percent_row(rows, groups, "Complete post-treatment FMA-UE", lambda df: df["complete_post_fma"], p_categorical(comparison_a, comparison_b, lambda df: df["complete_post_fma"]))
    add_percent_row(rows, groups, "Complete supervised label", lambda df: df["complete_supervised_label"], "design")

    columns = [
        "section",
        "item",
        "All clinical records",
        "Supervised labelled cohort",
        "Additional EEG-indexed SSL pool",
        "P",
        "notes",
    ]
    return pd.DataFrame(rows).loc[:, columns]


def read_patient_clinical_source(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=3)
    clinical = raw.rename(
        columns={
            "编号": "subject_id",
            "姓名": "name",
            "年龄": "age",
            "病程": "duration_raw",
            "性别": "sex",
            "患病侧（手）": "affected_hand",
            "治疗前FMA": "FMA_pre",
            "治疗后FMA": "FMA_post",
            "治疗前MBI": "MBI_pre",
            "治疗后MBI": "MBI_post",
            "缺少数据": "missing_data",
            "脱落原因": "drop_reason",
        }
    )
    clinical = clinical[clinical["subject_id"].astype(str).str.contains(r"sub\s*\d+", case=False, regex=True, na=False)].copy()
    clinical["subject_id"] = clinical["subject_id"].map(normalize_subject_id)
    clinical = clinical[clinical["affected_hand"].notna()].copy()
    clinical["duration_months"] = clinical["duration_raw"].map(parse_duration_months)
    for column in ["age", "FMA_pre", "FMA_post", "MBI_pre", "MBI_post"]:
        clinical[column] = pd.to_numeric(clinical[column].map(extract_numeric_value), errors="coerce")
    return clinical


def discover_patient_eeg_subjects(root: Path) -> set[str]:
    subject_ids: set[str] = set()
    for path in root.rglob("*"):
        match = re.search(r"sub\s*\d+", str(path), flags=re.IGNORECASE)
        if match:
            subject_ids.add(normalize_subject_id(match.group(0)))
    return subject_ids


def parse_duration_months(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    raw = str(value).strip()
    numeric = extract_numeric_value(raw)
    if pd.isna(numeric):
        return float("nan")
    if "天" in raw:
        return float(numeric) / 30.0
    return float(numeric)


def extract_numeric_value(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else float("nan")


def add_percent_row(rows: list[dict[str, object]], groups: dict[str, pd.DataFrame], item: str, predicate, p_value: str = "", notes: str = "") -> None:
    values = {}
    for group_name, frame in groups.items():
        mask = predicate(frame)
        n = int(mask.sum())
        denominator = max(len(frame), 1)
        values[group_name] = f"{100.0 * n / denominator:.0f}%"
    row = {"section": "", "item": item, "P": p_value, "notes": notes}
    row.update(values)
    rows.append(row)


def add_continuous_row(rows: list[dict[str, object]], groups: dict[str, pd.DataFrame], item: str, column: str, p_value: str = "", notes: str = "") -> None:
    values = {group_name: mean_sd(frame[column]) for group_name, frame in groups.items()}
    row = {"section": "", "item": item, "P": p_value, "notes": notes}
    row.update(values)
    rows.append(row)


def p_continuous(a: pd.Series, b: pd.Series) -> str:
    p_value, _ = continuous_group_test(a, b)
    return format_table_p(p_value)


def p_categorical(frame_a: pd.DataFrame, frame_b: pd.DataFrame, predicate) -> str:
    mask_a = predicate(frame_a)
    mask_b = predicate(frame_b)
    table = [
        [int(mask_a.sum()), int((~mask_a).sum())],
        [int(mask_b.sum()), int((~mask_b).sum())],
    ]
    p_value = float(stats.fisher_exact(table).pvalue)
    return format_table_p(p_value)


def continuous_group_test(a: pd.Series, b: pd.Series) -> tuple[float, str]:
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), "not tested"
    normal_a = len(a) >= 3 and stats.shapiro(a).pvalue >= 0.05
    normal_b = len(b) >= 3 and stats.shapiro(b).pvalue >= 0.05
    if normal_a and normal_b:
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue), "Welch t-test"
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue), "Mann-Whitney U"


def categorical_group_test(labels: pd.DataFrame, variable: str) -> tuple[float, str]:
    levels = sorted(labels[variable].dropna().astype(str).unique())
    if len(levels) != 2:
        return float("nan"), "not tested"
    table = []
    for label_value in [1, 0]:
        frame = labels[labels["label"] == label_value]
        table.append([int((frame[variable].astype(str) == level).sum()) for level in levels])
    if variable == "label":
        return float("nan"), "definition"
    return float(stats.fisher_exact(table).pvalue), "Fisher exact"


def mean_sd(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return "NA"
    if len(values) == 1:
        return f"{values.iloc[0]:.2f} (n=1)"
    return f"{values.mean():.2f} ({chr(177)}{values.std(ddof=1):.2f})"


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_table_p(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return "<0.001"
    if value > 0.995:
        return "1"
    return f"{value:.2f}"


def rebuild_table3() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    core_path = PROJECT_ROOT / "results" / "metrics" / "core_ablation_10seed_summary.csv"
    if core_path.exists():
        core = pd.read_csv(core_path)
        core = core[core["row_type"].isin(["reported", "mean", "ensemble10", "seedmean10"])].copy()
        core["ablation_block"] = "core"
        core["ablation_name"] = core["contrast"]
        frames.append(core)

    modality_path = PROJECT_ROOT / "results" / "metrics" / "modality_state_band_ablation.csv"
    if modality_path.exists():
        modality = pd.read_csv(modality_path)
        if "roc_auc" in modality.columns:
            modality = (
                modality.sort_values(["ablation_name", "roc_auc", "brier_score"], ascending=[True, False, True])
                .groupby("ablation_name", as_index=False)
                .head(1)
            )
        modality["ablation_block"] = "modality_state_band"
        frames.append(modality)

    table = pd.concat(frames, ignore_index=True, sort=False)
    if "model_key" in table.columns and "model" not in table.columns:
        table["model"] = table["model_key"]
    columns = [
        "ablation_block",
        "ablation_name",
        "model",
        "model_key",
        "row_type",
        "n_input_columns",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "interpretation",
        "explainability_alignment",
    ]
    return table.loc[:, [column for column in columns if column in table.columns]].copy()


def standardize_core_ablation_summary() -> None:
    core_path = PROJECT_ROOT / "results" / "metrics" / "core_ablation_10seed_summary.csv"
    core = pd.read_csv(core_path)
    keep = ~core["contrast"].isin(
        [
            "b_no_ssl_cnn_same_arch_10seed",
            "c_patient_barlow_ssl_no_residual_heads",
        ]
    )
    updated = core.loc[keep].copy()
    replacement = pd.concat(
        [
            _standard_no_ssl_core_rows(),
            _standard_barlow_core_rows(),
        ],
        ignore_index=True,
        sort=False,
    )
    updated = pd.concat([updated, replacement], ignore_index=True, sort=False)
    order = {
        "a_ml_psd_wpli_baseline": 0,
        "b_no_ssl_cnn_same_arch_10seed": 1,
        "c_patient_barlow_ssl_no_residual_heads": 2,
        "d_no_ssl_cnn_residual_heads": 3,
        "e_patient_barlow_ssl_residual_heads": 4,
    }
    row_order = {"reported": 0, "mean": 1, "std": 2, "min": 3, "max": 4, "seedmean10": 5, "ensemble10": 5}
    updated["_contrast_order"] = updated["contrast"].map(order).fillna(99)
    updated["_row_order"] = updated["row_type"].map(row_order).fillna(99)
    updated = updated.sort_values(["_contrast_order", "_row_order"]).drop(columns=["_contrast_order", "_row_order"])
    updated.to_csv(core_path, index=False)


def _standard_no_ssl_core_rows() -> pd.DataFrame:
    summary = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "updated_sub05_sub28_no_ssl_exact6cfg_standard10_summary.csv")
    summary = summary.copy()
    summary["row_type"] = summary["row_type"].replace({"ensemble10_fixed_0.5": "seedmean10"})
    return _core_rows_from_summary(
        summary,
        contrast="b_no_ssl_cnn_same_arch_10seed",
        model_key="No-SSL CNN same architecture",
        source_file="updated_sub05_sub28_no_ssl_exact6cfg_standard10_summary.csv",
        interpretation="CNN architecture contribution over traditional ML, using the standard 10-seed set shared by the Barlow CNN comparison.",
        reproduce_command="python -B scripts/05_train_supervised_loso.py --architecture multimodal --feature-kind psd-fc-wpli --fusion gated --encoder cnn --seeds 0 1 2 3 4 5 7 13 21 42",
    )


def _standard_barlow_core_rows() -> pd.DataFrame:
    summary = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "barlow_cnn_10seed_summary.csv")
    seedmean = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "barlow_cnn_10seed_seedmean_metrics.csv")
    seedmean["row_type"] = "seedmean10"
    seedmean["n_seed_runs"] = 10
    seedmean["seed_set"] = STANDARD_SEEDS
    combined = pd.concat([summary, seedmean], ignore_index=True, sort=False)
    return _core_rows_from_summary(
        combined,
        contrast="c_patient_barlow_ssl_no_residual_heads",
        model_key="Patient-level Barlow CNN without residual-aware heads",
        source_file="barlow_cnn_10seed_summary.csv; barlow_cnn_10seed_seedmean_metrics.csv",
        interpretation="Patient-level Barlow SSL contribution before residual-aware auxiliary training, using the standard 10-seed set.",
        reproduce_command="python -B scripts/29_train_patient_barlow_stabilized.py --seeds 0 1 2 3 4 5 7 13 21 42",
    )


def _core_rows_from_summary(
    summary: pd.DataFrame,
    *,
    contrast: str,
    model_key: str,
    source_file: str,
    interpretation: str,
    reproduce_command: str,
) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "sensitivity",
        "specificity",
    ]
    rows = []
    for _, record in summary.iterrows():
        row = {
            "contrast": contrast,
            "model_key": model_key,
            "source_file": source_file,
            "interpretation": interpretation,
            "reproduce_command": reproduce_command,
            "availability": "available",
            "row_type": record["row_type"],
        }
        for col in metric_cols:
            row[col] = record.get(col, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def make_model_comparison_figure(output_dir: Path) -> None:
    perf = build_model_comparison_rows()
    perf.to_csv(output_dir / "figure4c_model_metric_source_data.csv", index=False)
    metrics = [
        ("accuracy", "Mean\naccuracy"),
        ("balanced_accuracy", "Balanced\naccuracy"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ]
    make_model_metric_histogram(perf, metrics, output_dir)
    make_brier_calibration_figure(perf, output_dir)


def make_model_metric_histogram(perf: pd.DataFrame, metrics: list[tuple[str, str]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    panel_label(ax, "a")
    x = np.arange(len(metrics))
    width = 0.12
    offsets = (np.arange(len(perf)) - (len(perf) - 1) / 2) * width

    for model_index, (_, record) in enumerate(perf.iterrows()):
        display = record["display"]
        values = [float(record[metric]) for metric, _ in metrics]
        color = MODEL_COMPARISON_COLORS.get(display, COLORS["blue"])
        edgecolor = COLORS["ink"] if display == "Residual-aware\nSSL-CNN" else "white"
        linewidth = 0.55 if display == "Residual-aware\nSSL-CNN" else 0.35
        bars = ax.bar(
            x + offsets[model_index],
            values,
            width=width * 0.92,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            label=display.replace("\n", " "),
            zorder=3,
        )
        if display == "Residual-aware\nSSL-CNN":
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.018,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    color=COLORS["ink"],
                    rotation=90,
                )

    ax.set_title("Patient-level model scorecard", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=0)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.2,
    )
    save_pub(fig, output_dir / "figure4c_a_model_metric_histogram")
    plt.close(fig)


def make_brier_calibration_figure(perf: pd.DataFrame, output_dir: Path) -> None:
    fig, ax_brier = plt.subplots(figsize=(3.65, 3.55))
    panel_label(ax_brier, "b")
    ordered = perf.sort_values("brier_score", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    ax_brier.hlines(y, 0, ordered["brier_score"], color=COLORS["grid"], lw=1.1)
    point_colors = [MODEL_COMPARISON_COLORS.get(label, COLORS["red"]) for label in ordered["display"]]
    ax_brier.scatter(ordered["brier_score"], y, s=46, color=point_colors, edgecolor="white", linewidth=0.5, zorder=3)
    for yi, value in zip(y, ordered["brier_score"], strict=True):
        ax_brier.text(value + 0.012, yi, f"{value:.3f}", va="center", fontsize=6.4)
    ax_brier.set_yticks(y)
    ax_brier.set_yticklabels(ordered["display"])
    ax_brier.invert_yaxis()
    ax_brier.set_xlim(0, max(0.26, float(ordered["brier_score"].max()) + 0.05))
    ax_brier.set_xlabel("Brier score")
    ax_brier.set_title("Calibration error\n(lower is better)", loc="left")
    ax_brier.grid(axis="x", color=COLORS["grid"], lw=0.5)
    save_pub(fig, output_dir / "figure4c_b_brier_calibration")
    plt.close(fig)


def build_model_comparison_rows() -> pd.DataFrame:
    base = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "table2_main_model_performance.csv")
    name_col = "model_name" if "model_name" in base.columns else "source_model"
    selected = [
        ("ML_EEG_updated_no_selector_logistic_l1", "Logistic L1"),
    ]
    rows = []
    for model_name, display in selected:
        row = base.loc[base[name_col] == model_name].iloc[0].to_dict()
        row["display"] = display
        rows.append(row)
    no_ssl = summarize_seed_metric_table(
        load_standard_seed_metric_table(
            PROJECT_ROOT / "results" / "metrics" / "updated_sub05_sub28_no_ssl_exact6cfg_standard10_per_seed.csv"
        )
    )
    no_ssl["display"] = "No-SSL CNN"
    rows.append(no_ssl)
    residual_cnn = summarize_seed_metric_table(
        load_seed_metric_files("dl_model_comparison_no_ssl_residualaware_highrank_swa_clsalpha1_seed*.csv")
    )
    residual_cnn["display"] = "Residual-aware CNN"
    rows.append(residual_cnn)
    final = summarize_seed_metric_table(
        compute_seed_metrics_from_predictions(PROJECT_ROOT / "results" / "explainability" / "explained_predictions.csv")
    )
    final["display"] = "Residual-aware\nSSL-CNN"
    rows.append(final)
    return pd.DataFrame(rows)


def make_redesigned_ablation_figure(output_dir: Path) -> None:
    seed = collect_seed_metrics()
    core = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "core_ablation_10seed_summary.csv")
    modality = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "modality_state_band_ablation.csv")
    make_split_ablation_figures(output_dir, seed, core, modality)

    fig = plt.figure(figsize=(7.5, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.38, wspace=0.42)
    ax_seed = fig.add_subplot(gs[0, 0])
    ax_core = fig.add_subplot(gs[0, 1])
    ax_rank = fig.add_subplot(gs[1, 0])
    ax_trade = fig.add_subplot(gs[1, 1])

    panel_label(ax_seed, "a")
    plot_seed_distribution(ax_seed, seed)
    panel_label(ax_core, "b")
    plot_core_ablation_bars(ax_core, core, compact=True)
    panel_label(ax_rank, "c")
    plot_ablation_rank(ax_rank, modality)
    panel_label(ax_trade, "d")
    plot_information_tradeoff(ax_trade, modality)

    fig.suptitle("Stability and EEG feature ablation", x=0.02, y=1.01, ha="left", fontsize=9, fontweight="bold")
    save_pub(fig, output_dir / "figure5_stability_ablation")
    plt.close(fig)


def make_split_ablation_figures(output_dir: Path, seed: pd.DataFrame, core: pd.DataFrame, modality: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    panel_label(ax, "a")
    plot_seed_distribution(ax, seed)
    save_pub(fig, output_dir / "figure5a_seed_stability")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 3.7))
    panel_label(ax, "b")
    plot_core_ablation_bars(ax, core, compact=False)
    save_pub(fig, output_dir / "figure5b_core_model_ablation_bars")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.9, 4.2))
    panel_label(ax, "c")
    plot_ablation_rank(ax, modality)
    save_pub(fig, output_dir / "figure5c_feature_state_band_ablation_ranking")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.9, 3.8))
    panel_label(ax, "d")
    plot_information_tradeoff(ax, modality)
    save_pub(fig, output_dir / "figure5d_information_efficiency")
    plt.close(fig)


def collect_seed_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    no_ssl = load_standard_seed_metric_table(PROJECT_ROOT / "results" / "metrics" / "updated_sub05_sub28_no_ssl_exact6cfg_standard10_per_seed.csv")
    no_ssl["model"] = "No-SSL CNN"
    rows.append(no_ssl)
    barlow = load_standard_seed_metric_table(PROJECT_ROOT / "results" / "metrics" / "barlow_cnn_10seed_per_seed.csv")
    barlow["model"] = "Barlow CNN"
    rows.append(barlow)
    no_ssl_residual = load_seed_metric_files("dl_model_comparison_no_ssl_residualaware_highrank_swa_clsalpha1_seed*.csv")
    no_ssl_residual["model"] = "No-SSL residual-aware"
    rows.append(no_ssl_residual)
    final = compute_seed_metrics_from_predictions(PROJECT_ROOT / "results" / "explainability" / "explained_predictions.csv")
    final["model"] = "Residual-aware SSL-CNN"
    rows.append(final)
    return pd.concat(rows, ignore_index=True, sort=False)


def load_standard_seed_metric_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    return ensure_standard_seed_set(frame, path.name)


def load_seed_metric_files(pattern: str) -> pd.DataFrame:
    collected = []
    for path in sorted((PROJECT_ROOT / "results" / "metrics").glob(pattern)):
        collected.append(pd.read_csv(path).iloc[0])
    if not collected:
        raise FileNotFoundError(f"No metric files matched {pattern}")
    frame = pd.DataFrame(collected)
    return ensure_standard_seed_set(frame, pattern)


def summarize_seed_metric_table(frame: pd.DataFrame) -> dict[str, float | int | str]:
    metrics = [
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
    summary: dict[str, float | int | str] = {
        "row_type": "seed_mean",
        "n_seed_runs": int(frame["seed"].nunique()),
        "seed_set": ",".join(str(seed) for seed in STANDARD_SEED_VALUES),
    }
    if "n_subjects" in frame.columns:
        summary["n_subjects"] = int(pd.to_numeric(frame["n_subjects"], errors="coerce").dropna().iloc[0])
    for metric in metrics:
        if metric in frame.columns:
            summary[metric] = float(pd.to_numeric(frame[metric], errors="coerce").mean())
    return summary


def ensure_standard_seed_set(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if "seed" not in frame.columns:
        raise ValueError(f"{source} does not contain a seed column.")
    frame = frame.copy()
    frame["seed"] = frame["seed"].astype(int)
    observed = sorted(frame["seed"].dropna().astype(int).unique().tolist())
    if observed != STANDARD_SEED_VALUES:
        raise ValueError(f"{source} uses seeds {observed}; expected {STANDARD_SEED_VALUES}.")
    return frame.sort_values("seed").reset_index(drop=True)


def compute_seed_metrics_from_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path)
    required = {"seed", "y_true", "y_pred", "y_score"}
    missing = required.difference(pred.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    pred["seed"] = pred["seed"].astype(int)
    observed = sorted(pred["seed"].dropna().unique().tolist())
    if observed != STANDARD_SEED_VALUES:
        raise ValueError(f"{path.name} uses seeds {observed}; expected {STANDARD_SEED_VALUES}.")
    rows = []
    for seed, group in pred.groupby("seed", sort=True):
        y_true = group["y_true"].astype(int).to_numpy()
        y_pred = group["y_pred"].astype(int).to_numpy()
        y_score = group["y_score"].astype(float).to_numpy()
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        rows.append(
            {
                "seed": int(seed),
                "accuracy": float(np.mean(y_true == y_pred)),
                "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
                "roc_auc": float(binary_roc_auc(y_true, y_score)),
                "pr_auc": float(binary_pr_auc(y_true, y_score)),
                "brier_score": float(np.mean((y_score - y_true) ** 2)),
                "sensitivity": sensitivity,
                "specificity": specificity,
            }
        )
    return pd.DataFrame(rows)


def compute_seedmean_metrics_from_predictions(path: Path) -> dict[str, float]:
    pred = pd.read_csv(path)
    required = {"seed", "subject_id", "y_true", "y_score"}
    missing = required.difference(pred.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    pred["seed"] = pred["seed"].astype(int)
    observed = sorted(pred["seed"].dropna().unique().tolist())
    if observed != STANDARD_SEED_VALUES:
        raise ValueError(f"{path.name} uses seeds {observed}; expected {STANDARD_SEED_VALUES}.")
    group_cols = ["subject_id", "y_true"]
    if "fold_index" in pred.columns:
        group_cols.insert(1, "fold_index")
    seedmean = pred.groupby(group_cols, as_index=False)["y_score"].mean()
    y_true = seedmean["y_true"].astype(int).to_numpy()
    y_score = seedmean["y_score"].astype(float).to_numpy()
    y_pred = (y_score >= 0.5).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "roc_auc": float(binary_roc_auc(y_true, y_score)),
        "pr_auc": float(binary_pr_auc(y_true, y_score)),
        "brier_score": float(np.mean((y_score - y_true) ** 2)),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positive = y_score[y_true == 1]
    negative = y_score[y_true == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = 0.5 * (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + ties)


def binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if int(np.sum(y_true == 1)) == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / tp[-1]
    return float(np.sum((recall - np.concatenate([[0.0], recall[:-1]])) * precision))


def plot_seed_distribution(ax: plt.Axes, seed: pd.DataFrame) -> None:
    order = ["No-SSL CNN", "Barlow CNN", "No-SSL residual-aware", "Residual-aware SSL-CNN"]
    palette = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["red"]]
    data = [seed.loc[seed["model"] == model, "accuracy"].dropna().to_numpy(float) for model in order]
    violins = ax.violinplot(data, showmeans=False, showmedians=False, widths=0.78)
    for body, color in zip(violins["bodies"], palette, strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.20)
        body.set_edgecolor(color)
    rng = np.random.default_rng(20260602)
    for idx, (values, color) in enumerate(zip(data, palette, strict=True), start=1):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(np.full(len(values), idx) + jitter, values, s=18, color=color, edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(idx, np.median(values), marker="D", s=36, color="white", edgecolor=color, linewidth=1.0, zorder=4)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(["No-SSL\nCNN", "Barlow\nCNN", "No-SSL\nresidual", "Final\nSSL-CNN"], rotation=0)
    ax.set_ylim(0.50, 0.94)
    ax.set_ylabel("Seed-level accuracy")
    ax.set_title("Random-seed stability", loc="left")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
    ax.text(0.02, 0.03, "Diamonds mark medians", transform=ax.transAxes, fontsize=6.2, color=COLORS["muted"])


def plot_core_scorecard(ax: plt.Axes, core: pd.DataFrame) -> None:
    mean = core[core["row_type"].isin(["reported", "mean"])].copy()
    mean = mean[mean["contrast"].isin(CORE_LABELS)].copy()
    mean = mean.set_index("contrast").loc[list(CORE_LABELS)].reset_index()
    rows = [CORE_LABELS[c] for c in mean["contrast"]]
    metrics = [
        ("accuracy", "Mean\naccuracy", False),
        ("roc_auc", "ROC-AUC", False),
        ("pr_auc", "PR-AUC", False),
        ("brier_score", "Brier", True),
    ]
    values = []
    labels = []
    for _, record in mean.iterrows():
        row = []
        label_row = []
        for metric, _, lower in metrics:
            value = record.get(metric, np.nan)
            label_row.append("NA" if pd.isna(value) else f"{value:.2f}")
            if pd.isna(value):
                row.append(np.nan)
            elif lower:
                row.append(1.0 - float(value))
            else:
                row.append(float(value))
        values.append(row)
        labels.append(label_row)
    arr = np.array(values, dtype=float)
    color_arr = arr.copy()
    for j in range(color_arr.shape[1]):
        col = color_arr[:, j]
        finite = np.isfinite(col)
        if finite.sum() > 1:
            lo = np.nanmin(col)
            hi = np.nanmax(col)
            if hi > lo:
                color_arr[finite, j] = (col[finite] - lo) / (hi - lo)
            else:
                color_arr[finite, j] = 0.5
        elif finite.sum() == 1:
            color_arr[finite, j] = 1.0
    masked = np.ma.masked_invalid(color_arr)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_title("Core model ablation", loc="left")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.tick_params(length=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isnan(arr[i, j]):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=6.3, color=COLORS["muted"])
                continue
            txt = ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=6.8, color=COLORS["ink"])
            txt.set_path_effects([pe.Stroke(linewidth=1.5, foreground="white"), pe.Normal()])
    cbar = plt.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_label("Column-normalized\nperformance\n(Brier inverted)")


def prepare_core_ablation_mean(core: pd.DataFrame) -> pd.DataFrame:
    mean = core[core["row_type"].isin(["reported", "mean"])].copy()
    mean = mean[mean["contrast"].isin(CORE_LABELS)].copy()
    mean = mean.set_index("contrast").loc[list(CORE_LABELS)].reset_index()
    mean["display"] = [CORE_LABELS[c] for c in mean["contrast"]]
    mean["short_display"] = [
        "ML\nPSD+WPLI",
        "No-SSL\nCNN",
        "Barlow\nCNN",
        "No-SSL\nresidual",
        "Final\nSSL-CNN",
    ]
    return mean


def plot_core_ablation_bars(ax: plt.Axes, core: pd.DataFrame, compact: bool = False) -> None:
    mean = prepare_core_ablation_mean(core)
    metrics = [
        ("accuracy", "Mean\naccuracy", COLORS["blue"]),
        ("roc_auc", "ROC-AUC", COLORS["gold"]),
        ("pr_auc", "PR-AUC", COLORS["violet"]),
        ("brier_score", "1-Brier", COLORS["red"]),
    ]
    x = np.arange(len(mean))
    width = 0.17 if not compact else 0.15
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2) * width
    for metric_index, (metric, label, color) in enumerate(metrics):
        raw = mean[metric].to_numpy(float)
        values = 1.0 - raw if metric == "brier_score" else raw
        bars = ax.bar(
            x + offsets[metric_index],
            values,
            width=width * 0.92,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=label.replace("\n", " "),
            zorder=3,
        )
        if not compact:
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.018,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    color=COLORS["ink"],
                    rotation=90,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(mean["short_display"], rotation=0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score\n(Brier shown as 1-Brier)")
    ax.set_title("Core model ablation", loc="left")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5, zorder=0)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20 if not compact else -0.24),
        ncol=4 if not compact else 2,
        columnspacing=0.9,
        handlelength=1.1,
    )


def plot_ablation_rank(ax: plt.Axes, modality: pd.DataFrame) -> None:
    plot = modality[modality["ablation_name"].isin(ABLATION_LABELS)].copy()
    plot = plot.sort_values("roc_auc", ascending=True)
    y = np.arange(len(plot))
    ax.hlines(y, plot["accuracy"], plot["roc_auc"], color=COLORS["grid"], lw=1.0)
    ax.scatter(plot["accuracy"], y, s=22, color=COLORS["blue"], label="Accuracy", zorder=3)
    ax.scatter(plot["roc_auc"], y, s=30, color=COLORS["red"], label="ROC-AUC", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([ABLATION_LABELS.get(v, v) for v in plot["ablation_name"]])
    ax.set_xlim(0.20, 0.88)
    ax.set_xlabel("Score")
    ax.set_title("Feature/state/band ablation ranking", loc="left")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.20), ncol=2)


def plot_information_tradeoff(ax: plt.Axes, modality: pd.DataFrame) -> None:
    plot = modality[modality["ablation_name"].isin(ABLATION_LABELS)].copy()
    families = []
    for name in plot["ablation_name"].astype(str):
        if name.startswith("psd") and "wpli" not in name:
            families.append("PSD")
        elif "wpli" in name and not name.startswith("psd_wpli"):
            families.append("WPLI")
        elif name in {"eo_only", "ec_only"}:
            families.append("State")
        elif "beta" in name:
            families.append("Band")
        else:
            families.append("Combined")
    plot["family"] = families
    family_colors = {"PSD": COLORS["blue"], "WPLI": COLORS["violet"], "State": COLORS["teal"], "Band": COLORS["gold"], "Combined": COLORS["red"]}
    for family, frame in plot.groupby("family"):
        sizes = 24 + 78 * frame["pr_auc"].fillna(0).to_numpy(float)
        ax.scatter(
            frame["n_input_columns"],
            frame["roc_auc"],
            s=sizes,
            color=family_colors[family],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.55,
            label=family,
        )
    label_offsets = {
        "psd_only": (18, 10, "left"),
        "ec_only": (20, 26, "left"),
        "motor_wpli_edges": (-18, -16, "right"),
        "beta_medium_only": (-18, 14, "right"),
        "eo_only": (20, 0, "left"),
    }
    for _, row in plot[plot["ablation_name"].isin(label_offsets)].iterrows():
        dx, dy, ha = label_offsets[row["ablation_name"]]
        ax.annotate(
            ABLATION_LABELS[row["ablation_name"]],
            xy=(row["n_input_columns"], row["roc_auc"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.8,
            ha=ha,
            va="center",
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.86},
            arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "lw": 0.45, "shrinkA": 0, "shrinkB": 5},
            zorder=5,
            clip_on=False,
        )
    ax.set_xscale("log")
    ax.set_xlim(320, 36000)
    ax.set_ylim(0.22, 0.86)
    ax.set_xlabel("Input dimensionality (log scale)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Information efficiency", loc="left")
    ax.grid(which="major", color=COLORS["grid"], lw=0.5)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=5.0,
            markerfacecolor=family_colors[name],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=name,
        )
        for name in ["Band", "Combined", "PSD", "State", "WPLI"]
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=5,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.0,
        fontsize=6.3,
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def save_pub(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def write_table(frame: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    try:
        md = frame.to_markdown(index=False)
    except Exception:
        md = frame.to_string(index=False)
    md_path.write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
