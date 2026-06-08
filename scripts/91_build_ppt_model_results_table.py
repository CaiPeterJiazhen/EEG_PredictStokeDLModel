from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "revised_initial"

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "roc_auc",
    "pr_auc",
    "brier_score",
]

DISPLAY_COLUMNS = [
    "模型类型",
    "模型",
    "输入/设置",
    "汇总口径",
    "Accuracy",
    "Balanced accuracy",
    "Sensitivity",
    "Specificity",
    "ROC-AUC",
    "PR-AUC",
    "Brier",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "brier_score": "Brier",
}

MODEL_NAME_MAP = {
    "clinical_only_logistic_l1": "Clinical Logistic L1",
    "clinical_only_logistic_l2": "Clinical Logistic L2",
    "clinical_only_svm_rbf": "Clinical SVM RBF",
    "clinical_only_random_forest": "Clinical Random forest",
    "clinical_only_gaussian_nb": "Clinical Gaussian NB",
    "clinical_only_knn": "Clinical KNN",
    "eeg_only_logistic_l1_none": "EEG Logistic L1",
    "eeg_only_logistic_l1_selectk100": "EEG Logistic L1",
    "eeg_only_logistic_l2_none": "EEG Logistic L2",
    "eeg_only_logistic_l2_selectk100": "EEG Logistic L2",
    "eeg_only_svm_linear_none": "EEG SVM linear",
    "eeg_only_svm_linear_selectk100": "EEG SVM linear",
    "eeg_only_svm_rbf_none": "EEG SVM RBF",
    "eeg_only_svm_rbf_selectk100": "EEG SVM RBF",
    "eeg_only_random_forest_none": "EEG Random forest",
    "eeg_only_random_forest_selectk100": "EEG Random forest",
    "eeg_only_gaussian_nb_none": "EEG Gaussian NB",
    "eeg_only_gaussian_nb_selectk100": "EEG Gaussian NB",
    "eeg_only_knn_none": "EEG KNN",
    "eeg_only_knn_selectk100": "EEG KNN",
    "eeg_clinical_logistic_l1_selectk100": "EEG+Clinical Logistic L1",
    "eeg_clinical_logistic_l2_selectk100": "EEG+Clinical Logistic L2",
    "eeg_clinical_svm_rbf_selectk100": "EEG+Clinical SVM RBF",
}

DEEP_MODEL_ROWS = {
    "b_no_ssl_cnn_same_arch_10seed": ("Deep EEG", "No-SSL CNN"),
    "c_patient_barlow_ssl_no_residual_heads": ("Deep EEG", "Barlow SSL-CNN"),
    "d_no_ssl_cnn_residual_heads": ("Deep EEG", "No-SSL residual-aware CNN"),
    "e_patient_barlow_ssl_residual_heads": ("Deep EEG", "Residual-aware SSL-CNN"),
}


def fmt_point(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def fmt_mean_std(mean: object, std: object) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(std):
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f} ± {float(std):.3f}"


def feature_setting(input_features: str, selector: str) -> str:
    feature_text = str(input_features)
    if "baseline clinical only" in feature_text:
        feature_text = "Clinical only"
    elif "PSD+WPLI EO+EC + baseline clinical" in feature_text:
        feature_text = "EEG PSD+WPLI + clinical"
    elif "PSD+WPLI EO+EC" in feature_text:
        feature_text = "EEG PSD+WPLI"

    selector = str(selector)
    if selector in {"none", "nan"}:
        return feature_text
    if selector == "selectk100":
        return f"{feature_text}; SelectK=100"
    return f"{feature_text}; {selector}"


def append_point_rows(rows: list[dict[str, str]], frame: pd.DataFrame, category: str) -> None:
    frame = frame[frame["status"].isin(["trained", "reference"])].copy()
    for _, row in frame.iterrows():
        output = {
            "模型类型": category,
            "模型": MODEL_NAME_MAP.get(str(row["model"]), str(row["model"])),
            "输入/设置": feature_setting(row.get("input_features", ""), row.get("feature_selection", "")),
            "汇总口径": "LOSO, n=19",
        }
        for metric in METRIC_COLUMNS:
            output[METRIC_LABELS[metric]] = fmt_point(row.get(metric, np.nan))
        rows.append(output)


def append_deep_rows(rows: list[dict[str, str]], core: pd.DataFrame) -> None:
    core = core[core["availability"].eq("available")].copy()
    for contrast, (category, display_name) in DEEP_MODEL_ROWS.items():
        mean_rows = core[(core["contrast"].eq(contrast)) & (core["row_type"].eq("mean"))]
        std_rows = core[(core["contrast"].eq(contrast)) & (core["row_type"].eq("std"))]
        if mean_rows.empty:
            continue
        mean_row = mean_rows.iloc[0]
        std_row = std_rows.iloc[0] if not std_rows.empty else pd.Series(dtype=object)
        output = {
            "模型类型": category,
            "模型": display_name,
            "输入/设置": "EEG PSD+WPLI; gated CNN",
            "汇总口径": "10-seed mean ± SD",
        }
        for metric in METRIC_COLUMNS:
            output[METRIC_LABELS[metric]] = fmt_mean_std(mean_row.get(metric, np.nan), std_row.get(metric, np.nan))
        rows.append(output)


def build_table() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    incremental = pd.read_csv(METRICS_DIR / "eeg_clinical_incremental_model_comparison.csv")
    core = pd.read_csv(METRICS_DIR / "core_ablation_10seed_summary.csv")

    append_point_rows(rows, incremental[incremental["model_family"].eq("eeg_only")], "EEG-only ML")
    append_deep_rows(rows, core)

    table = pd.DataFrame(rows, columns=DISPLAY_COLUMNS)
    sort_order = {
        "EEG-only ML": 0,
        "Deep EEG": 1,
    }
    table["_sort"] = table["模型类型"].map(sort_order)
    table = table.sort_values(["_sort", "模型", "输入/设置"]).drop(columns="_sort").reset_index(drop=True)
    return table


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "| " + " | ".join(DISPLAY_COLUMNS) + " |",
        "|---" * len(DISPLAY_COLUMNS) + "|",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in DISPLAY_COLUMNS) + " |")
    lines.extend(
        [
            "",
            "注：传统 EEG-only ML 结果为 19 例监督队列的 patient-level LOSO 点估计；Deep EEG 结果为标准 10 seeds（0,1,2,3,4,5,7,13,21,42）的均值 ± 标准差。Brier 分数越低越好，其余指标越高越好。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def choose_font() -> FontProperties:
    for candidate in [Path(r"C:/Windows/Fonts/msyh.ttc"), Path(r"C:/Windows/Fonts/simhei.ttf")]:
        if candidate.exists():
            return FontProperties(fname=str(candidate))
    return FontProperties()


def wrap(value: object, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False))


def draw_table(table: pd.DataFrame, path_base: Path) -> None:
    font = choose_font()
    title_font = font.copy()
    title_font.set_size(15)
    title_font.set_weight("bold")
    header_font = font.copy()
    header_font.set_size(8.8)
    header_font.set_weight("bold")
    body_font = font.copy()
    body_font.set_size(7.8)
    note_font = font.copy()
    note_font.set_size(7.4)

    n_rows = len(table)
    fig, ax = plt.subplots(figsize=(18, 1.65 + 0.46 * n_rows))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, right = 0.02, 0.985
    top = 0.93
    row_h = 0.031
    col_widths = np.array([0.13, 0.18, 0.22, 0.12, 0.07, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07])
    col_widths = col_widths / col_widths.sum() * (right - left)
    col_lefts = np.r_[left, left + np.cumsum(col_widths[:-1])]
    col_centers = col_lefts + col_widths / 2

    ax.text(0.5, 0.985, "EEG model performance summary", ha="center", va="top", fontproperties=title_font)
    ax.hlines([top, top - 0.008, top - 0.052], left, right, colors="#202020", linewidths=[1.5, 0.7, 1.0])

    header_y = top - 0.030
    for idx, col in enumerate(DISPLAY_COLUMNS):
        ha = "left" if idx < 3 else "center"
        x = col_lefts[idx] + 0.003 if idx < 3 else col_centers[idx]
        ax.text(x, header_y, wrap(col, 16), ha=ha, va="center", fontproperties=header_font)

    y = top - 0.075
    last_group = None
    for row_idx, row in table.iterrows():
        group = row["模型类型"]
        if group != last_group and last_group is not None:
            ax.hlines(y + row_h * 0.50, left, right, colors="#D6DCE4", linewidth=0.8)
        if row_idx % 2 == 0:
            ax.add_patch(plt.Rectangle((left, y - row_h * 0.48), right - left, row_h * 0.92, color="#FAFBFD", zorder=0))
        for idx, col in enumerate(DISPLAY_COLUMNS):
            value = row[col]
            if idx == 0:
                value = group if group != last_group else ""
            width = 18 if idx == 1 else 24 if idx == 2 else 16
            ha = "left" if idx < 3 else "center"
            x = col_lefts[idx] + 0.003 if idx < 3 else col_centers[idx]
            ax.text(x, y, wrap(value, width), ha=ha, va="center", fontproperties=body_font)
        y -= row_h
        last_group = group

    bottom = y + row_h * 0.20
    ax.hlines(bottom, left, right, colors="#202020", linewidth=1.4)
    note = (
        "Note: traditional EEG-only ML rows are patient-level LOSO point estimates in the 19 labelled patients. "
        "Deep EEG rows are mean ± SD across seeds 0,1,2,3,4,5,7,13,21,42. Brier is lower-is-better; other metrics are higher-is-better."
    )
    ax.text(left, bottom - 0.022, note, ha="left", va="top", fontproperties=note_font, color="#333333")

    for ext in [".png", ".svg", ".pdf", ".tiff"]:
        output = path_base.with_suffix(ext)
        if ext in {".png", ".tiff"}:
            fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    csv_path = TABLE_DIR / "all_machine_learning_results_ppt_table.csv"
    tsv_path = TABLE_DIR / "all_machine_learning_results_ppt_table.tsv"
    md_path = TABLE_DIR / "all_machine_learning_results_ppt_table.md"
    fig_base = FIGURE_DIR / "all_machine_learning_results_ppt_table"

    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    table.to_csv(tsv_path, index=False, sep="\t", encoding="utf-8-sig")
    write_markdown(table, md_path)
    draw_table(table, fig_base)
    print(f"Wrote {csv_path}")
    print(f"Wrote {tsv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {fig_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
