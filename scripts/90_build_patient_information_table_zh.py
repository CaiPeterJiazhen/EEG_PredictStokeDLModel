from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLINICAL_XLSX = Path(r"F:/CJZFile/EEG_M1/M1组病历记录表.xlsx")
INTEGRITY_XLSX = Path(r"F:/CJZFile/EEG_M1/19例患者脑电数据完整性检查.xlsx")
EEG_ROOT = Path(r"F:/CJZFile/EEG_M1/Patient_tACS_M1_RestingStateEEG_afterProcess")
LABEL_SOURCE_CSV = PROJECT_ROOT / "results" / "explainability" / "explained_predictions.csv"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "revised_initial"


def normalize_subject_id(value: object) -> str | None:
    match = re.search(r"sub\s*0*(\d+)", str(value), flags=re.IGNORECASE)
    if not match:
        return None
    return f"sub{int(match.group(1)):02d}"


def extract_number(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else float("nan")


def parse_duration_months(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    raw = str(value).strip()
    if not raw:
        return float("nan")
    if "十余年" in raw:
        return 10.0 * 12.0
    numeric = extract_number(raw)
    if not math.isfinite(numeric):
        return float("nan")
    if "天" in raw:
        return numeric / 30.0
    if "年" in raw:
        return numeric * 12.0
    if "月" in raw:
        return numeric
    return numeric / 30.0


def parse_affected_side_bbt(value: object, side: object) -> float:
    if pd.isna(value) or pd.isna(side):
        return float("nan")
    raw = str(value)
    left = re.search(r"左\s*([-+]?\d+(?:\.\d+)?)", raw)
    right = re.search(r"右\s*([-+]?\d+(?:\.\d+)?)", raw)
    side_text = str(side)
    if "左" in side_text and left:
        return float(left.group(1))
    if "右" in side_text and right:
        return float(right.group(1))
    return float("nan")


def discover_eeg_subjects(root: Path) -> set[str]:
    subjects: set[str] = set()
    for path in root.rglob("*"):
        subject_id = normalize_subject_id(path)
        if subject_id:
            subjects.add(subject_id)
    return subjects


def read_supervised_subjects(path: Path) -> set[str]:
    frame = pd.read_excel(path)
    if "患者ID" not in frame.columns:
        raise ValueError(f"Cannot find 患者ID in {path}")
    return {sid for sid in frame["患者ID"].map(normalize_subject_id).dropna()}


def read_label_source(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"subject_id", "y_true", "residual", "signed_distance"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    labels = labels.loc[:, ["subject_id", "y_true", "residual", "signed_distance"]].drop_duplicates().copy()
    labels["subject_id"] = labels["subject_id"].map(normalize_subject_id)
    labels = labels[labels["subject_id"].notna()].copy()
    label_counts = labels.groupby("subject_id")["y_true"].nunique()
    if (label_counts > 1).any():
        conflicted = label_counts[label_counts > 1].index.tolist()
        raise ValueError(f"Conflicting labels for subjects: {conflicted}")
    return labels.drop_duplicates("subject_id")


def read_clinical_table() -> pd.DataFrame:
    frame = pd.read_excel(CLINICAL_XLSX)
    frame["subject_id"] = frame["编号"].map(normalize_subject_id)
    frame = frame[frame["subject_id"].notna()].copy()
    frame["age"] = frame["年龄"].map(extract_number)
    frame["duration_months"] = frame["病程"].map(parse_duration_months)
    frame["is_female"] = frame["性别"].astype(str).str.contains("女", na=False)
    frame["is_left_affected"] = frame["患病侧"].astype(str).str.contains("左", na=False)

    numeric_columns = {
        "FMA_pre": "治疗前FMA",
        "FMA_post": "治疗后FMA",
        "MBI_pre": "治疗前MBI",
        "MBI_post": "治疗后MBI",
        "MMSE": "MMSE",
    }
    for out_col, in_col in numeric_columns.items():
        frame[out_col] = pd.to_numeric(frame[in_col].map(extract_number), errors="coerce")

    frame["Delta_FMA"] = frame["FMA_post"] - frame["FMA_pre"]
    frame["Delta_MBI"] = frame["MBI_post"] - frame["MBI_pre"]
    frame["BBT_pre_affected"] = [
        parse_affected_side_bbt(value, side)
        for value, side in zip(frame["治疗前BBT"], frame["患病侧"], strict=False)
    ]
    frame["BBT_post_affected"] = [
        parse_affected_side_bbt(value, side)
        for value, side in zip(frame["治疗后BBT"], frame["患病侧"], strict=False)
    ]
    frame["Delta_BBT_affected"] = frame["BBT_post_affected"] - frame["BBT_pre_affected"]

    supervised = read_supervised_subjects(INTEGRITY_XLSX)
    eeg_subjects = discover_eeg_subjects(EEG_ROOT)
    frame["is_eeg_indexed"] = frame["subject_id"].isin(eeg_subjects)
    frame["is_supervised"] = frame["subject_id"].isin(supervised)
    frame["is_ssl_extra"] = frame["is_eeg_indexed"] & ~frame["is_supervised"]
    frame["complete_post_fma"] = frame["FMA_pre"].notna() & frame["FMA_post"].notna()
    frame["complete_supervised_label"] = frame["is_supervised"]
    labels = read_label_source(LABEL_SOURCE_CSV)
    frame = frame.merge(labels, on="subject_id", how="left")
    frame["is_proportional_recovery"] = frame["y_true"].eq(1)
    frame["is_poor_recovery"] = frame["y_true"].eq(0)
    return frame


def format_p(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if value < 0.001:
        return "<0.001"
    if value > 0.999:
        return "1"
    if value < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def continuous_p(a: pd.Series, b: pd.Series) -> tuple[str, str]:
    a_values = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    b_values = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    if len(a_values) < 2 or len(b_values) < 2:
        return "", "not tested: n<2 in at least one group"
    normal_a = len(a_values) >= 3 and stats.shapiro(a_values).pvalue >= 0.05
    normal_b = len(b_values) >= 3 and stats.shapiro(b_values).pvalue >= 0.05
    if normal_a and normal_b:
        p_value = float(stats.ttest_ind(a_values, b_values, equal_var=False).pvalue)
        return format_p(p_value), "Welch t-test"
    p_value = float(stats.mannwhitneyu(a_values, b_values, alternative="two-sided").pvalue)
    return format_p(p_value), "Mann-Whitney U"


def categorical_p(a: pd.Series, b: pd.Series) -> tuple[str, str]:
    a_mask = a.fillna(False).astype(bool)
    b_mask = b.fillna(False).astype(bool)
    table = [
        [int(a_mask.sum()), int((~a_mask).sum())],
        [int(b_mask.sum()), int((~b_mask).sum())],
    ]
    p_value = float(stats.fisher_exact(table).pvalue)
    return format_p(p_value), "Fisher exact test"


def mean_sd(values: pd.Series, denominator: int) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    n = len(numeric)
    if n == 0:
        return "NA"
    suffix = f" (n={n})" if n < denominator else ""
    if n == 1:
        return f"{numeric.iloc[0]:.2f}{suffix or ' (n=1)'}"
    return f"{numeric.mean():.2f} ± {numeric.std(ddof=1):.2f}{suffix}"


def n_percent(mask: pd.Series, denominator: int) -> str:
    count = int(mask.fillna(False).astype(bool).sum())
    return f"{count} ({100.0 * count / denominator:.1f}%)"


def build_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = read_clinical_table()
    groups = {
        "全部临床记录": clinical,
        "比例恢复组": clinical[clinical["is_proportional_recovery"]],
        "恢复不良组": clinical[clinical["is_poor_recovery"]],
    }
    positive = groups["比例恢复组"]
    negative = groups["恢复不良组"]

    rows: list[dict[str, str]] = []
    details: list[dict[str, str]] = []

    def add_section(label: str) -> None:
        rows.append({"指标": label, "全部临床记录": "", "比例恢复组": "", "恢复不良组": "", "P值": ""})

    def add_count(label: str) -> None:
        rows.append(
            {
                "指标": label,
                "全部临床记录": str(len(groups["全部临床记录"])),
                "比例恢复组": str(len(groups["比例恢复组"])),
                "恢复不良组": str(len(groups["恢复不良组"])),
                "P值": "",
            }
        )

    def add_binary(label: str, column: str, with_p: bool = True) -> None:
        p_text, method = categorical_p(positive[column], negative[column]) if with_p else ("", "")
        rows.append(
            {
                "指标": label,
                "全部临床记录": n_percent(groups["全部临床记录"][column], len(groups["全部临床记录"])),
                "比例恢复组": n_percent(groups["比例恢复组"][column], len(groups["比例恢复组"])),
                "恢复不良组": n_percent(groups["恢复不良组"][column], len(groups["恢复不良组"])),
                "P值": p_text,
            }
        )
        details.append({"指标": label, "检验方法": method, "变量": column})

    def add_continuous(label: str, column: str) -> None:
        p_text, method = continuous_p(positive[column], negative[column])
        rows.append(
            {
                "指标": label,
                "全部临床记录": mean_sd(groups["全部临床记录"][column], len(groups["全部临床记录"])),
                "比例恢复组": mean_sd(groups["比例恢复组"][column], len(groups["比例恢复组"])),
                "恢复不良组": mean_sd(groups["恢复不良组"][column], len(groups["恢复不良组"])),
                "P值": p_text,
            }
        )
        details.append({"指标": label, "检验方法": method, "变量": column})

    add_count("受试者数")
    add_section("人口学资料")
    add_binary("女性，n (%)", "is_female")
    add_continuous("年龄，岁", "age")
    add_continuous("病程，月", "duration_months")

    add_section("临床评估")
    add_binary("患侧为左手，n (%)", "is_left_affected")
    add_continuous("FMA-UE，治疗前", "FMA_pre")
    add_continuous("FMA-UE，14次治疗后", "FMA_post")
    add_continuous("FMA-UE 改变量", "Delta_FMA")
    add_continuous("比例恢复残差", "residual")
    add_continuous("MBI，治疗前", "MBI_pre")
    add_continuous("MBI，14次治疗后", "MBI_post")
    add_continuous("MBI 改变量", "Delta_MBI")
    add_continuous("BBT 患侧手，治疗前", "BBT_pre_affected")
    add_continuous("BBT 患侧手，14次治疗后", "BBT_post_affected")
    add_continuous("BBT 患侧手改变量", "Delta_BBT_affected")
    add_continuous("MMSE", "MMSE")

    add_section("数据可用性")
    add_binary("基线静息态 EEG 可用，n (%)", "is_eeg_indexed", with_p=False)
    add_binary("完整治疗后 FMA-UE，n (%)", "complete_post_fma")
    add_binary("可构建监督标签，n (%)", "complete_supervised_label", with_p=False)

    table = pd.DataFrame(rows, columns=["指标", "全部临床记录", "比例恢复组", "恢复不良组", "P值"])
    detail = pd.DataFrame(details)
    return table, detail


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "| 指标 | 全部临床记录 | 比例恢复组 | 恢复不良组 | P值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        if not any(str(row[col]).strip() for col in ["全部临床记录", "比例恢复组", "恢复不良组", "P值"]):
            lines.append(f"| **{row['指标']}** |  |  |  |  |")
        else:
            lines.append(
                f"| {row['指标']} | {row['全部临床记录']} | {row['比例恢复组']} | {row['恢复不良组']} | {row['P值']} |"
            )
    lines.extend(
        [
            "",
            "注：连续变量以均值 ± 标准差表示；当该指标存在缺失时，括号内标注可用记录数。P值为比例恢复组与恢复不良组比较；连续变量根据Shapiro-Wilk正态性检验结果采用Welch t检验或Mann-Whitney U检验，分类变量采用Fisher精确检验。比例恢复组与恢复不良组由监督队列的比例恢复标签定义；`十余年`病程按10年进行保守换算。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def choose_font() -> FontProperties:
    candidates = [
        Path(r"C:/Windows/Fonts/msyh.ttc"),
        Path(r"C:/Windows/Fonts/simhei.ttf"),
        Path(r"C:/Windows/Fonts/simsun.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return FontProperties(fname=str(candidate))
    return FontProperties()


def wrap_cell(text: object, width: int) -> str:
    raw = str(text)
    if len(raw) <= width:
        return raw
    return "\n".join(textwrap.wrap(raw, width=width, break_long_words=True, replace_whitespace=False))


def draw_three_line_table(table: pd.DataFrame, path_base: Path) -> None:
    font = choose_font()
    title_font = font.copy()
    title_font.set_size(15)
    title_font.set_weight("bold")
    header_font = font.copy()
    header_font.set_size(10.5)
    header_font.set_weight("bold")
    body_font = font.copy()
    body_font.set_size(9.5)
    section_font = font.copy()
    section_font.set_size(10)
    section_font.set_weight("bold")
    note_font = font.copy()
    note_font.set_size(8.2)

    n_rows = len(table)
    fig_height = 1.55 + 0.46 * n_rows
    fig, ax = plt.subplots(figsize=(12.8, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.972, "TABLE 1  患者信息", ha="center", va="top", fontproperties=title_font)
    left, right = 0.035, 0.965
    top = 0.91
    header_y = top - 0.035
    row_h = 0.034
    col_widths = np.array([0.34, 0.21, 0.19, 0.19, 0.07])
    col_widths = col_widths / col_widths.sum() * (right - left)
    col_lefts = np.r_[left, left + np.cumsum(col_widths[:-1])]
    col_centers = col_lefts + col_widths / 2

    ax.hlines([top, top - 0.01, top - 0.065], left, right, colors="#222222", linewidths=[1.6, 0.8, 1.2])
    headers = ["指标", "全部临床记录", "比例恢复组", "恢复不良组", "P值"]
    for idx, header in enumerate(headers):
        ha = "left" if idx == 0 else "center"
        x = col_lefts[idx] + 0.004 if idx == 0 else col_centers[idx]
        ax.text(x, header_y, header, ha=ha, va="center", fontproperties=header_font, color="#111111")

    y = top - 0.093
    for row_idx, row in table.iterrows():
        is_section = not any(str(row[col]).strip() for col in headers[1:])
        if is_section:
            ax.add_patch(plt.Rectangle((left, y - row_h * 0.52), right - left, row_h * 0.9, color="#F3F5F8", zorder=0))
            ax.text(col_lefts[0] + 0.004, y, str(row["指标"]), ha="left", va="center", fontproperties=section_font)
        else:
            if row_idx % 2 == 0:
                ax.add_patch(
                    plt.Rectangle((left, y - row_h * 0.52), right - left, row_h * 0.9, color="#FAFBFC", zorder=0)
                )
            ax.text(col_lefts[0] + 0.004, y, wrap_cell(row["指标"], 19), ha="left", va="center", fontproperties=body_font)
            for idx, column in enumerate(headers[1:], start=1):
                ax.text(
                    col_centers[idx],
                    y,
                    wrap_cell(row[column], 32 if idx < 4 else 6),
                    ha="center",
                    va="center",
                    fontproperties=body_font,
                )
        y -= row_h

    bottom_line = y + row_h * 0.28
    ax.hlines(bottom_line, left, right, colors="#222222", linewidth=1.6)

    note = (
        "注：连续变量以均值 ± 标准差表示；括号 n 为可用记录数。P值比较比例恢复组与恢复不良组；"
        "连续变量按正态性采用 Welch t 检验或 Mann-Whitney U 检验，分类变量采用 Fisher 精确检验。"
        "比例恢复组与恢复不良组由监督队列的比例恢复标签定义；“十余年”病程按 10 年保守换算。"
    )
    wrapped_note = "\n".join(textwrap.wrap(note, width=120, break_long_words=False))
    ax.text(left, bottom_line - 0.026, wrapped_note, ha="left", va="top", fontproperties=note_font, color="#333333")

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
    table, detail = build_table()
    table_path = TABLE_DIR / "table1_patient_information_zh.csv"
    detail_path = TABLE_DIR / "table1_patient_information_zh_stats_details.csv"
    markdown_path = TABLE_DIR / "table1_patient_information_zh.md"
    figure_base = FIGURE_DIR / "table1_patient_information_zh"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    write_markdown(table, markdown_path)
    draw_three_line_table(table, figure_base)
    print(f"Wrote {table_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {figure_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
