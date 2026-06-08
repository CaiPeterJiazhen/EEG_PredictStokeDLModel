from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "results" / "metrics" / "eeg_clinical_incremental_model_comparison.csv"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "revised_initial"

DISPLAY_COLUMNS = [
    "模型",
    "Accuracy",
    "Balanced accuracy",
    "Sensitivity",
    "Specificity",
    "ROC-AUC",
    "PR-AUC",
    "Brier",
]

METRIC_COLUMNS = {
    "Accuracy": "accuracy",
    "Balanced accuracy": "balanced_accuracy",
    "Sensitivity": "sensitivity",
    "Specificity": "specificity",
    "ROC-AUC": "roc_auc",
    "PR-AUC": "pr_auc",
    "Brier": "brier_score",
}

MODEL_LABELS = {
    "eeg_only_gaussian_nb": "Gaussian NB",
    "eeg_only_knn": "KNN",
    "eeg_only_logistic_l1": "Logistic L1",
    "eeg_only_logistic_l2": "Logistic L2",
    "eeg_only_random_forest": "Random forest",
    "eeg_only_svm_linear": "SVM linear",
    "eeg_only_svm_rbf": "SVM RBF",
}


def format_metric(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def model_label(model: str, selector: str) -> str:
    suffix = "SelectK=100" if selector == "selectk100" else "No selector"
    base = model
    for prefix, label in MODEL_LABELS.items():
        if model.startswith(prefix):
            base = label
            break
    return f"{base} ({suffix})"


def build_table() -> pd.DataFrame:
    source = pd.read_csv(SOURCE)
    source = source[(source["model_family"].eq("eeg_only")) & (source["status"].isin(["trained", "reference"]))].copy()
    rows: list[dict[str, str]] = []
    for _, row in source.iterrows():
        out = {"模型": model_label(str(row["model"]), str(row["feature_selection"]))}
        for display, source_col in METRIC_COLUMNS.items():
            out[display] = format_metric(row[source_col])
        rows.append(out)
    table = pd.DataFrame(rows, columns=DISPLAY_COLUMNS)
    order = {
        "Logistic L1": 0,
        "Logistic L2": 1,
        "SVM linear": 2,
        "SVM RBF": 3,
        "Random forest": 4,
        "Gaussian NB": 5,
        "KNN": 6,
    }
    selector_order = {"No selector": 0, "SelectK=100": 1}
    table["_model_order"] = table["模型"].map(lambda value: order.get(value.split(" (")[0], 99))
    table["_selector_order"] = table["模型"].map(
        lambda value: selector_order.get(value.rsplit("(", 1)[-1].rstrip(")"), 99)
    )
    return table.sort_values(["_model_order", "_selector_order"]).drop(columns=["_model_order", "_selector_order"])


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "| " + " | ".join(DISPLAY_COLUMNS) + " |",
        "|---" + "|---:" * (len(DISPLAY_COLUMNS) - 1) + "|",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in DISPLAY_COLUMNS) + " |")
    lines.extend(
        [
            "",
            "注：所有结果均为 19 例监督队列的 patient-level LOSO 点估计；Brier 越低越好，其余指标越高越好。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(table: pd.DataFrame, path: Path) -> None:
    rows_html = []
    for _, row in table.iterrows():
        cells = "".join(f"<td>{row[col]}</td>" for col in DISPLAY_COLUMNS)
        rows_html.append(f"<tr>{cells}</tr>")
    header = "".join(f"<th>{col}</th>" for col in DISPLAY_COLUMNS)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>EEG-only ML performance three-line table</title>
<style>
body {{
  font-family: "Times New Roman", "Microsoft YaHei", "SimSun", serif;
  margin: 28px;
  color: #111;
}}
table.three-line {{
  border-collapse: collapse;
  border-top: 2.2px solid #111;
  border-bottom: 2.2px solid #111;
  font-size: 16px;
  min-width: 980px;
}}
table.three-line thead tr {{
  border-bottom: 1.4px solid #111;
}}
table.three-line th, table.three-line td {{
  border: none;
  padding: 8px 14px;
  text-align: center;
  white-space: nowrap;
}}
table.three-line th:first-child, table.three-line td:first-child {{
  text-align: left;
}}
.note {{
  margin-top: 10px;
  font-size: 14px;
}}
</style>
</head>
<body>
<table class="three-line">
<thead><tr>{header}</tr></thead>
<tbody>
{chr(10).join(rows_html)}
</tbody>
</table>
<div class="note">注：所有结果均为 19 例监督队列的 patient-level LOSO 点估计；Brier 越低越好，其余指标越高越好。</div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def choose_font() -> FontProperties:
    for candidate in [Path(r"C:/Windows/Fonts/msyh.ttc"), Path(r"C:/Windows/Fonts/simhei.ttf")]:
        if candidate.exists():
            return FontProperties(fname=str(candidate))
    return FontProperties()


def draw_three_line_table(table: pd.DataFrame, path_base: Path) -> None:
    font = choose_font()
    title_font = font.copy()
    title_font.set_size(15)
    title_font.set_weight("bold")
    header_font = font.copy()
    header_font.set_size(9.5)
    header_font.set_weight("bold")
    body_font = font.copy()
    body_font.set_size(9)
    note_font = font.copy()
    note_font.set_size(8)

    fig, ax = plt.subplots(figsize=(12.8, 1.4 + 0.36 * len(table)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, right = 0.035, 0.97
    top = 0.90
    row_h = 0.048
    col_widths = np.array([0.29, 0.10, 0.14, 0.10, 0.10, 0.09, 0.09, 0.09])
    col_widths = col_widths / col_widths.sum() * (right - left)
    col_lefts = np.r_[left, left + np.cumsum(col_widths[:-1])]
    col_centers = col_lefts + col_widths / 2

    ax.text(0.5, 0.975, "EEG-only machine-learning performance", ha="center", va="top", fontproperties=title_font)
    ax.hlines([top, top - 0.01, top - 0.060], left, right, colors="#202020", linewidths=[1.5, 0.7, 1.1])

    for idx, col in enumerate(DISPLAY_COLUMNS):
        ha = "left" if idx == 0 else "center"
        x = col_lefts[idx] + 0.004 if idx == 0 else col_centers[idx]
        ax.text(x, top - 0.036, col, ha=ha, va="center", fontproperties=header_font)

    y = top - 0.087
    for row_idx, row in table.iterrows():
        if row_idx % 2 == 0:
            ax.add_patch(plt.Rectangle((left, y - row_h * 0.43), right - left, row_h * 0.86, color="#FAFBFD", zorder=0))
        for idx, col in enumerate(DISPLAY_COLUMNS):
            ha = "left" if idx == 0 else "center"
            x = col_lefts[idx] + 0.004 if idx == 0 else col_centers[idx]
            ax.text(x, y, str(row[col]), ha=ha, va="center", fontproperties=body_font)
        y -= row_h

    bottom = y + row_h * 0.26
    ax.hlines(bottom, left, right, colors="#202020", linewidth=1.5)
    ax.text(
        left,
        bottom - 0.028,
        "注：所有结果均为 19 例监督队列的 patient-level LOSO 点估计；Brier 越低越好，其余指标越高越好。",
        ha="left",
        va="top",
        fontproperties=note_font,
        color="#333333",
    )

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
    stem = "eeg_only_ml_three_line_table"
    table.to_csv(TABLE_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    table.to_csv(TABLE_DIR / f"{stem}.tsv", index=False, sep="\t", encoding="utf-8-sig")
    write_markdown(table, TABLE_DIR / f"{stem}.md")
    write_html(table, TABLE_DIR / f"{stem}.html")
    draw_three_line_table(table, FIGURE_DIR / stem)
    print(f"Wrote {TABLE_DIR / f'{stem}.tsv'}")
    print(f"Wrote {TABLE_DIR / f'{stem}.html'}")
    print(f"Wrote {FIGURE_DIR / f'{stem}.png'}")


if __name__ == "__main__":
    main()
