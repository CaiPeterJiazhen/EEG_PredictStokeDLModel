from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = Path(r"C:\Users\HPGZZ\Desktop\预后模型相关文献")
WHITE_PDF = REF_DIR / (
    "White 等 - 2024 - Predicting recovery following stroke Deep learning, "
    "multimodal data and feature selection using exp.pdf"
)
WHITE_OUT = ROOT / "docs" / "white_2024_nature_reader"
FIG_OUT = ROOT / "results" / "figures" / "revised_initial"
METRIC_OUT = ROOT / "results" / "metrics"
SEEDS = [0, 1, 2, 3, 4, 5, 7, 13, 21, 42]


def translate_caption_stub(text: str) -> str:
    """Conservative Chinese caption summary for reader use."""
    lower = text.lower()
    if "stitched mri" in lower or "roi" in lower:
        return "展示不同影像输入构造方式，用于说明模型输入和特征选择方案。"
    if "clear" in lower:
        return "展示可解释人工智能输出如何将模型概率与关键特征或图像区域对应。"
    if "validation loss" in lower or "balanced test accuracy" in lower:
        return "展示特征数量、验证损失和测试准确率之间的关系，用于支持模型选择。"
    if text.strip().lower().startswith("table"):
        return "汇总模型性能或特征选择结果，用于支撑结果段落中的定量比较。"
    return "该图表用于连接方法设计、模型比较和可解释性论证。"


def extract_white_reader() -> None:
    WHITE_OUT.mkdir(parents=True, exist_ok=True)
    assets = WHITE_OUT / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(WHITE_PDF)
    source_blocks: list[dict] = []
    captions: list[dict] = []
    sections: list[dict] = []

    section_re = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+(.+)$")
    caption_re = re.compile(r"^\s*(Fig\.|Table)\s+([A-Za-z0-9.]+)", re.IGNORECASE)

    for page_index, page in enumerate(doc, start=1):
        page_path = assets / f"page_{page_index:02d}.png"
        if not page_path.exists():
            pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
            pix.save(page_path)

        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
        for block_index, block in enumerate(blocks, start=1):
            text = re.sub(r"\s+", " ", block[4]).strip()
            if len(text) < 30:
                continue
            block_id = f"W{len(source_blocks) + 1:03d}"
            item = {
                "id": block_id,
                "page": page_index,
                "block_index": block_index,
                "bbox": [round(float(x), 2) for x in block[:4]],
                "text": text,
            }
            source_blocks.append(item)
            if caption_re.match(text):
                cap = {
                    "id": f"C{len(captions) + 1:03d}",
                    "page": page_index,
                    "source_block_id": block_id,
                    "caption": text,
                    "caption_zh_summary": translate_caption_stub(text),
                }
                captions.append(cap)
            m = section_re.match(text)
            if m and len(text) < 140:
                sections.append(
                    {
                        "id": f"S{len(sections) + 1:03d}",
                        "page": page_index,
                        "number": m.group(1),
                        "title": m.group(2),
                        "source_block_id": block_id,
                    }
                )

    (WHITE_OUT / "source_map.json").write_text(
        json.dumps(
            {
                "source_pdf": str(WHITE_PDF),
                "page_count": doc.page_count,
                "sections": sections,
                "captions": captions,
                "blocks": source_blocks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    section_cards = [
        (
            "2.1 Dataset",
            "先交代队列来源、时间范围、纳入/排除和标签，再说明不同输入数据集的构造方式。"
            "对本文的启发是：研究对象小节不仅列样本量，还要解释哪些患者进入监督训练、哪些患者只能进入无标签 SSL 池，以及这些数据可用性如何影响模型问题。"
        ),
        (
            "2.2 CLEAR image explainable AI system",
            "方法部分只定义可解释性工具、输入和输出，不把解释结果提前写进去。"
            "对本文的启发是：可解释性方法应写成归因、遮挡、topomap 和 connectivity 的分析流程，解释性结论放在 Results。"
        ),
        (
            "3 Experiments and inference",
            "按模型家族逐一说明 baseline、轻量网络、深度网络、多模态模型和 DAFT，并为每个实验说明比较目的。"
            "对本文的启发是：传统 ML、CNN、SSL、残差感知和最终模型应分段写清楚各自解决什么问题，而不是只罗列模型名。"
        ),
        (
            "4 Results",
            "先报告模型比较，再报告特征选择与可解释性输出，表格和图件紧跟支撑的结论。"
            "对本文的启发是：最终模型表现应包含 ROC、混淆矩阵、模型指标对比和损失曲线，并在正文中逐图解释。"
        ),
        (
            "5 Discussion and future work",
            "围绕中心发现、相对已有方法的意义、可解释性发现、数据限制和后续验证展开。"
            "对本文的启发是：讨论要解释为什么 EEG-only、SSL 和残差感知各自有价值，同时明确小样本内部验证的边界。"
        ),
    ]

    captions_md = []
    for cap in captions:
        captions_md.append(
            f"- **{cap['id']} (p.{cap['page']})** {cap['caption']}\n"
            f"  - 中文用途：{cap['caption_zh_summary']}"
        )

    paper_md = [
        "# White et al. 2024 结构化阅读稿",
        "",
        f"Source PDF: `{WHITE_PDF}`",
        "",
        "## 阅读目的",
        "",
        "该文作为本文第二个写作参照，重点用于学习卒中恢复预测论文中 Methods、Experiments/Results、Explainable AI 和 Discussion 的组织方式。"
        "本文不复述全文，而是保留页码、章节和图表锚点，便于将参考论文的写法转化为当前 EEG-tACS 论文的正文结构。",
        "",
        "## 章节写作功能",
        "",
    ]
    for title, note in section_cards:
        paper_md.extend([f"### {title}", "", note, ""])

    paper_md.extend(
        [
            "## 主要图表锚点",
            "",
            "\n".join(captions_md) if captions_md else "未检测到图表标题，请检查 PDF 抽取结果。",
            "",
            "## 页面图像",
            "",
            "页面级图像位于 `assets/page_XX.png`，用于人工核对图表和正文位置。",
        ]
    )
    (WHITE_OUT / "paper.md").write_text("\n".join(paper_md), encoding="utf-8")

    blueprint = [
        "# Lin 2022 与 White 2024 对当前稿件的写作映射",
        "",
        "## 1. Materials and Methods",
        "",
        "- Lin 的研究对象写法以患者表格开场，随后交代伦理、纳入、治疗和评估时间点。当前稿件应把 Table 1 放在研究对象小节，明确 29 例临床源记录、19 例监督队列和 9 例额外 SSL 池的关系。",
        "- White 的方法部分按数据集、模型和解释系统拆分，每个小标题说明一个设计目的。当前稿件应把传统 ML、自监督学习、CNN+残差感知拆成独立小节。",
        "- Lin 和 White 都把解释方法写在 Methods，把解释结论写在 Results。当前稿件的可解释性方法应只写 integrated gradients、SmoothGrad、遮挡、PSD topomap 和 WPLI connectivity 如何计算。",
        "",
        "## 2. Results",
        "",
        "- Lin 的结果段落使用“表格/图件先支持分类性能，再支持关键因素”的证据顺序。当前稿件应在最终模型表现下依次解释 ROC、混淆矩阵、模型指标对比和损失曲线。",
        "- White 使用验证损失与测试准确率关系说明模型选择。当前稿件应新增最终模型训练/验证损失曲线，说明训练是否收敛、是否出现明显不稳定，以及辅助损失是否在训练后期趋稳。",
        "- 可解释性结果不能只放图。正文需要分别说明 PSD topomap 中 EO/EC 和频段的空间模式，以及 WPLI connectivity 中 beta 频段和额-中央/中央-顶连接的模式。",
        "",
        "## 3. Discussion",
        "",
        "- Lin 的讨论将深度模型性能、关键神经生理因素和泛化限制连接起来。当前稿件应围绕 EEG-only 预测、残差感知训练、Barlow SSL 的作用和可解释性候选生物标志物组织段落。",
        "- White 的讨论强调模型比较、特征选择和 Explainable AI 的意义。当前稿件应避免“项目说明式”表达，改成“结果支持/提示/需要外部验证”的论文语言。",
        "",
        "## 4. 为什么选择 Barlow SSL",
        "",
        "- Barlow Twins 不需要显式负样本或大型 memory queue，适合患者数较少且个体差异大的 EEG 队列。",
        "- 冗余降低损失同时约束同一患者两个增强视图的一致性和表征维度之间的去相关，降低小样本下表征坍塌风险。",
        "- 与重建式 SSL 相比，该目标更贴近患者层面判别表征，而不是重构高维噪声 EEG 输入。",
        "- 当前数据支持 Barlow SSL 作为纳入无完整监督标签 EEG 和稳定表征学习的组件；其平均性能增益需要更大队列验证。",
    ]
    (ROOT / "docs" / "reference_section_writing_blueprint_20260602.md").write_text(
        "\n".join(blueprint),
        encoding="utf-8",
    )

    notes = [
        "# Translation and extraction notes",
        "",
        "- The reader is structure-focused because the current manuscript task is to borrow section organization and figure-result writing patterns, not to reproduce the full article text.",
        "- Page-level PNGs were rendered for manual visual checking.",
        "- `source_map.json` preserves extracted blocks, section headings and figure/table captions with page anchors.",
    ]
    (WHITE_OUT / "translation_notes.md").write_text("\n".join(notes), encoding="utf-8")


def mean_sem(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    grouped = frame.groupby("epoch", as_index=False)[value].agg(["mean", "sem"]).reset_index()
    grouped["sem"] = grouped["sem"].fillna(0.0)
    return grouped


def build_loss_figure() -> None:
    rows = []
    for seed in SEEDS:
        path = (
            ROOT
            / "results"
            / "training_logs"
            / f"dl_loss_history_patient_barlow_residualaware_highrank_swa_seed{seed}.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["seed"] = seed
        rows.append(df)
    data = pd.concat(rows, ignore_index=True)

    for prefix in ["train", "val"]:
        data[f"{prefix}_weighted_bce"] = data[f"{prefix}_loss_bce"]
        data[f"{prefix}_weighted_residual"] = (
            data["lambda_reg"] * data[f"{prefix}_loss_residual_regression"]
        )
        data[f"{prefix}_weighted_rank"] = (
            data["lambda_rank"] * data[f"{prefix}_loss_pairwise_ranking"]
        )
        data[f"{prefix}_weighted_soft"] = data["lambda_soft"] * data[f"{prefix}_loss_soft_label"]

    # First average within seed/fold/epoch rows are already unique; then summarize across
    # all seed-fold training runs at each epoch.
    component_cols = [
        "train_loss_total",
        "val_loss_total",
        "train_weighted_bce",
        "val_weighted_bce",
        "train_weighted_residual",
        "val_weighted_residual",
        "train_weighted_rank",
        "val_weighted_rank",
        "train_weighted_soft",
        "val_weighted_soft",
    ]
    summary_parts = []
    for col in component_cols:
        s = mean_sem(data, col)
        s["metric"] = col
        summary_parts.append(s.rename(columns={"mean": "value_mean", "sem": "value_sem"}))
    summary = pd.concat(summary_parts, ignore_index=True)
    METRIC_OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(METRIC_OUT / "final_model_loss_curve_summary.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    colors = {
        "train_loss_total": "#1f77b4",
        "val_loss_total": "#ff7f0e",
        "bce": "#2b5c8a",
        "residual": "#d95f02",
        "ranking": "#7570b3",
        "soft": "#1b9e77",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)

    ax = axes[0]
    for metric, label, color in [
        ("train_loss_total", "Training total", colors["train_loss_total"]),
        ("val_loss_total", "Validation total", colors["val_loss_total"]),
    ]:
        s = summary[summary["metric"] == metric]
        x = s["epoch"].to_numpy(dtype=float)
        y = s["value_mean"].to_numpy(dtype=float)
        e = s["value_sem"].to_numpy(dtype=float)
        ax.plot(x, y, lw=1.6, color=color, label=label)
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.18, lw=0)
    ax.axvline(50, color="#777777", lw=0.8, ls="--")
    ax.text(51.5, ax.get_ylim()[1] * 0.92, "SWA", color="#555555", fontsize=6.5)
    ax.set_title("Total loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", fontsize=6.4)

    ax = axes[1]
    for train_metric, label, color_key in [
        ("train_weighted_bce", "BCE", "bce"),
        ("train_weighted_residual", "Residual", "residual"),
        ("train_weighted_rank", "Ranking", "ranking"),
        ("train_weighted_soft", "Soft label", "soft"),
    ]:
        s = summary[summary["metric"] == train_metric]
        ax.plot(s["epoch"], s["value_mean"], lw=1.25, color=colors[color_key], label=label)
    ax.axvline(50, color="#777777", lw=0.8, ls="--")
    ax.set_title("Weighted training components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted contribution")
    ax.legend(loc="upper right", fontsize=6.2)

    ax = axes[2]
    for val_metric, label, color_key in [
        ("val_weighted_bce", "BCE", "bce"),
        ("val_weighted_residual", "Residual", "residual"),
        ("val_weighted_soft", "Soft label", "soft"),
    ]:
        s = summary[summary["metric"] == val_metric]
        ax.plot(s["epoch"], s["value_mean"], lw=1.25, color=colors[color_key], label=label)
    ax.axvline(50, color="#777777", lw=0.8, ls="--")
    ax.set_title("Weighted validation components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted contribution")
    ax.legend(loc="upper right", fontsize=6.2)

    for idx, ax in enumerate(axes):
        ax.text(
            -0.12,
            1.08,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
        )
        ax.grid(axis="y", color="#e5e5e5", lw=0.5)

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    base = FIG_OUT / "figure4d_final_model_loss_curves"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    extract_white_reader()
    build_loss_figure()


if __name__ == "__main__":
    main()
