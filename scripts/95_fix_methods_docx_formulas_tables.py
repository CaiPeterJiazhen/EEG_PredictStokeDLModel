from __future__ import annotations

import copy
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
INPUT_DOCX = Path(r"C:\Users\HPGZZ\Downloads\第二部分_材料与方法_学术论文风格重写稿_v2.docx")
OUTPUT_DOCX = ROOT / "output" / "doc" / "第二部分_材料与方法_学术论文风格重写稿_v2_公式与三线表.docx"

MATH_NS = {"m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}


FORMULA_REPLACEMENTS = {
    r"\mathrm{wPLI}_{xy}=\frac{\left|\mathbb{E}\left[\operatorname{Im}(S_{xy})\right]\right|}{\mathbb{E}\left[\left|\operatorname{Im}(S_{xy})\right|\right]}": r"\mathrm{wPLI}_{xy}=\frac{\left|\mathbb{E}\left[\operatorname{Im}(S_{xy})\right]\right|}{\mathbb{E}\left[\left|\operatorname{Im}(S_{xy})\right|\right]}",
    r"\frac{62\times 61}{2}=1891": r"\frac{62\times 61}{2}=1891",
    r"\Delta FMA_{\mathrm{pred}}=0.7\times(66-FMA_{\mathrm{pre}})": r"\Delta FMA_{\mathrm{pred}}=0.7\times(66-FMA_{\mathrm{pre}})",
    r"\Delta FMA_{\mathrm{obs}}=FMA_{\mathrm{post}}-FMA_{\mathrm{pre}}": r"\Delta FMA_{\mathrm{obs}}=FMA_{\mathrm{post}}-FMA_{\mathrm{pre}}",
    r"Residual=\Delta FMA_{\mathrm{pred}}-\Delta FMA_{\mathrm{obs}}": r"\mathrm{Residual}=\Delta FMA_{\mathrm{pred}}-\Delta FMA_{\mathrm{obs}}",
    r"d_i=1.5-Residual_i": r"d_i=1.5-\mathrm{Residual}_i",
    r"z_i=\frac{d_i-\mu_{d,\mathrm{train}}}{\sigma_{d,\mathrm{train}}+\epsilon}": r"z_i=\frac{d_i-\mu_{d,\mathrm{train}}}{\sigma_{d,\mathrm{train}}+\epsilon}",
    r"\tilde{y}_i=\sigma(d_i/\tau)": r"\tilde{y}_i=\sigma(d_i/\tau)",
    r"\mathcal{L}=\mathcal{L}_{BCE}(y,\hat{y})+\lambda_{reg}\mathcal{L}_{Huber}(z,\hat{z})+\lambda_{rank}\mathcal{L}_{rank}+\lambda_{soft}\mathcal{L}_{BCE}(\tilde{y},\hat{y})": r"\mathcal{L}=\mathcal{L}_{BCE}(y,\hat{y})+\lambda_{reg}\mathcal{L}_{Huber}(z,\hat{z})+\lambda_{rank}\mathcal{L}_{rank}+\lambda_{soft}\mathcal{L}_{BCE}(\tilde{y},\hat{y})",
    r"C_{ij}=\frac{\sum_b z^A_{b,i}z^B_{b,j}}{\sqrt{\sum_b (z^A_{b,i})^2}\sqrt{\sum_b (z^B_{b,j})^2}}": r"C_{ij}=\frac{\sum_b z^A_{b,i}z^B_{b,j}}{\sqrt{\sum_b (z^A_{b,i})^2}\sqrt{\sum_b (z^B_{b,j})^2}}",
    r"\mathcal{L}_{BT}=\sum_i(1-C_{ii})^2+\lambda_{BT}\sum_i\sum_{j\ne i}C_{ij}^2": r"\mathcal{L}_{BT}=\sum_i(1-C_{ii})^2+\lambda_{BT}\sum_i\sum_{j\ne i}C_{ij}^2",
    r"\operatorname{IG}_i(x)=(x_i-x_i^\prime)\int_0^1\frac{\partial F(x^\prime+\alpha(x-x^\prime))}{\partial x_i}\,d\alpha": r"\operatorname{IG}_i(x)=(x_i-x_i^\prime)\int_0^1\frac{\partial F(x^\prime+\alpha(x-x^\prime))}{\partial x_i}\,d\alpha",
    r"\operatorname{SG\text{-}IG}(x)=\frac{1}{K}\sum_{k=1}^{K}\operatorname{IG}(x+\epsilon_k),\quad \epsilon_k\sim\mathcal{N}(0,\sigma^2)": r"\operatorname{SG-IG}(x)=\frac{1}{K}\sum_{k=1}^{K}\operatorname{IG}(x+\epsilon_k),\quad \epsilon_k\sim\mathcal{N}(0,\sigma^2)",
    r"A^{\prime}_{n,j}=\frac{|A_{n,j}|}{\sum_j |A_{n,j}|+\epsilon}": r"A^{\prime}_{n,j}=\frac{|A_{n,j}|}{\sum_j |A_{n,j}|+\epsilon}",
    r"\bar{A}_j=\frac{1}{N}\sum_{n=1}^{N}A^{\prime}_{n,j}": r"\bar{A}_j=\frac{1}{N}\sum_{n=1}^{N}A^{\prime}_{n,j}",
}


INLINE_REPLACEMENTS = [
    (r"p = 3.81 × 10^{-6}", "p = 3.81 × 10⁻⁶"),
    (r"S_{xy}", "Sxy"),
    (r"Im(S_{xy})", "Im(Sxy)"),
    (r"\mathbb{E}[\cdot]", "E[·]"),
    (r"\mu_{d,\mathrm{train}}", "μd,train"),
    (r"\sigma_{d,\mathrm{train}}", "σd,train"),
    (r"d_i", "di"),
    (r"\tilde{y}_i", "ỹi"),
    (r"\sigma(\cdot)", "σ(·)"),
    (r"\tau", "τ"),
    (r"\hat{y}", "ŷ"),
    (r"\hat{z}", "ẑ"),
    (r"\lambda_{reg}", "λreg"),
    (r"\lambda_{rank}", "λrank"),
    (r"\lambda_{soft}", "λsoft"),
    (r"\lambda_{BT}", "λBT"),
    (r"Z^A", "Zᴬ"),
    (r"Z^B", "Zᴮ"),
    (r"FMA_{pre}", "FMApre"),
    (r"FMA_{post}", "FMApost"),
    (r"x^\prime", "x′"),
]


TABLE_1A = [
    ["指标", "全部患者", "比例恢复组", "恢复不良组", "P 值"],
    ["受试者数", "29", "10", "9", ""],
    ["年龄，岁", "64.03 ± 9.03", "64.00 ± 7.60", "66.11 ± 5.30", "0.49"],
    ["性别（女性），n（%）", "15（51.7%）", "6（60.0%）", "6（66.7%）", "1.00"],
    ["病程，月", "5.86 ± 22.47（n=28）", "0.97 ± 0.52", "1.27 ± 0.71", "0.29"],
    ["患侧为左手，n（%）", "14（48.3%）", "7（70.0%）", "4（44.4%）", "0.37"],
    ["治疗前 FMA-UE", "38.88 ± 23.72（n=25）", "59.60 ± 4.74", "19.22 ± 16.95", "<0.001"],
    ["治疗前 MBI", "60.00 ± 18.42（n=24）", "72.50 ± 13.59", "43.75 ± 15.29（n=8）", "<0.001"],
    ["MMSE", "27.67 ± 1.71（n=27）", "27.90 ± 1.60", "27.22 ± 2.11", "0.61"],
    ["基线 EEG 可用，n（%）", "28（96.6%）", "10（100.0%）", "9（100.0%）", ""],
]


TABLE_1B = [
    ["指标", "n", "治疗前", "治疗后", "改善量", "统计检验", "P 值"],
    ["FMA-UE", "19", "40.47 ± 23.83", "45.47 ± 23.23", "5.00 ± 4.07", "Wilcoxon signed-rank 检验", "<0.001"],
    ["MBI", "19", "55.53 ± 19.92", "76.84 ± 21.49", "21.32 ± 14.22", "配对 t 检验", "<0.001"],
]


def latex_to_omml_para(latex: str, work_dir: Path) -> etree._Element:
    md_path = work_dir / "formula.md"
    docx_path = work_dir / "formula.docx"
    md_path.write_text(f"$${latex}$$\n", encoding="utf-8")
    subprocess.run(
        ["pandoc", str(md_path), "-o", str(docx_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    xml = ZipFile(docx_path).read("word/document.xml")
    root = etree.fromstring(xml)
    omml = root.xpath("//m:oMathPara", namespaces=MATH_NS)
    if not omml:
        raise RuntimeError(f"Pandoc did not emit an OMML formula for: {latex}")
    element = copy.deepcopy(omml[0])
    for node in element.xpath(".//m:t", namespaces=MATH_NS):
        if node.text:
            node.text = node.text.replace("¡Á", "×")
    return element


def replace_paragraph_with_omml(paragraph, omml_para: etree._Element) -> None:
    p = paragraph._p
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    p.append(omml_para)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(4)


def set_paragraph_text(paragraph, text: str) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)
    run = paragraph.add_run(text)
    run.font.name = "宋体"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(10.5)


def normalize_inline_latex(doc: Document) -> None:
    formula_texts = set(FORMULA_REPLACEMENTS)
    for paragraph in doc.paragraphs:
        text = paragraph.text
        if not text or text.strip() in formula_texts:
            continue
        new_text = text
        for old, new in INLINE_REPLACEMENTS:
            new_text = new_text.replace(old, new)
        new_text = re.sub(r"10\^\{(-?\d+)\}", lambda m: "10" + to_superscript(m.group(1)), new_text)
        if new_text != text:
            set_paragraph_text(paragraph, new_text)


def to_superscript(text: str) -> str:
    table = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return text.translate(table)


def clear_borders(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        tag = "w:" + edge
        element = tc_borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            tc_borders.append(element)
        element.set(qn("w:val"), "nil")


def set_border(cell, edge: str, val: str = "single", size: str = "8", color: str = "000000") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    element = tc_borders.find(qn("w:" + edge))
    if element is None:
        element = OxmlElement("w:" + edge)
        tc_borders.append(element)
    element.set(qn("w:val"), val)
    element.set(qn("w:sz"), size)
    element.set(qn("w:space"), "0")
    element.set(qn("w:color"), color)


def set_cell_text(cell, text: str, bold: bool = False, align: WD_ALIGN_PARAGRAPH | None = None, font_size: float = 9.5) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = align or WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "宋体"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(font_size)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_cell_margins(cell, top: int = 90, start: int = 80, bottom: int = 90, end: int = 80) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_inches: float) -> None:
    width = int(width_inches * 1440)
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.first_child_found_in("w:tcW")
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width))
    tc_w.set(qn("w:type"), "dxa")


def style_three_line_table(table, widths: list[float], font_size: float = 9.5) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            clear_borders(cell)
            set_cell_margins(cell)
            if c_idx < len(widths):
                set_cell_width(cell, widths[c_idx])
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.name = "宋体"
                    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
                    run.font.size = Pt(font_size)
                    if r_idx == 0:
                        run.bold = True
            if c_idx == 0:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for cell in table.rows[0].cells:
        set_border(cell, "top", size="10")
        set_border(cell, "bottom", size="8")
    for cell in table.rows[-1].cells:
        set_border(cell, "bottom", size="10")


def make_table(doc: Document, data: list[list[str]], old_table, widths: list[float], font_size: float) -> None:
    new_table = doc.add_table(rows=len(data), cols=len(data[0]))
    for r_idx, row in enumerate(data):
        for c_idx, value in enumerate(row):
            align = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            set_cell_text(new_table.cell(r_idx, c_idx), value, bold=(r_idx == 0), align=align, font_size=font_size)
    style_three_line_table(new_table, widths, font_size=font_size)
    old_tbl = old_table._tbl
    old_tbl.addprevious(new_table._tbl)
    old_tbl.getparent().remove(old_tbl)


def update_captions(doc: Document) -> None:
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text == "表 1 患者人口学与基线临床特征":
            set_paragraph_text(paragraph, "表 1A 基线人口学与临床特征的分组比较")
            paragraph.runs[0].bold = True
        elif text == "表 2 监督学习队列治疗前后临床量表变化":
            set_paragraph_text(paragraph, "表 1B 监督学习队列治疗前后临床量表变化")
            paragraph.runs[0].bold = True


def convert_formula_paragraphs(doc: Document) -> int:
    converted = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        omml_cache: dict[str, etree._Element] = {}
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text not in FORMULA_REPLACEMENTS:
                continue
            latex = FORMULA_REPLACEMENTS[text]
            if latex not in omml_cache:
                omml_cache[latex] = latex_to_omml_para(latex, tmp_path)
            replace_paragraph_with_omml(paragraph, copy.deepcopy(omml_cache[latex]))
            converted += 1
    return converted


def main() -> None:
    if not INPUT_DOCX.exists():
        raise FileNotFoundError(INPUT_DOCX)
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc is required for LaTeX-to-Word-equation conversion")

    doc = Document(INPUT_DOCX)
    converted = convert_formula_paragraphs(doc)
    normalize_inline_latex(doc)
    update_captions(doc)

    if len(doc.tables) < 2:
        raise RuntimeError(f"Expected at least 2 tables, found {len(doc.tables)}")
    make_table(doc, TABLE_1A, doc.tables[0], [1.45, 1.35, 1.25, 1.25, 0.65], 8.8)
    make_table(doc, TABLE_1B, doc.tables[1], [0.75, 0.35, 1.0, 1.0, 1.0, 1.55, 0.6], 8.5)

    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_DOCX)
    print(f"converted_formula_paragraphs={converted}")
    print(f"saved={OUTPUT_DOCX}")


if __name__ == "__main__":
    main()
