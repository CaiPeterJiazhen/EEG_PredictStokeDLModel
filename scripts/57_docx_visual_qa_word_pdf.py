from __future__ import annotations

import base64
import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont, ImageStat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "doc_visual_qa"
PAGES_DIR = OUTPUT_ROOT / "pages"
PDF_DIR = OUTPUT_ROOT / "pdf"
CONTACT_DIR = OUTPUT_ROOT / "contact_sheets"
OUTPUT_MD = ROOT / "docs" / "docx_visual_qa_report.md"
OUTPUT_CSV = ROOT / "results" / "tables" / "docx_visual_qa_metrics.csv"


@dataclass(frozen=True)
class DocumentItem:
    label: str
    docx_path: Path


DOCX_ITEMS = [
    DocumentItem("nature_manuscript", ROOT / "output" / "doc" / "ResidualAware_SSL_CNN_Nature_Manuscript.docx"),
    DocumentItem("nature_clean_placeholder", ROOT / "output" / "doc" / "ResidualAware_SSL_CNN_Nature_Clean_Placeholder.docx"),
    DocumentItem("jne_structured", ROOT / "output" / "doc" / "ResidualAware_SSL_CNN_JNE_Structured_Manuscript.docx"),
    DocumentItem("jne_clean_placeholder", ROOT / "output" / "doc" / "ResidualAware_SSL_CNN_JNE_Clean_Placeholder.docx"),
    DocumentItem("supplementary_information", ROOT / "output" / "doc" / "ResidualAware_SSL_CNN_Supplementary_Information.docx"),
    DocumentItem("author_required_information_form", ROOT / "output" / "doc" / "Author_Required_Information_Form.docx"),
]


def main() -> None:
    for path in [OUTPUT_ROOT, PAGES_DIR, PDF_DIR, CONTACT_DIR, OUTPUT_CSV.parent, OUTPUT_MD.parent]:
        path.mkdir(parents=True, exist_ok=True)

    for item in DOCX_ITEMS:
        if not item.docx_path.exists():
            raise FileNotFoundError(item.docx_path)

    pdf_paths = export_docx_to_pdf(DOCX_ITEMS)
    rows: list[dict[str, str | int | float]] = []
    doc_summaries: list[dict[str, str | int | float]] = []

    for item in DOCX_ITEMS:
        page_paths = render_pdf_pages(item.label, pdf_paths[item.label])
        metrics = collect_page_metrics(item.label, page_paths)
        rows.extend(metrics)
        contact_path = make_contact_sheet(item.label, page_paths)
        doc_status = summarize_doc(item, pdf_paths[item.label], page_paths, metrics, contact_path)
        doc_summaries.append(doc_status)

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "document",
                "page",
                "width_px",
                "height_px",
                "mean_rgb",
                "std_rgb",
                "dark_fraction",
                "nonwhite_fraction",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_MD.write_text(build_report(doc_summaries, rows), encoding="utf-8")
    print(OUTPUT_MD)
    print(OUTPUT_CSV)

    failed = [row for row in rows if row["status"] == "FAIL"]
    if failed:
        raise SystemExit(f"Visual QA found {len(failed)} failed page checks")


def export_docx_to_pdf(items: list[DocumentItem]) -> dict[str, Path]:
    pdf_paths = {item.label: PDF_DIR / f"{item.label}.pdf" for item in items}
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "$items = @(",
    ]
    for item in items:
        docx = powershell_literal(str(item.docx_path))
        pdf = powershell_literal(str(pdf_paths[item.label]))
        lines.append(f"  @{{ Docx = {docx}; Pdf = {pdf} }}")
    lines.extend(
        [
            ")",
            "$word = New-Object -ComObject Word.Application",
            "$word.Visible = $false",
            "$word.DisplayAlerts = 0",
            "try {",
            "  foreach ($item in $items) {",
            "    $doc = $word.Documents.Open($item.Docx, $false, $true)",
            "    try {",
            "      $doc.ExportAsFixedFormat($item.Pdf, 17)",
            "    } finally {",
            "      $doc.Close($false)",
            "    }",
            "  }",
            "} finally {",
            "  $word.Quit()",
            "}",
        ]
    )
    encoded = base64.b64encode("\n".join(lines).encode("utf-16le")).decode("ascii")
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
        cwd=ROOT,
        check=True,
    )
    for path in pdf_paths.values():
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"PDF export failed: {path}")
    return pdf_paths


def powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def render_pdf_pages(label: str, pdf_path: Path) -> list[Path]:
    document = pdfium.PdfDocument(str(pdf_path))
    out_dir = PAGES_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_path in out_dir.glob("page_*.png"):
        old_path.unlink()

    page_paths: list[Path] = []
    for index in range(len(document)):
        page = document[index]
        bitmap = page.render(scale=1.7, rotation=0)
        image = bitmap.to_pil().convert("RGB")
        page_path = out_dir / f"page_{index + 1:03d}.png"
        image.save(page_path, optimize=True)
        page_paths.append(page_path)
    return page_paths


def collect_page_metrics(label: str, page_paths: list[Path]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for index, path in enumerate(page_paths, start=1):
        image = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(image)
        mean = tuple(round(value, 1) for value in stat.mean)
        std = tuple(round(value, 1) for value in stat.stddev)
        pixels = image.getdata()
        total = image.width * image.height
        dark = sum(1 for r, g, b in pixels if r < 45 and g < 45 and b < 45)
        nonwhite = sum(1 for r, g, b in pixels if min(r, g, b) < 245)
        dark_fraction = dark / total
        nonwhite_fraction = nonwhite / total
        status = "PASS"
        if min(image.size) < 700 or max(stat.stddev) < 2.5 or nonwhite_fraction < 0.002:
            status = "FAIL"
        rows.append(
            {
                "document": label,
                "page": index,
                "width_px": image.width,
                "height_px": image.height,
                "mean_rgb": str(mean),
                "std_rgb": str(std),
                "dark_fraction": round(dark_fraction, 5),
                "nonwhite_fraction": round(nonwhite_fraction, 5),
                "status": status,
            }
        )
    return rows


def make_contact_sheet(label: str, page_paths: list[Path]) -> Path:
    thumb_width = 280
    header_height = 34
    gap = 18
    cols = 3 if len(page_paths) > 4 else 2
    thumbnails = []
    for index, path in enumerate(page_paths, start=1):
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        thumb_height = max(1, int(image.height * ratio))
        thumb = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_width, thumb_height + header_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, thumb_width - 1, header_height - 1), fill=(232, 238, 246), outline=(150, 160, 174))
        draw.text((8, 9), f"{label} p{index}", fill=(20, 30, 45), font=default_font())
        canvas.paste(thumb, (0, header_height))
        draw.rectangle((0, header_height, thumb_width - 1, thumb_height + header_height - 1), outline=(160, 160, 160))
        thumbnails.append(canvas)

    rows = math.ceil(len(thumbnails) / cols)
    cell_height = max(image.height for image in thumbnails)
    sheet = Image.new(
        "RGB",
        (cols * thumb_width + (cols + 1) * gap, rows * cell_height + (rows + 1) * gap),
        (248, 249, 251),
    )
    for index, thumb in enumerate(thumbnails):
        row = index // cols
        col = index % cols
        x = gap + col * (thumb_width + gap)
        y = gap + row * (cell_height + gap)
        sheet.paste(thumb, (x, y))
    out_path = CONTACT_DIR / f"{label}_contact_sheet.png"
    sheet.save(out_path, optimize=True)
    return out_path


def default_font():
    try:
        return ImageFont.truetype("arial.ttf", 13)
    except OSError:
        return ImageFont.load_default()


def summarize_doc(
    item: DocumentItem,
    pdf_path: Path,
    page_paths: list[Path],
    metrics: list[dict[str, str | int | float]],
    contact_path: Path,
) -> dict[str, str | int | float]:
    fail_count = sum(1 for row in metrics if row["status"] == "FAIL")
    warn_count = sum(1 for row in metrics if row["status"] == "WARN")
    return {
        "label": item.label,
        "docx": item.docx_path.relative_to(ROOT).as_posix(),
        "pdf": pdf_path.relative_to(ROOT).as_posix(),
        "contact_sheet": contact_path.relative_to(ROOT).as_posix(),
        "pages": len(page_paths),
        "failed_pages": fail_count,
        "warn_pages": warn_count,
        "status": "PASS" if fail_count == 0 else "FAIL",
    }


def build_report(
    summaries: list[dict[str, str | int | float]],
    rows: list[dict[str, str | int | float]],
) -> str:
    pass_count = sum(1 for row in rows if row["status"] == "PASS")
    fail_count = sum(1 for row in rows if row["status"] == "FAIL")
    lines = [
        "# DOCX Visual QA Report",
        "",
        "This report was generated by exporting manuscript DOCX files through Microsoft Word COM to PDF, rendering pages with pypdfium2, and checking page images for nonblank content, dimensions, and simple raster integrity metrics. It supplements the structural DOCX audit; it does not replace final human page-by-page review in the target journal template.",
        "",
        "## Summary",
        "",
        f"- Page checks passed: {pass_count}",
        f"- Page checks failed: {fail_count}",
        "",
        "## Documents",
        "",
        "| Document | Pages | Failed pages | PDF preview | Contact sheet | Status |",
        "|---|---:|---:|---|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['label']} | {summary['pages']} | {summary['failed_pages']} | "
            f"`{summary['pdf']}` | `{summary['contact_sheet']}` | {summary['status']} |"
        )
    lines.extend(
        [
            "",
            "## Page Metrics",
            "",
            "`results/tables/docx_visual_qa_metrics.csv` stores per-page dimensions, RGB means and standard deviations, dark-pixel fraction, nonwhite-pixel fraction, and page status.",
            "",
            "## Remaining Manual Checks",
            "",
            "- Open the PDF previews in Word/Adobe/Edge before upload and inspect page breaks, figure sharpness, long table wrapping, and any journal-specific template requirements.",
            "- The working Nature and JNE manuscripts intentionally retain author-query text; clean placeholder variants are included for upload after author information is filled.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
