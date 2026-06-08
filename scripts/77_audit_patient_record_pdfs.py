from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GROUPS = [
    ("patient_record_books", Path(r"F:\CJZFile\EEG_M1\患者记录本")),
    ("healthy_control_record_books", Path(r"F:\CJZFile\EEG_M1\健康人记录本")),
]
OUTPUT_CSV = ROOT / "results" / "tables" / "patient_record_pdf_text_audit.csv"
OUTPUT_MD = ROOT / "docs" / "patient_record_pdf_text_audit.md"


@dataclass(frozen=True)
class PdfAudit:
    record_group: str
    record_id: str
    page_count: int
    pages_sampled_for_images: int
    extracted_text_chars: int
    pages_with_text: int
    embedded_image_count: int
    mean_image_coverage_sampled_pages: float
    text_layer_status: str
    metadata_utility: str
    interpretation: str


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = audit_all_pdfs()
    write_csv(rows)
    OUTPUT_MD.write_text(build_markdown(rows), encoding="utf-8")
    print(OUTPUT_CSV)
    print(OUTPUT_MD)
    print(f"pdfs_audited={len(rows)}")
    print("status_counts=" + ",".join(f"{key}:{value}" for key, value in sorted(Counter(row.text_layer_status for row in rows).items())))


def audit_all_pdfs() -> list[PdfAudit]:
    rows: list[PdfAudit] = []
    for group_name, directory in SOURCE_GROUPS:
        if not directory.exists():
            continue
        for index, pdf_path in enumerate(sorted(directory.glob("*.pdf")), start=1):
            rows.append(audit_pdf(group_name, f"{group_name}_{index:03d}", pdf_path))
    return rows


def audit_pdf(group_name: str, record_id: str, pdf_path: Path) -> PdfAudit:
    document = fitz.open(pdf_path)
    text_chars = 0
    pages_with_text = 0
    image_count = 0
    sampled_coverages: list[float] = []
    sample_pages = min(len(document), 3)

    for page_index, page in enumerate(document):
        text = page.get_text("text") or ""
        compact_text = "".join(text.split())
        text_chars += len(compact_text)
        if compact_text:
            pages_with_text += 1
        images = page.get_images(full=True)
        image_count += len(images)
        if page_index < sample_pages:
            sampled_coverages.append(image_coverage(page, images))

    mean_coverage = sum(sampled_coverages) / len(sampled_coverages) if sampled_coverages else 0.0
    if text_chars >= 200:
        text_status = "text_layer_present"
        utility = "review_text_before_any_prefill"
        interpretation = "A text layer is present; non-identifying protocol terms may be searchable after manual review."
    elif image_count > 0:
        text_status = "scan_likely_no_text_layer"
        utility = "ocr_or_manual_review_required"
        interpretation = (
            "The PDF appears image-based with little or no extractable text. It cannot support automatic author-metadata "
            "prefill without OCR or manual review."
        )
    else:
        text_status = "no_text_or_images_detected"
        utility = "not_usable_for_prefill"
        interpretation = "No extractable text or embedded images were detected by PyMuPDF."

    return PdfAudit(
        record_group=group_name,
        record_id=record_id,
        page_count=len(document),
        pages_sampled_for_images=sample_pages,
        extracted_text_chars=text_chars,
        pages_with_text=pages_with_text,
        embedded_image_count=image_count,
        mean_image_coverage_sampled_pages=round(mean_coverage, 4),
        text_layer_status=text_status,
        metadata_utility=utility,
        interpretation=interpretation,
    )


def image_coverage(page: fitz.Page, images: list[tuple]) -> float:
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    area = 0.0
    seen: set[int] = set()
    for image in images:
        xref = int(image[0])
        if xref in seen:
            continue
        seen.add(xref)
        for rect in page.get_image_rects(xref):
            area += max(0.0, float(rect.width * rect.height))
    return min(area / page_area, 1.0)


def write_csv(rows: list[PdfAudit]) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PdfAudit.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def build_markdown(rows: list[PdfAudit]) -> str:
    status_counts = Counter(row.text_layer_status for row in rows)
    utility_counts = Counter(row.metadata_utility for row in rows)
    group_counts = Counter(row.record_group for row in rows)
    total_pages = sum(row.page_count for row in rows)
    total_text_chars = sum(row.extracted_text_chars for row in rows)
    lines = [
        "# Patient-Record PDF Text-Layer Audit",
        "",
        "This audit checks whether local patient-record PDFs can be used as searchable evidence for author-required manuscript metadata. It deliberately reports only non-identifying counts and pseudonymous record IDs; patient names, PDF file names, and extracted source text are not written to this report.",
        "",
        "## Summary",
        "",
        f"- PDFs audited: {len(rows)}",
        f"- Total pages: {total_pages}",
        f"- Extracted text characters: {total_text_chars}",
    ]
    for group, count in sorted(group_counts.items()):
        lines.append(f"- {group}: {count}")
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    for utility, count in sorted(utility_counts.items()):
        lines.append(f"- {utility}: {count}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current PDFs do not provide searchable protocol, consent, ethics, device, rehabilitation, or safety text unless a text layer is detected below.",
            "- Image-based record books require OCR or manual review before any manuscript metadata can be inferred from them.",
            "- Even if OCR is later performed, patient-level notes should be converted into non-identifying cohort-level evidence before manuscript use.",
            "",
            "## Non-Identifying Record Audit",
            "",
            "| Record group | Record ID | Pages | Text chars | Pages with text | Images | Mean image coverage | Text-layer status | Metadata utility |",
            "|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.record_group} | {row.record_id} | {row.page_count} | {row.extracted_text_chars} | "
            f"{row.pages_with_text} | {row.embedded_image_count} | {row.mean_image_coverage_sampled_pages:.4f} | "
            f"{row.text_layer_status} | {row.metadata_utility} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
