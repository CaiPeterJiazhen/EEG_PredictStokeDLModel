from __future__ import annotations

import csv
import importlib.util
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF_ROOT = Path(r"F:\CJZFile\EEG_M1")
TEXT_AUDIT_CSV = ROOT / "results" / "tables" / "patient_record_pdf_text_audit.csv"
OUTPUT_MD = ROOT / "docs" / "pdf_ocr_readiness_audit.md"


OCR_COMMANDS = ("tesseract",)
OCR_PACKAGES = ("pytesseract", "easyocr", "cnocr", "rapidocr_onnxruntime")
RENDER_PACKAGES = ("pypdfium2", "pypdf", "PIL")

AUTHOR_FIELDS_REQUIRING_MANUAL_REVIEW = [
    ("ethics_approval", "ethics committee name, approval number, approval date, and site applicability"),
    ("informed_consent", "consent route, consent provider, procedure coverage, and data-sharing scope"),
    ("trial_or_study_registration", "registry identifier or author-approved non-registration statement"),
    ("study_site_dates_design", "hospital/department, recruitment dates, follow-up window, and design"),
    ("eligibility_stroke_timing", "inclusion/exclusion criteria, stroke subtype, lesion and timing rules"),
    ("tacs_device_electrodes", "device model, electrode dimensions/materials, and protocol confirmation"),
    ("concurrent_rehabilitation", "conventional rehabilitation dose, content, and consistency"),
    ("tacs_safety_adverse_events", "adverse events, tolerability, withdrawals, and monitoring method"),
    ("fma_assessors_timing", "assessor credentials, blinding, and assessment timing"),
    ("eeg_hardware_reference_impedance", "amplifier, cap, software, online reference, ground, and impedance"),
    ("raw_eeg_preprocessing", "filters, notch, rereference, bad-channel handling, ICA/artifact handling, export rules"),
    ("data_repository_doi_scope", "repository DOI/accession, licence, public/restricted scope, and request route"),
]


def main() -> None:
    OUTPUT_MD.write_text(build_report(), encoding="utf-8")
    print(OUTPUT_MD)


def build_report() -> str:
    command_rows = [{"tool": tool, "available": bool(shutil.which(tool))} for tool in OCR_COMMANDS]
    package_rows = [{"package": pkg, "available": importlib.util.find_spec(pkg) is not None} for pkg in OCR_PACKAGES]
    render_rows = [{"package": pkg, "available": importlib.util.find_spec(pkg) is not None} for pkg in RENDER_PACKAGES]
    pdf_summary = summarize_pdf_text_audit()
    render_status = render_smoke_test()
    automatic_ocr_ready = any(row["available"] for row in command_rows) or any(row["available"] for row in package_rows)

    lines = [
        "# PDF OCR Readiness Audit",
        "",
        "This audit checks whether the local patient-record PDF scans can be processed automatically for author-required manuscript metadata. It reports only non-identifying counts and tool availability. Patient names, PDF filenames, OCR text, and page images are not written to this report.",
        "",
        "## Summary",
        "",
        f"- Existing PDF text-layer audit rows: {pdf_summary['rows']}",
        f"- PDFs audited in text-layer report: {pdf_summary['pdfs']}",
        f"- Pages audited in text-layer report: {pdf_summary['pages']}",
        f"- Pages with any extractable text layer: {pdf_summary['pages_with_text']}",
        f"- Extracted text characters in text-layer report: {pdf_summary['extracted_text_chars']}",
        f"- PDFs requiring OCR or manual review: {pdf_summary['ocr_required']}",
        f"- OCR command available: {any(row['available'] for row in command_rows)}",
        f"- Python OCR package available: {any(row['available'] for row in package_rows)}",
        f"- PDF rendering smoke test: {render_status}",
        f"- Automatic OCR-ready in current environment: {automatic_ocr_ready}",
        "",
        "## OCR Tool Availability",
        "",
        "| Tool or package | Type | Available |",
        "|---|---|---|",
    ]
    for row in command_rows:
        lines.append(f"| {row['tool']} | command | {row['available']} |")
    for row in package_rows:
        lines.append(f"| {row['package']} | python_ocr_package | {row['available']} |")
    for row in render_rows:
        lines.append(f"| {row['package']} | python_render_or_pdf_package | {row['available']} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current environment can inspect PDF metadata and render pages, but it does not have a usable OCR engine for Chinese/English scanned clinical documents.",
            "- The existing text-layer audit found that all record-book PDFs require OCR or manual review before they can support ethics, consent, safety, protocol, or device metadata.",
            "- Until OCR or manual review is completed, the scanned PDFs should not be used to write submission-ready statements.",
            "- Any future OCR output should be reduced to non-identifying cohort-level evidence before manuscript insertion.",
            "",
            "## Manual Review Targets",
            "",
            "| Author field | What manual review must confirm |",
            "|---|---|",
        ]
    )
    for field, requirement in AUTHOR_FIELDS_REQUIRING_MANUAL_REVIEW:
        lines.append(f"| {field} | {requirement} |")

    lines.extend(
        [
            "",
            "## Recommended Next Step",
            "",
            "Install or run an OCR workflow that supports simplified Chinese and English clinical documents, then review outputs manually before any manuscript wording is changed. A suitable local route would be Tesseract with `chi_sim` and `eng` language data, PaddleOCR/RapidOCR, or institutional OCR software. The review should export only de-identified, cohort-level protocol facts and evidence-source labels.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_pdf_text_audit() -> dict[str, int]:
    if not TEXT_AUDIT_CSV.exists():
        return {"rows": 0, "pdfs": 0, "pages": 0, "pages_with_text": 0, "extracted_text_chars": 0, "ocr_required": 0}
    with TEXT_AUDIT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "rows": len(rows),
        "pdfs": len(rows),
        "pages": sum(int(float(row.get("page_count") or 0)) for row in rows),
        "pages_with_text": sum(int(float(row.get("pages_with_text") or 0)) for row in rows),
        "extracted_text_chars": sum(int(float(row.get("extracted_text_chars") or 0)) for row in rows),
        "ocr_required": sum(1 for row in rows if row.get("metadata_utility") == "ocr_or_manual_review_required"),
    }


def render_smoke_test() -> str:
    if importlib.util.find_spec("pypdfium2") is None:
        return "not_run_pypdfium2_missing"
    pdf_paths = sorted((PDF_ROOT / "患者记录本").glob("*.pdf")) + sorted((PDF_ROOT / "健康人记录本").glob("*.pdf"))
    if not pdf_paths:
        return "not_run_no_local_record_pdfs_found"
    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_paths[0]))
        page = document[0]
        bitmap = page.render(scale=0.1)
        image = bitmap.to_pil()
        width, height = image.size
        page.close()
        document.close()
    except Exception as exc:  # pragma: no cover - diagnostic report path
        return f"fail_{type(exc).__name__}"
    return f"pass_rendered_first_available_record_page_{width}x{height}px"


if __name__ == "__main__":
    main()
