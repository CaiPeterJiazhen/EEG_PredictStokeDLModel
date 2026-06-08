from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MD = ROOT / "docs" / "final_submission_gate_report.md"
PACKAGE_ROOT = ROOT / "outputs" / "submission_package_20260601"
ZIP_PATH = ROOT / "outputs" / "ResidualAware_SSL_CNN_submission_package_20260601.zip"
SOURCE_WORKBOOK = ROOT / "outputs" / "manuscript_package" / "ResidualAware_SSL_CNN_Source_Data.xlsx"
INTAKE_WORKBOOK = ROOT / "outputs" / "manuscript_package" / "Author_Submission_Metadata_Intake.xlsx"
INTAKE_JSON = ROOT / "outputs" / "manuscript_package" / "author_submission_metadata_from_intake.json"


class GateCheck(NamedTuple):
    category: str
    check: str
    status: str
    observed: str
    notes: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final manuscript submission readiness gate.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless every gate is ready for final submission.",
    )
    args = parser.parse_args()

    checks = collect_checks()
    overall = determine_overall_status(checks)
    OUTPUT_MD.write_text(build_report(overall, checks), encoding="utf-8")

    counts = count_statuses(checks)
    print(OUTPUT_MD)
    print(f"overall_status={overall}")
    print("status_counts=" + ",".join(f"{key}:{counts[key]}" for key in sorted(counts)))

    if args.strict and overall != "READY_FOR_FINAL_SUBMISSION":
        raise SystemExit(1)


def collect_checks() -> list[GateCheck]:
    checks: list[GateCheck] = []
    checks.extend(audit_csv("manuscript", "manuscript_integrity", ROOT / "results/tables/manuscript_integrity_audit.csv"))
    checks.extend(audit_csv("visual_qa", "docx_visual_qa", ROOT / "results/tables/docx_visual_qa_metrics.csv"))
    checks.extend(audit_csv("artifact", "submission_artifact_quality", ROOT / "results/tables/submission_artifact_quality_audit.csv"))
    checks.extend(
        audit_csv(
            "validation",
            "prediction_validation_integrity",
            ROOT / "results/tables/prediction_validation_integrity_audit.csv",
        )
    )
    checks.append(check_source_workbook())
    checks.append(check_intake_workbook())
    checks.append(check_package_integrity())
    checks.append(check_package_required_entries())
    checks.append(check_strict_command("author_metadata", "strict_validation", ["scripts/63_validate_author_submission_metadata.py", "--metadata", str(INTAKE_JSON), "--strict"]))
    checks.append(check_strict_command("author_metadata", "insertion_protocol", ["scripts/64_build_author_metadata_insertion_protocol.py", "--metadata", str(INTAKE_JSON), "--strict"]))
    return checks


def audit_csv(category: str, check: str, path: Path) -> list[GateCheck]:
    if not path.exists():
        return [GateCheck(category, check, "FAIL", "missing", f"Expected audit CSV at {path}.")]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    statuses = Counter((row.get("status") or "").strip() or "UNKNOWN" for row in rows)
    observed = ", ".join(f"{key}={statuses[key]}" for key in sorted(statuses))
    if statuses.get("FAIL", 0) or statuses.get("UNKNOWN", 0):
        status = "FAIL"
    elif statuses.get("WARN", 0):
        status = "WARN"
    else:
        status = "PASS"
    return [GateCheck(category, check, status, observed, f"{len(rows)} rows checked.")]


def check_source_workbook() -> GateCheck:
    if not SOURCE_WORKBOOK.exists():
        return GateCheck("source_data", "source_workbook", "FAIL", "missing", str(SOURCE_WORKBOOK))
    workbook = load_workbook(SOURCE_WORKBOOK, read_only=True, data_only=True)
    required = {
        "README",
        "Data_Dictionary",
        "Artifact_QA",
        "Docx_Visual_QA",
        "Validation_Audit",
        "Replacement_Map",
        "PatientPDF_Audit",
    }
    missing = sorted(required - set(workbook.sheetnames))
    data_dictionary_dimension = (
        workbook["Data_Dictionary"].calculate_dimension(force=True)
        if "Data_Dictionary" in workbook.sheetnames
        else "missing"
    )
    patient_pdf_dimension = (
        workbook["PatientPDF_Audit"].calculate_dimension(force=True)
        if "PatientPDF_Audit" in workbook.sheetnames
        else "missing"
    )
    data_dictionary_ok = False
    if "Data_Dictionary" in workbook.sheetnames:
        min_col, min_row, max_col, max_row = range_boundaries(data_dictionary_dimension)
        data_dictionary_ok = min_col == 1 and min_row == 1 and max_col == 8 and max_row >= 447
    patient_pdf_ok = False
    if "PatientPDF_Audit" in workbook.sheetnames:
        min_col, min_row, max_col, max_row = range_boundaries(patient_pdf_dimension)
        patient_pdf_ok = min_col == 1 and min_row == 1 and max_col == 11 and max_row >= 32
    observed = (
        f"sheets={len(workbook.sheetnames)}; Data_Dictionary={data_dictionary_dimension}; "
        f"PatientPDF_Audit={patient_pdf_dimension}; missing_required={','.join(missing) if missing else 'none'}"
    )
    status = "PASS" if len(workbook.sheetnames) >= 36 and not missing and data_dictionary_ok and patient_pdf_ok else "FAIL"
    return GateCheck("source_data", "source_workbook", status, observed, "Checks source-data workbook structure.")


def check_intake_workbook() -> GateCheck:
    if not INTAKE_WORKBOOK.exists():
        return GateCheck("author_metadata", "intake_workbook", "FAIL", "missing", str(INTAKE_WORKBOOK))
    workbook = load_workbook(INTAKE_WORKBOOK, read_only=False, data_only=True)
    required = {
        "README",
        "Minimal_Completion",
        "Author_Input",
        "Field_Summary",
        "Replacement_Map",
        "Project_Prefill",
        "Source_Workbook_Audit",
        "Validation_Guide",
    }
    missing = sorted(required - set(workbook.sheetnames))
    dimension = workbook["Author_Input"].calculate_dimension() if "Author_Input" in workbook.sheetnames else "missing"
    minimal_dimension = (
        workbook["Minimal_Completion"].calculate_dimension()
        if "Minimal_Completion" in workbook.sheetnames
        else "missing"
    )
    prefill_dimension = workbook["Project_Prefill"].calculate_dimension() if "Project_Prefill" in workbook.sheetnames else "missing"
    source_audit_dimension = (
        workbook["Source_Workbook_Audit"].calculate_dimension()
        if "Source_Workbook_Audit" in workbook.sheetnames
        else "missing"
    )
    prefill_ok = False
    if "Project_Prefill" in workbook.sheetnames:
        min_col, min_row, max_col, max_row = range_boundaries(prefill_dimension)
        prefill_ok = min_col == 1 and min_row == 1 and max_col == 7 and max_row >= 11
    source_audit_ok = False
    if "Source_Workbook_Audit" in workbook.sheetnames:
        min_col, min_row, max_col, max_row = range_boundaries(source_audit_dimension)
        source_audit_ok = min_col == 1 and min_row == 1 and max_col == 9 and max_row >= 15
    minimal_ok = False
    if "Minimal_Completion" in workbook.sheetnames:
        min_col, min_row, max_col, max_row = range_boundaries(minimal_dimension)
        minimal_ok = min_col == 1 and min_row == 1 and max_col == 16 and max_row >= 19
    observed = (
        f"sheets={len(workbook.sheetnames)}; Author_Input={dimension}; "
        f"Minimal_Completion={minimal_dimension}; Project_Prefill={prefill_dimension}; "
        f"Source_Workbook_Audit={source_audit_dimension}; "
        f"missing_required={','.join(missing) if missing else 'none'}"
    )
    status = "PASS" if not missing and dimension == "A1:K124" and minimal_ok and prefill_ok and source_audit_ok else "FAIL"
    return GateCheck("author_metadata", "intake_workbook", status, observed, "Checks author-facing metadata workbook.")


def check_package_integrity() -> GateCheck:
    manifest = PACKAGE_ROOT / "submission_package_manifest.csv"
    if not manifest.exists() or not ZIP_PATH.exists():
        return GateCheck("package", "zip_integrity", "FAIL", "manifest_or_zip_missing", "Package manifest and zip must exist.")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [row["relative_path"] for row in rows if not (PACKAGE_ROOT / row["relative_path"]).exists()]
    with zipfile.ZipFile(ZIP_PATH) as archive:
        entries = [name for name in archive.namelist() if not name.endswith("/")]
        bad = archive.testzip()
    observed = f"manifest_rows={len(rows)}; missing_paths={len(missing)}; zip_entries={len(entries)}; testzip={bad}"
    status = "PASS" if not missing and len(rows) == len(entries) and bad is None else "FAIL"
    return GateCheck("package", "zip_integrity", status, observed, "Checks assembled package manifest and archive.")


def check_package_required_entries() -> GateCheck:
    manifest = PACKAGE_ROOT / "submission_package_manifest.csv"
    required = {
        "07_submission_materials/Author_Submission_Metadata_Intake.xlsx",
        "07_submission_materials/author_minimal_completion_answers.json",
        "07_submission_materials/author_minimal_completion_pack.md",
        "07_submission_materials/author_submission_metadata_from_intake.json",
        "07_submission_materials/author_metadata_intake_import_report.md",
        "07_submission_materials/author_metadata_insertion_protocol.md",
        "07_submission_materials/author_quick_response_request_zh.md",
        "07_submission_materials/Author_Quick_Response_Request_ZH.docx",
        "04_audits/patient_record_pdf_text_audit.md",
        "04_audits/pdf_ocr_readiness_audit.md",
        "04_audits/figure_submission_readiness_audit.md",
        "04_audits/numeric_claim_source_trace_audit.md",
        "03_references/citation_claim_coverage_audit.md",
        "02_source_data/tables/numeric_claim_source_trace_audit.csv",
        "02_source_data/tables/patient_record_pdf_text_audit.csv",
        "06_reproducibility/scripts/66_import_author_metadata_intake_workbook.py",
        "06_reproducibility/scripts/76_build_author_minimal_completion_pack.py",
        "06_reproducibility/scripts/77_audit_patient_record_pdfs.py",
        "06_reproducibility/scripts/78_audit_pdf_ocr_readiness.py",
        "06_reproducibility/scripts/79_audit_figure_submission_readiness.py",
        "06_reproducibility/scripts/80_audit_citation_claim_coverage.py",
        "06_reproducibility/scripts/81_audit_numeric_claim_source_trace.py",
    }
    if not manifest.exists():
        return GateCheck("package", "required_entries", "FAIL", "manifest_missing", "Cannot inspect package entries.")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        paths = {row["relative_path"] for row in csv.DictReader(handle)}
    missing = sorted(required - paths)
    observed = f"missing_required={','.join(missing) if missing else 'none'}"
    status = "PASS" if not missing else "FAIL"
    return GateCheck("package", "required_entries", status, observed, "Checks final author-metadata workflow files are packaged.")


def check_strict_command(category: str, check: str, command: list[str]) -> GateCheck:
    completed = subprocess.run([sys.executable, *command], cwd=ROOT, text=True, capture_output=True)
    output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
    observed = f"exit={completed.returncode}; output={compact(output)}"
    if completed.returncode == 0:
        return GateCheck(category, check, "PASS", observed, "Strict gate passed.")
    return GateCheck(category, check, "BLOCKED", observed, "Author metadata are not ready for final insertion.")


def compact(value: str, limit: int = 240) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def count_statuses(checks: list[GateCheck]) -> dict[str, int]:
    return dict(Counter(check.status for check in checks))


def determine_overall_status(checks: list[GateCheck]) -> str:
    if any(check.status == "FAIL" for check in checks):
        return "FAIL"
    blocked = [check for check in checks if check.status == "BLOCKED"]
    if blocked and all(check.category == "author_metadata" for check in blocked):
        return "BLOCKED_BY_AUTHOR_METADATA"
    if blocked:
        return "BLOCKED"
    if any(check.status == "WARN" for check in checks):
        return "READY_WITH_WARNINGS"
    return "READY_FOR_FINAL_SUBMISSION"


def build_report(overall: str, checks: list[GateCheck]) -> str:
    counts = count_statuses(checks)
    lines = [
        "# Final Submission Gate Report",
        "",
        f"Overall status: **{overall}**",
        "",
        "This report consolidates the current manuscript/package readiness checks. It is intentionally conservative: final submission remains blocked until author-approved ethics, consent, study-design, EEG acquisition/preprocessing, safety, and repository metadata pass strict validation.",
        "",
        "## Status Counts",
        "",
    ]
    for status in ["PASS", "WARN", "BLOCKED", "FAIL"]:
        lines.append(f"- {status}: {counts.get(status, 0)}")
    lines.extend(["", "## Checks", "", "| Category | Check | Status | Observed | Notes |", "|---|---|---|---|---|"])
    for check in checks:
        lines.append(
            f"| {escape(check.category)} | {escape(check.check)} | {escape(check.status)} | {escape(check.observed)} | {escape(check.notes)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `READY_FOR_FINAL_SUBMISSION` means all automated gates passed and no author metadata blocker remains.",
            "- `READY_WITH_WARNINGS` means no hard failures were found, but warnings should be reviewed before upload.",
            "- `BLOCKED_BY_AUTHOR_METADATA` means the manuscript/package is structurally ready, but final author-provided submission metadata are incomplete.",
            "- `FAIL` means a non-author-metadata artifact, package, workbook, or validation check failed and must be fixed before submission.",
            "",
        ]
    )
    return "\n".join(lines)


def escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
