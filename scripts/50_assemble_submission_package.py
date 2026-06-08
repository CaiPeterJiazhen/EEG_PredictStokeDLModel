from __future__ import annotations

import csv
import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "outputs" / "submission_package_20260601"
ZIP_PATH = ROOT / "outputs" / "ResidualAware_SSL_CNN_submission_package_20260601.zip"


@dataclass(frozen=True)
class PackageItem:
    source: Path
    destination: str
    role: str


def main() -> None:
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    items = collect_items()
    copied: list[dict[str, str | int]] = []

    for item in items:
        if not item.source.exists():
            raise FileNotFoundError(f"Missing package source: {item.source}")
        destination = PACKAGE_ROOT / item.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, destination)
        copied.append(
            {
                "relative_path": destination.relative_to(PACKAGE_ROOT).as_posix(),
                "role": item.role,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "source_path": item.source.relative_to(ROOT).as_posix(),
            }
        )

    manifest_path = PACKAGE_ROOT / "submission_package_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "role", "size_bytes", "sha256", "source_path"],
        )
        writer.writeheader()
        writer.writerows(copied)

    readme_path = PACKAGE_ROOT / "README_submission_package.md"
    readme_path.write_text(build_readme(copied), encoding="utf-8")

    # Include manifest and README in the archive manifest after writing them.
    for extra_path, role in [(manifest_path, "package_manifest"), (readme_path, "package_readme")]:
        copied.append(
            {
                "relative_path": extra_path.relative_to(PACKAGE_ROOT).as_posix(),
                "role": role,
                "size_bytes": extra_path.stat().st_size,
                "sha256": sha256_file(extra_path),
                "source_path": extra_path.relative_to(ROOT).as_posix(),
            }
        )

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "role", "size_bytes", "sha256", "source_path"],
        )
        writer.writeheader()
        writer.writerows(copied)

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PACKAGE_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(PACKAGE_ROOT))

    print(PACKAGE_ROOT)
    print(ZIP_PATH)
    print(f"Packaged {len(copied)} files")


def collect_items() -> list[PackageItem]:
    items: list[PackageItem] = [
        item("output/doc/ResidualAware_SSL_CNN_Nature_Manuscript.docx", "01_manuscript/ResidualAware_SSL_CNN_Nature_Manuscript.docx", "main_manuscript_docx"),
        item("output/doc/ResidualAware_SSL_CNN_Nature_Clean_Placeholder.docx", "01_manuscript/ResidualAware_SSL_CNN_Nature_Clean_Placeholder.docx", "clean_placeholder_manuscript_docx"),
        item("output/doc/ResidualAware_SSL_CNN_JNE_Structured_Manuscript.docx", "01_manuscript/ResidualAware_SSL_CNN_JNE_Structured_Manuscript.docx", "jne_structured_manuscript_docx"),
        item("output/doc/ResidualAware_SSL_CNN_JNE_Clean_Placeholder.docx", "01_manuscript/ResidualAware_SSL_CNN_JNE_Clean_Placeholder.docx", "jne_clean_placeholder_manuscript_docx"),
        item("output/doc/ResidualAware_SSL_CNN_Supplementary_Information.docx", "01_manuscript/ResidualAware_SSL_CNN_Supplementary_Information.docx", "supplementary_information_docx"),
        item("docs/manuscript_residual_aware_ssl_cnn_nature_polished.md", "01_manuscript/manuscript_residual_aware_ssl_cnn_nature_polished.md", "main_manuscript_markdown"),
        item("docs/manuscript_residual_aware_ssl_cnn_nature_clean_placeholder.md", "01_manuscript/manuscript_residual_aware_ssl_cnn_nature_clean_placeholder.md", "clean_placeholder_manuscript_markdown"),
        item("docs/manuscript_residual_aware_ssl_cnn_jne_structured.md", "01_manuscript/manuscript_residual_aware_ssl_cnn_jne_structured.md", "jne_structured_manuscript_markdown"),
        item("docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md", "01_manuscript/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md", "jne_clean_placeholder_manuscript_markdown"),
        item("outputs/manuscript_package/ResidualAware_SSL_CNN_Source_Data.xlsx", "02_source_data/ResidualAware_SSL_CNN_Source_Data.xlsx", "source_data_workbook"),
        item("docs/citation_artifacts/selected_references.ris", "03_references/selected_references.ris", "reference_manager_import"),
        item("docs/citation_artifacts/selected_references.md", "03_references/selected_references.md", "reference_list"),
        item("docs/citation_artifacts/manuscript_citation_map.md", "03_references/manuscript_citation_map.md", "citation_audit"),
        item("docs/reference_metadata_audit.md", "03_references/reference_metadata_audit.md", "reference_metadata_audit"),
        item("docs/citation_claim_coverage_audit.md", "03_references/citation_claim_coverage_audit.md", "citation_claim_coverage_audit"),
        item("docs/data_availability_and_fair_audit.md", "04_audits/data_availability_and_fair_audit.md", "data_availability_audit"),
        item("docs/repository_readme_for_deposit.md", "02_source_data/repository_readme_for_deposit.md", "repository_readme_for_deposit"),
        item("docs/methods_detail_provenance.md", "04_audits/methods_detail_provenance.md", "methods_provenance"),
        item("docs/methods_gap_resolution_from_project_files.md", "04_audits/methods_gap_resolution_from_project_files.md", "methods_gap_resolution_audit"),
        item("docs/eeg_metadata_audit.md", "04_audits/eeg_metadata_audit.md", "eeg_metadata_audit"),
        item("docs/participant_flow_and_safety_source_notes.md", "04_audits/participant_flow_and_safety_source_notes.md", "participant_flow_safety_source_notes"),
        item("docs/probast_tripod_ai_risk_audit.md", "04_audits/probast_tripod_ai_risk_audit.md", "prediction_model_risk_audit"),
        item("docs/performance_precision_audit.md", "04_audits/performance_precision_audit.md", "performance_precision_audit"),
        item("docs/numeric_claim_source_trace_audit.md", "04_audits/numeric_claim_source_trace_audit.md", "numeric_claim_source_trace_audit"),
        item("docs/claim_strength_audit.md", "04_audits/claim_strength_audit.md", "claim_strength_audit"),
        item("docs/model_reporting_card.md", "04_audits/model_reporting_card.md", "model_reporting_card"),
        item("docs/source_workbook_author_metadata_audit.md", "04_audits/source_workbook_author_metadata_audit.md", "source_workbook_author_metadata_audit"),
        item("docs/patient_record_pdf_text_audit.md", "04_audits/patient_record_pdf_text_audit.md", "patient_record_pdf_text_audit"),
        item("docs/pdf_ocr_readiness_audit.md", "04_audits/pdf_ocr_readiness_audit.md", "pdf_ocr_readiness_audit"),
        item("docs/author_information_request_table.md", "04_audits/author_information_request_table.md", "author_query_table"),
        item("docs/author_quick_response_request_zh.md", "07_submission_materials/author_quick_response_request_zh.md", "author_quick_response_request_zh"),
        item("docs/author_required_information_form.md", "07_submission_materials/author_required_information_form.md", "author_required_information_form"),
        item("docs/author_minimal_completion_pack.md", "07_submission_materials/author_minimal_completion_pack.md", "author_minimal_completion_pack"),
        item("docs/author_required_evidence_trace.md", "07_submission_materials/author_required_evidence_trace.md", "author_required_evidence_trace"),
        item("docs/author_field_replacement_map.md", "07_submission_materials/author_field_replacement_map.md", "author_field_replacement_map"),
        item("docs/author_submission_metadata_template.json", "07_submission_materials/author_submission_metadata_template.json", "author_submission_metadata_template"),
        item("outputs/manuscript_package/author_minimal_completion_answers.json", "07_submission_materials/author_minimal_completion_answers.json", "author_minimal_completion_answers"),
        item("docs/author_submission_metadata_validation_report.md", "07_submission_materials/author_submission_metadata_validation_report.md", "author_submission_metadata_validation_report"),
        item("docs/author_metadata_insertion_protocol.md", "07_submission_materials/author_metadata_insertion_protocol.md", "author_metadata_insertion_protocol"),
        item("output/doc/Author_Quick_Response_Request_ZH.docx", "07_submission_materials/Author_Quick_Response_Request_ZH.docx", "author_quick_response_request_docx"),
        item("output/doc/Author_Required_Information_Form.docx", "07_submission_materials/Author_Required_Information_Form.docx", "author_required_information_form_docx"),
        item("outputs/manuscript_package/Author_Submission_Metadata_Intake.xlsx", "07_submission_materials/Author_Submission_Metadata_Intake.xlsx", "author_submission_metadata_intake_workbook"),
        item("outputs/manuscript_package/author_submission_metadata_from_intake.json", "07_submission_materials/author_submission_metadata_from_intake.json", "author_submission_metadata_from_intake"),
        item("outputs/manuscript_package/author_submission_metadata_project_prefill.json", "07_submission_materials/author_submission_metadata_project_prefill.json", "author_submission_metadata_project_prefill"),
        item("docs/author_metadata_intake_import_report.md", "07_submission_materials/author_metadata_intake_import_report.md", "author_metadata_intake_import_report"),
        item("docs/author_metadata_project_prefill_report.md", "07_submission_materials/author_metadata_project_prefill_report.md", "author_metadata_project_prefill_report"),
        item("docs/tripod_ai_reporting_checklist.md", "04_audits/tripod_ai_reporting_checklist.md", "reporting_checklist"),
        item("docs/submission_readiness_audit.md", "04_audits/submission_readiness_audit.md", "submission_readiness_audit"),
        item("docs/submission_artifact_quality_audit.md", "04_audits/submission_artifact_quality_audit.md", "submission_artifact_quality_audit"),
        item("docs/presubmission_editorial_readiness_audit.md", "04_audits/presubmission_editorial_readiness_audit.md", "presubmission_editorial_readiness_audit"),
        item("docs/final_submission_gate_report.md", "04_audits/final_submission_gate_report.md", "final_submission_gate_report"),
        item("docs/main_paper_completion_status_20260602.md", "04_audits/main_paper_completion_status_20260602.md", "main_paper_completion_status"),
        item("docs/figure_submission_readiness_audit.md", "04_audits/figure_submission_readiness_audit.md", "figure_submission_readiness_audit"),
        item("docs/manuscript_integrity_audit.md", "04_audits/manuscript_integrity_audit.md", "manuscript_integrity_audit"),
        item("docs/prediction_validation_integrity_audit.md", "04_audits/prediction_validation_integrity_audit.md", "prediction_validation_integrity_audit"),
        item("docs/docx_visual_qa_report.md", "04_audits/docx_visual_qa_report.md", "docx_visual_qa_report"),
        item("docs/manuscript_figure_table_map.md", "04_audits/manuscript_figure_table_map.md", "figure_table_map"),
        item("docs/statistical_validation_summary.md", "04_audits/statistical_validation_summary.md", "statistical_validation_summary"),
        item("docs/local_reference_audit.md", "04_audits/local_reference_audit.md", "local_reference_audit"),
        item("docs/local_reference_positioning_matrix.md", "03_references/local_reference_positioning_matrix.md", "local_reference_positioning_matrix"),
        item("docs/target_journal_strategy.md", "07_submission_materials/target_journal_strategy.md", "target_journal_strategy"),
        item("docs/jne_submission_checklist.md", "07_submission_materials/jne_submission_checklist.md", "jne_submission_checklist"),
        item("docs/jne_final_declaration_templates.md", "07_submission_materials/jne_final_declaration_templates.md", "jne_final_declaration_templates"),
        item("docs/cover_letter_template.md", "07_submission_materials/cover_letter_template.md", "cover_letter_template"),
        item("pyproject.toml", "06_reproducibility/pyproject.toml", "environment_metadata"),
    ]

    for path in sorted((ROOT / "results" / "figures" / "nature").glob("*")):
        if path.is_file():
            items.append(PackageItem(path, f"05_figures/nature/{path.name}", "main_figure_export"))

    connectivity_dir = ROOT / "results" / "figures" / "explainability" / "mne_wpli_connectivity"
    for path in sorted(connectivity_dir.glob("*")):
        if path.is_file():
            items.append(PackageItem(path, f"05_figures/mne_wpli_connectivity/{path.name}", "supplementary_connectivity_figure_export"))

    topomap_dir = ROOT / "results" / "figures" / "explainability" / "mne_topomaps"
    for path in sorted(topomap_dir.glob("*")):
        if path.is_file():
            items.append(PackageItem(path, f"05_figures/mne_topomaps/{path.name}", "mne_topomap_figure_export"))

    for path in sorted((ROOT / "results" / "tables").glob("*.csv")):
        items.append(PackageItem(path, f"02_source_data/tables/{path.name}", "source_table_csv"))

    for path in sorted((ROOT / "results" / "statistics").glob("*.csv")):
        items.append(PackageItem(path, f"02_source_data/statistics/{path.name}", "statistics_csv"))

    visual_qa_root = ROOT / "outputs" / "doc_visual_qa"
    for subdir, role in [("pdf", "docx_visual_qa_pdf_preview"), ("contact_sheets", "docx_visual_qa_contact_sheet")]:
        directory = visual_qa_root / subdir
        if directory.exists():
            for path in sorted(directory.glob("*")):
                if path.is_file():
                    items.append(PackageItem(path, f"04_audits/docx_visual_qa/{subdir}/{path.name}", role))

    for script_name in [
        "39_train_clinical_and_incremental_baselines.py",
        "45_make_mne_explainability_topomaps.py",
        "45_make_nature_manuscript_figures.py",
        "46_build_submission_docx.py",
        "46_make_mne_wpli_connectivity.py",
        "47_build_source_data_workbook.cjs",
        "48_audit_methods_source_data.py",
        "49_build_supplementary_information_docx.py",
        "50_assemble_submission_package.py",
        "51_make_jne_submission_variant.py",
        "52_audit_submission_artifacts.py",
        "53_make_clean_submission_variants.py",
        "54_build_repository_data_dictionary.py",
        "55_build_author_required_info_form.py",
        "56_audit_manuscript_integrity.py",
        "57_docx_visual_qa_word_pdf.py",
        "58_verify_reference_metadata.py",
        "59_build_local_reference_positioning_matrix.py",
        "60_audit_prediction_validation_integrity.py",
        "61_audit_author_required_evidence.py",
        "62_build_author_field_replacement_map.py",
        "63_validate_author_submission_metadata.py",
        "64_build_author_metadata_insertion_protocol.py",
        "65_build_author_metadata_intake_workbook.cjs",
        "66_import_author_metadata_intake_workbook.py",
        "67_run_final_submission_gate.py",
        "68_build_author_metadata_project_prefill.py",
        "69_build_participant_flow_source_data.py",
        "70_build_prediction_model_risk_audit.py",
        "71_build_performance_precision_audit.py",
        "72_build_claim_strength_audit.py",
        "73_build_model_reporting_card.py",
        "74_audit_source_workbook_author_metadata.py",
        "75_audit_presubmission_editorial_readiness.py",
        "76_build_author_minimal_completion_pack.py",
        "77_audit_patient_record_pdfs.py",
        "78_audit_pdf_ocr_readiness.py",
        "79_audit_figure_submission_readiness.py",
        "80_audit_citation_claim_coverage.py",
        "81_audit_numeric_claim_source_trace.py",
    ]:
        items.append(item(f"scripts/{script_name}", f"06_reproducibility/scripts/{script_name}", "reproducibility_script"))

    for test_name in ["test_mne_topomap_coordinates.py", "test_mne_wpli_connectivity.py", "test_final_submission_gate.py"]:
        items.append(item(f"tests/{test_name}", f"06_reproducibility/tests/{test_name}", "verification_test"))

    return items


def item(source: str, destination: str, role: str) -> PackageItem:
    return PackageItem(ROOT / source, destination, role)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_readme(rows: list[dict[str, str | int]]) -> str:
    return "\n".join(
        [
            "# Residual-Aware SSL-CNN Manuscript Submission Package",
            "",
            "This package contains the current manuscript draft, supplementary information, source-data workbook, figure exports, references, audit files, and reproducibility scripts for the stroke EEG proportional-recovery manuscript.",
            "",
            "## Contents",
            "",
            "- `01_manuscript/`: main DOCX manuscript, clean placeholder variants, JNE structured-abstract variants, supplementary DOCX, and Markdown manuscript drafts.",
            "- `02_source_data/`: source-data workbook plus CSV tables and statistical outputs.",
            "- `03_references/`: RIS import file, selected references, citation map, reference metadata audit, and claim-level citation coverage audit.",
            "- `04_audits/`: data availability, methods provenance, methods-gap, EEG metadata, numeric-claim source trace, patient-record PDF text-layer audit, PDF OCR readiness audit, figure/table readiness audit, TRIPOD+AI, author-query, manuscript-integrity, DOCX visual QA, and submission-readiness audits.",
            "- `05_figures/`: main and supplementary figure exports, including MNE-Python topomap and connectivity outputs.",
            "- `06_reproducibility/`: scripts, selected tests, and environment metadata used to rebuild the package artifacts.",
            "- `07_submission_materials/`: target-journal strategy matrix, JNE submission checklist, final declaration templates, cover-letter template, author-required information form, minimal author completion pack, author metadata validation template, final-insertion protocol, XLSX author metadata intake workbook, and workbook-to-JSON import report.",
            "",
            "## Important Limitations",
            "",
            "- Raw EEG and clinical source workbooks are not included because they may contain human-participant or identifiable data.",
            "- Ethics approval, informed consent/data-sharing wording, EEG acquisition hardware, preprocessing filters, artifact rejection, recruitment dates, repository DOI/licences, and target-journal formatting still require author confirmation.",
            "- The supervised cohort is small (n=19) and has no external validation; manuscript conclusions must remain exploratory.",
            "- The residual threshold of 1.5 is cohort-median-derived and should not be presented as an externally validated clinical cut-off.",
            "",
            "## Manifest",
            "",
            "`submission_package_manifest.csv` lists packaged files, SHA256 checksums, source paths, and roles.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
