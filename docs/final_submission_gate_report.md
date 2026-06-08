# Final Submission Gate Report

Overall status: **BLOCKED_BY_AUTHOR_METADATA**

This report consolidates the current manuscript/package readiness checks. It is intentionally conservative: final submission remains blocked until author-approved ethics, consent, study-design, EEG acquisition/preprocessing, safety, and repository metadata pass strict validation.

## Status Counts

- PASS: 7
- WARN: 1
- BLOCKED: 2
- FAIL: 0

## Checks

| Category | Check | Status | Observed | Notes |
|---|---|---|---|---|
| manuscript | manuscript_integrity | PASS | PASS=78 | 78 rows checked. |
| visual_qa | docx_visual_qa | PASS | PASS=75 | 75 rows checked. |
| artifact | submission_artifact_quality | WARN | PASS=48, WARN=6 | 54 rows checked. |
| validation | prediction_validation_integrity | PASS | PASS=21 | 21 rows checked. |
| source_data | source_workbook | PASS | sheets=36; Data_Dictionary=A1:H447; PatientPDF_Audit=A1:K32; missing_required=none | Checks source-data workbook structure. |
| author_metadata | intake_workbook | PASS | sheets=8; Author_Input=A1:K124; Minimal_Completion=A1:P114; Project_Prefill=A1:G25; Source_Workbook_Audit=A1:I15; missing_required=none | Checks author-facing metadata workbook. |
| package | zip_integrity | PASS | manifest_rows=224; missing_paths=0; zip_entries=224; testzip=None | Checks assembled package manifest and archive. |
| package | required_entries | PASS | missing_required=none | Checks final author-metadata workflow files are packaged. |
| author_metadata | strict_validation | BLOCKED | exit=1; output=F:\CJZProjectFile\EEG_PredictStokeDLModel\docs\author_submission_metadata_validation_report.md metadata_fields=19 complete_fields=1 incomplete_blocking_fields=12 incomplete_high_priority_fields=6 | Author metadata are not ready for final insertion. |
| author_metadata | insertion_protocol | BLOCKED | exit=1; output=F:\CJZProjectFile\EEG_PredictStokeDLModel\docs\author_metadata_insertion_protocol.md insertion_fields=19 ready_fields=0 blocked_fields=19 strict_validation_exit=1 | Author metadata are not ready for final insertion. |

## Interpretation

- `READY_FOR_FINAL_SUBMISSION` means all automated gates passed and no author metadata blocker remains.
- `READY_WITH_WARNINGS` means no hard failures were found, but warnings should be reviewed before upload.
- `BLOCKED_BY_AUTHOR_METADATA` means the manuscript/package is structurally ready, but final author-provided submission metadata are incomplete.
- `FAIL` means a non-author-metadata artifact, package, workbook, or validation check failed and must be fixed before submission.
