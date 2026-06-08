# Submission Artifact Quality Audit

This audit performs structural checks on DOCX files, figure rasters, visual DOCX-to-PDF QA outputs, and the assembled submission package.

## DOCX Structure

| File | Paragraphs | Tables | Figures | Placeholder tokens | Author-query text | Status |
|---|---:|---:|---:|---|---|---|
| ResidualAware_SSL_CNN_Nature_Manuscript.docx | 132 | 4 | 6 | none | True | WARN |
| ResidualAware_SSL_CNN_Nature_Clean_Placeholder.docx | 124 | 4 | 6 | none | False | PASS |
| ResidualAware_SSL_CNN_JNE_Structured_Manuscript.docx | 137 | 4 | 6 | none | True | WARN |
| ResidualAware_SSL_CNN_JNE_Clean_Placeholder.docx | 129 | 4 | 6 | none | False | PASS |
| ResidualAware_SSL_CNN_Supplementary_Information.docx | 66 | 11 | 4 | none | False | PASS |
| Author_Required_Information_Form.docx | 14 | 12 | 0 | none | False | PASS |
| Author_Quick_Response_Request_ZH.docx | 166 | 0 | 0 | none | False | PASS |

## DOCX Visual QA

| Report | Documents | Page checks | Failed pages | Status |
|---|---:|---:|---:|---|
| docx_visual_qa_report.md | 6 | 75 | 0 | PASS |

## Reference Metadata

| Audit file | References | DOI rows | URL rows | Failed/Warned | Status |
|---|---:|---:|---:|---:|---|
| reference_metadata_audit.csv | 25 | 20 | 5 | 0 | PASS |

## Source-Data Workbook

| Workbook | Sheet count | Required sheets | Status |
|---|---:|---|---|
| ResidualAware_SSL_CNN_Source_Data.xlsx | 36 | all present | PASS |

Required sheet set checked: Artifact_QA, Author_Evidence, Claim_Audit, Data_Audit, Data_Dictionary, Docx_Visual_QA, FigureManifest, Local_Refs, MNE_WPLI_Figures, Manuscript_Audit, Methods_Provenance, Model_Card, PatientPDF_Audit, Precision_Audit, README, Reference_Metadata, Replacement_Map, Source_Metadata_Audit, Validation_Audit.

## Figure Raster Checks

| Figure | Width px | Height px | RGB mean | RGB std | Status |
|---|---:|---:|---|---|---|
| figure1_study_design_model.png | 3742 | 2527 | (243.0, 242.9, 242.4) | (44.2, 44.3, 44.2) | PASS |
| figure2_performance_calibration.png | 3820 | 2627 | (231.2, 228.2, 227.3) | (60.1, 62.0, 64.7) | PASS |
| figure3_robustness_ablation.png | 3934 | 2627 | (222.1, 225.5, 223.5) | (67.2, 60.1, 64.0) | PASS |
| figure4_explainability_neurophysiology.png | 3562 | 2529 | (232.4, 238.5, 245.4) | (63.0, 45.7, 32.6) | PASS |
| supplementary_error_subjects.png | 3560 | 1069 | (246.4, 244.0, 245.0) | (35.2, 44.8, 42.3) | PASS |
| supplementary_performance_precision.png | 3962 | 2408 | (234.7, 233.6, 235.1) | (58.1, 58.3, 56.3) | PASS |
| supplementary_clinical_incremental_value.png | 3684 | 2761 | (238.7, 235.6, 232.7) | (50.9, 54.6, 60.6) | PASS |
| nature_figures_contact_sheet.png | 1920 | 2520 | (242.1, 242.1, 242.4) | (43.3, 41.3, 41.6) | PASS |
| mne_wpli_connectivity_contact_sheet.png | 2650 | 2038 | (244.3, 244.0, 245.7) | (39.9, 39.6, 36.8) | PASS |

## Package Integrity

- Manifest rows: 261
- Missing package paths: 0
- Zip entries: 261
- Zip integrity result: None
- Status: PASS

## Interpretation

- Main and JNE working manuscript DOCX files retain author-query text by design. Clean placeholder variants remove those query lines but still require author-supplied ethics, consent, data-access, and protocol fields before final upload.
- Reference metadata audit resolved all current DOI/URL reference identifiers; final journal style still depends on the target journal.
- No generic placeholder tokens were detected in the DOCX manuscript text.
- Figure rasters are non-blank by variance checks and exceed the minimum inspected dimensions.
- DOCX visual QA was performed through Microsoft Word COM PDF export and pypdfium2 page rendering. Before final journal upload, still open the PDF previews in Word/Adobe/Edge and inspect page breaks, table wrapping, figure sharpness, and target-journal template requirements.

