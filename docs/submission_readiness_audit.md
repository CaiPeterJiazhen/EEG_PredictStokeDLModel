# Submission Readiness Audit

## Current Deliverables

| Requirement | Evidence | Status |
|---|---|---|
| Nature-style language polishing | `docs/manuscript_residual_aware_ssl_cnn_nature_polished.md` | Completed draft |
| Clean Nature placeholder manuscript | `docs/manuscript_residual_aware_ssl_cnn_nature_clean_placeholder.md`; `output/doc/ResidualAware_SSL_CNN_Nature_Clean_Placeholder.docx` | Completed; author-query text removed, author-supplied ethics/protocol fields still required |
| Journal of Neural Engineering structured-abstract variant | `docs/manuscript_residual_aware_ssl_cnn_jne_structured.md`; `output/doc/ResidualAware_SSL_CNN_JNE_Structured_Manuscript.docx` | Completed draft, author fields pending |
| Clean JNE placeholder manuscript | `docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md`; `output/doc/ResidualAware_SSL_CNN_JNE_Clean_Placeholder.docx` | Completed; author-query text removed, author-supplied ethics/protocol fields still required |
| Complete citation layer | `docs/citation_artifacts/selected_references.md`, `selected_references.ris`, `manuscript_citation_map.md`; `docs/reference_metadata_audit.md`; `results/tables/reference_metadata_audit.csv` | Completed draft; 25 reference identifiers pass DOI/URL metadata checks, final style pending target journal |
| Data/statistical analysis integration | `results/statistics/*.csv`, `docs/statistical_validation_summary.md`, `outputs/manuscript_package/ResidualAware_SSL_CNN_Source_Data.xlsx` | Completed for available data |
| Source-data dictionary and repository README | `results/tables/source_data_dictionary.csv`; `docs/repository_readme_for_deposit.md` | Completed draft with 446 field definitions across 37 derived CSV or figure-manifest files; DOI/licence/access fields pending |
| Manuscript figures | `results/figures/nature/figure_manifest.csv`, Figure 1-4, Supplementary Figure 1, MNE WPLI Supplementary Figure 2, and Supplementary Figure 3 performance-precision exports | Completed draft; Figure 1 now includes de-identified participant-flow counts from `results/tables/participant_flow_safety_source_notes.csv` |
| Word manuscript | `output/doc/ResidualAware_SSL_CNN_Nature_Manuscript.docx` | Completed draft |
| Supplementary Information document | `output/doc/ResidualAware_SSL_CNN_Supplementary_Information.docx` | Completed draft |
| Assembled submission package | `outputs/submission_package_20260601`; `outputs/ResidualAware_SSL_CNN_submission_package_20260601.zip` | Completed package with manifest and checksums |
| Figure/table placement map | `docs/manuscript_figure_table_map.md` | Completed |
| Data availability/FAIR statement | `docs/data_availability_and_fair_audit.md` | Completed draft, author fields pending |
| Methods provenance audit | `docs/methods_detail_provenance.md` | Completed for project-verifiable details |
| TRIPOD+AI-oriented reporting checklist | `docs/tripod_ai_reporting_checklist.md` | Completed draft, journal form pending |
| EEG and clinical source metadata audit | `docs/eeg_metadata_audit.md`; `results/tables/eeg_recording_metadata_audit.csv`; `results/tables/eeg_recording_summary.csv`; `results/tables/clinical_workbook_structure_audit.csv` | Completed non-identifying audit |
| Methods gap resolution audit | `docs/methods_gap_resolution_from_project_files.md` | Completed; current feature-pipeline preprocessing boundary clarified |
| Author-required evidence trace | `docs/author_required_evidence_trace.md`; `results/tables/author_required_evidence_trace.csv` | Completed reproducible keyword sweep for ethics, consent, registration, recruitment, eligibility, tACS, rehabilitation, safety, EEG acquisition/preprocessing, and repository fields |
| Author field replacement map | `docs/author_field_replacement_map.md`; `results/tables/author_field_replacement_map.csv` | Completed map from author-supplied fields to manuscript, declaration, cover-letter, repository, and verification targets |
| Author submission metadata template | `docs/author_submission_metadata_template.json`; `docs/author_submission_metadata_validation_report.md` | Completed machine-readable template and validation report for author-supplied final submission fields |
| Author metadata insertion protocol | `docs/author_metadata_insertion_protocol.md`; `scripts/64_build_author_metadata_insertion_protocol.py` | Completed dry-run protocol mapping validated author metadata to final manuscript, declaration, cover-letter, repository, and verification targets |
| Author metadata intake workbook | `outputs/manuscript_package/Author_Submission_Metadata_Intake.xlsx`; `scripts/65_build_author_metadata_intake_workbook.cjs` | Completed bilingual 8-sheet XLSX workbook for author responses, a directly importable `Minimal_Completion` sheet, evidence sources, target-file mapping, project prefill, source-workbook evidence grades, and validation workflow |
| Author metadata intake import | `outputs/manuscript_package/author_submission_metadata_from_intake.json`; `docs/author_metadata_intake_import_report.md`; `scripts/66_import_author_metadata_intake_workbook.py` | Completed workbook-to-JSON import path for author responses before strict validation and final insertion |
| Final submission gate | `docs/final_submission_gate_report.md`; `scripts/67_run_final_submission_gate.py`; `tests/test_final_submission_gate.py` | Completed consolidated readiness gate across manuscript, DOCX visual QA, artifact audit, prediction validation, source workbook, author metadata, and package integrity |
| Author metadata project prefill | `outputs/manuscript_package/author_submission_metadata_project_prefill.json`; `docs/author_metadata_project_prefill_report.md`; `scripts/68_build_author_metadata_project_prefill.py` | Completed 15-value project-evidence prefill draft for metadata values already supported by audits, provenance files, and de-identified review of the M1 clinical source workbook |
| Author minimal completion pack | `docs/author_minimal_completion_pack.md`; `outputs/manuscript_package/author_minimal_completion_answers.json`; `results/tables/author_minimal_completion_pack.csv`; `scripts/76_build_author_minimal_completion_pack.py` | Completed compact author-response pack covering the 18 fields that currently prevent final replacement, with project-prefill suggestions separated from author-approved answers |
| Local reference audit and positioning matrix | `docs/local_reference_audit.md`; `docs/local_reference_positioning_matrix.md`; `results/tables/local_reference_positioning_matrix.csv` | Completed; 14 locally supplied prognostic-model PDFs screened, three additional directly relevant references added, off-domain analogues retained as guarded background rather than overextended citations |
| Submission artifact quality audit | `docs/submission_artifact_quality_audit.md`; `results/tables/submission_artifact_quality_audit.csv` | Completed structural and visual QA using Microsoft Word COM PDF export and pypdfium2 page rendering |
| Pre-submission editorial readiness audit | `docs/presubmission_editorial_readiness_audit.md`; `scripts/75_audit_presubmission_editorial_readiness.py` | Completed editorial preflight for title length, abstract length, clean variants, citation/availability coverage, DOCX visual QA, package integrity, and final-gate status |
| DOCX visual QA report | `docs/docx_visual_qa_report.md`; `results/tables/docx_visual_qa_metrics.csv`; `outputs/doc_visual_qa/pdf/*.pdf`; `outputs/doc_visual_qa/contact_sheets/*.png` | Completed; 72 page-image checks passed across 6 DOCX files, 0 failed |
| Manuscript-source integrity audit | `docs/manuscript_integrity_audit.md`; `results/tables/manuscript_integrity_audit.csv` | Completed; citation numbering, DOI/URL presence, key numeric claims, table/figure source paths, figure exports, and clean manuscript variants pass current automated checks |
| Prediction validation and leakage integrity audit | `docs/prediction_validation_integrity_audit.md`; `results/tables/prediction_validation_integrity_audit.csv` | Completed machine-checkable audit of patient-level n=19 validation unit, LOSO/fold-local feature-selection safeguards, subject-level bootstrap/permutation checks, paired comparison structure, and seed-summary interpretation |
| PROBAST/TRIPOD+AI risk audit | `docs/probast_tripod_ai_risk_audit.md`; `results/tables/probast_tripod_ai_risk_audit.csv` | Completed conservative prediction-model risk audit; high-risk/some-concern items retained for small sample size, no external validation, cohort-derived threshold, and missing author-confirmed protocol metadata |
| Performance precision audit | `docs/performance_precision_audit.md`; `results/tables/performance_precision_audit.csv` | Completed validation-boundary audit quantifying one-case metric resolution, final-model bootstrap interval width, paired-comparison uncertainty, and above-chance testing |
| Claim-strength audit | `docs/claim_strength_audit.md`; `results/tables/claim_strength_audit.csv` | Completed evidence-to-wording audit for central manuscript claims and overclaims to avoid |
| AI model reporting card | `docs/model_reporting_card.md`; `results/tables/model_reporting_card.csv` | Completed intended-use, validation-safeguard, reproducibility-action, deployment-boundary, and unsupported-use summary for Supplementary Table 11 |
| Source-workbook author metadata audit | `docs/source_workbook_author_metadata_audit.md`; `results/tables/source_workbook_author_metadata_audit.csv` | Completed non-identifying scan of the three M1 source workbooks for ethics, consent, registration, site/date, eligibility, tACS, rehabilitation, safety, EEG acquisition, preprocessing, repository, and declaration evidence |
| Target-journal strategy matrix | `docs/target_journal_strategy.md` | Updated with 2026-06-02 public source check, APC/scope/format notes, recommended submission order, and CAS/JCR verification boundary |
| JNE submission checklist | `docs/jne_submission_checklist.md` | Completed practical upload map for the JNE-first package; author-supplied ethics, consent, registration, data/code, and declaration fields remain blocking |
| JNE final declaration templates | `docs/jne_final_declaration_templates.md` | Completed ready-to-paste ethics, consent, registration, data/code availability, competing-interest, funding, contribution, and acknowledgement template set; data/code language aligned to the 35-sheet workbook and audit package |
| Cover-letter template | `docs/cover_letter_template.md` | Updated to a JNE-first cover letter with conservative pilot framing, explicit contribution points, author-confirmation placeholders, and current source-data/code-availability wording |
| Author-required information form | `docs/author_required_information_form.md`; `output/doc/Author_Required_Information_Form.docx` | Completed bilingual form for ethics, consent, EEG acquisition, preprocessing, tACS, data sharing, declarations, and final statements; now points authors to the minimal completion pack as the fastest path |

## Verification Performed

| Check | Result |
|---|---|
| Figure script syntax | `python -m py_compile scripts\45_make_nature_manuscript_figures.py` passed |
| Figure regeneration | `python scripts\45_make_nature_manuscript_figures.py` completed |
| MNE topomap/connectivity tests | `python -m pytest tests\test_mne_topomap_coordinates.py tests\test_mne_wpli_connectivity.py` passed, 2 tests |
| Manuscript DOCX generation | `scripts/46_build_submission_docx.py` generated DOCX |
| Manuscript DOCX structural check | 4 tables and 6 embedded figures |
| Supplementary DOCX generation | `scripts/49_build_supplementary_information_docx.py` generated DOCX |
| Supplementary DOCX structural check | 11 supplementary tables and 3 embedded figures |
| Source-data dictionary generation | `scripts/54_build_repository_data_dictionary.py` generated source-data dictionary and repository README draft |
| Source-data workbook generation | `scripts/47_build_source_data_workbook.cjs` generated 35-sheet workbook |
| Source-data workbook check | README, Data_Audit, Data_Dictionary, Methods_Provenance, Manuscript_Audit, Reference_Metadata, Local_Refs, Artifact_QA, Docx_Visual_QA, Model_Card, Validation_Audit, Source_Metadata_Audit, Author_Evidence, Replacement_Map, EEG metadata, figure/table/statistics sheets present |
| Manuscript-source integrity audit | `scripts/56_audit_manuscript_integrity.py` completed with 76 PASS, 0 WARN, and 0 FAIL |
| DOCX visual QA | `scripts/57_docx_visual_qa_word_pdf.py` exported DOCX files with Microsoft Word COM, rendered PDF pages with pypdfium2, and completed 72 page checks with 0 failures |
| Reference metadata audit | `scripts/58_verify_reference_metadata.py` completed with 25 PASS lookup checks and 25 PASS match checks using Crossref DOI metadata or URL reachability |
| Local reference positioning matrix | `scripts/59_build_local_reference_positioning_matrix.py` screened the 14 locally supplied prognostic-model PDFs and generated a conservative citation-positioning matrix |
| Prediction validation and leakage integrity audit | `scripts/60_audit_prediction_validation_integrity.py` completed with 21 PASS, 0 WARN, and 0 FAIL checks for patient-level validation safeguards |
| Author-required evidence trace | `scripts/61_audit_author_required_evidence.py` completed with 11 submission-critical fields traced to source evidence or author-required status |
| Author field replacement map | `scripts/62_build_author_field_replacement_map.py` generated 19 replacement targets for final author-field insertion |
| Author submission metadata validation | `scripts/63_validate_author_submission_metadata.py` checked 19 author metadata fields and reports final-insertion readiness |
| Author metadata insertion protocol | `scripts/64_build_author_metadata_insertion_protocol.py` generated dry-run insertion protocol and prevents final replacement while strict metadata validation fails |
| Author metadata intake workbook | `scripts/65_build_author_metadata_intake_workbook.cjs` generated the 8-sheet XLSX intake workbook and preview for author completion, including `Minimal_Completion`, `Project_Prefill`, and `Source_Workbook_Audit` sheets |
| Author metadata intake import | `scripts/66_import_author_metadata_intake_workbook.py` generated converted JSON and import report from the XLSX workbook |
| Final submission gate | `scripts/67_run_final_submission_gate.py` generated consolidated gate report; current overall status is blocked by missing author metadata |
| Author metadata project prefill | `scripts/68_build_author_metadata_project_prefill.py` generated a separate 15-value project-evidence prefill JSON and report without overwriting the author-approved metadata template; newly added entries cover follow-up window, disease-duration evidence caveat, FMA-UE scoring maximum, EO/EC file-state convention, and participant-flow/safety source notes |
| Author minimal completion pack | `scripts/76_build_author_minimal_completion_pack.py` generated an 18-field, 104-row compact author-response pack from the active validation rules, replacement map, and project-prefill file |
| Submission package assembly | `scripts/50_assemble_submission_package.py` generated 206-file package and zip |
| Submission package check | Manifest has 206 rows, no missing package paths, zip integrity test returned no corrupt entry |
| Pre-submission editorial readiness audit | `scripts/75_audit_presubmission_editorial_readiness.py` completed with 22 PASS, 0 WARN, and 1 author-metadata BLOCKED item |
| DOCX/XLSX/reference/figure structural and visual QA | `scripts/52_audit_submission_artifacts.py` checked manuscript and author-form DOCX table/figure counts, source-data workbook sheets, reference metadata audit outputs, placeholder tokens, author-query blockers, DOCX visual QA outputs, figure dimensions, raster variance, and package integrity; latest status is 42 PASS and 6 expected WARN |

## Blocking Author Inputs Before Submission

These fields cannot be inferred safely from the project files and must be supplied by the author:

1. Ethics approval institution, approval number, and consent/data-sharing wording.
2. Whether standardized conventional rehabilitation was delivered concurrently with the tACS protocol.
3. Recruitment dates, inclusion/exclusion criteria, stroke subtype, lesion-side handling beyond affected-hand alignment, and timing from stroke to EEG/tACS.
4. EEG acquisition details not already present in code: device, cap/montage description, reference, preprocessing filters, artifact rejection, and epoching criteria before feature extraction. Recording duration is now estimated from `.set` metadata.
5. Data-sharing decision for raw EEG, minimally processed EEG, derived PSD/WPLI features, clinical variables, code, and source tables.
6. Target journal and reference style.

## Scientific Risk Flags To Preserve

- Supervised cohort is n=19, with no external validation.
- Clinical-only baseline is strong and exploratory; current data do not prove EEG incremental clinical value.
- Residual-aware SSL-CNN improves ranking/calibration-oriented metrics but not hard-label accuracy relative to the no-SSL CNN.
- Explainability findings are model-dependent and hypothesis-generating.
- Seed-level and segment-level rows must not be treated as independent patients.
- The residual threshold is cohort-median-derived and must not be framed as an externally validated clinical cut-off.
