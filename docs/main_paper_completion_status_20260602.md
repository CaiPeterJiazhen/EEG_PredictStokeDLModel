# Main Paper Completion Status, 2026-06-02

## Scope

This report evaluates the manuscript main-body objective after the author deferred ethics approval, informed consent, registration, conventional-rehabilitation protocol, device details, repository DOI, and licence fields. Those deferred items remain necessary for final journal upload, but they are not treated as blockers for completing the main scientific manuscript draft.

## Current Status

Main scientific manuscript body: PASS for current project evidence.

Submission upload readiness: not final, pending author/institutional metadata.

## Completed Manuscript Components

| Component | Status | Evidence |
|---|---|---|
| Lin 2022 reading and structure transfer | PASS | `docs/lin_2022_nature_reader/paper.md`; `docs/lin_2022_nature_reader/lin_section_figure_blueprint.md` |
| Main manuscript text | PASS | `docs/manuscript_residual_aware_ssl_cnn_nature_polished.md` |
| Clean manuscript variants | PASS | `output/doc/ResidualAware_SSL_CNN_Nature_Clean_Placeholder.docx`; `output/doc/ResidualAware_SSL_CNN_JNE_Clean_Placeholder.docx` |
| Supplementary information | PASS | `output/doc/ResidualAware_SSL_CNN_Supplementary_Information.docx` |
| Main tables | PASS | `results/tables/table1_cohort_characteristics.csv`; `results/tables/table2_main_model_performance.csv`; `results/tables/table3_ablation.csv`; `results/tables/table4_explainability_biomarkers.csv` |
| Main figures | PASS | `results/figures/nature/figure1_study_design_model.*`; `figure2_performance_calibration.*`; `figure3_robustness_ablation.*`; `figure4_explainability_neurophysiology.*` |
| MNE topomap figures | PASS | `results/figures/explainability/mne_topomaps/mne_topomap_contact_sheet.png`; `mne_topomap_manifest.csv`; 16 MNE topomap rows with PNG/SVG exports |
| MNE WPLI connectivity figures | PASS | `results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_contact_sheet.png`; 12 MNE connectivity rows with PNG/SVG/PDF exports |
| Source-data workbook | PASS | `outputs/manuscript_package/ResidualAware_SSL_CNN_Source_Data.xlsx`, 36 sheets |
| Data availability and FAIR planning | PASS with deferred DOI/licence | `docs/data_availability_and_fair_audit.md` |
| Citation coverage | PASS | `docs/citation_claim_coverage_audit.md`, 25/25 references cited, 14/14 claim requirements passing |
| Numeric claim trace | PASS | `docs/numeric_claim_source_trace_audit.md`, 20/20 numeric claims passing |
| Figure/table readiness | PASS | `docs/figure_submission_readiness_audit.md` |
| Manuscript-source integrity | PASS | `docs/manuscript_integrity_audit.md`, PASS 78, WARN 0, FAIL 0 |
| Submission package structure | PASS | `docs/submission_artifact_quality_audit.md`, manifest rows 261, zip entries 261, zip integrity result None |

## Verification Run

- `python scripts\45_make_mne_explainability_topomaps.py --formats png svg`: generated 16 MNE topomap sets and `mne_topomap_contact_sheet.png`.
- `python scripts\56_audit_manuscript_integrity.py`: PASS 78, WARN 0, FAIL 0.
- `python scripts\79_audit_figure_submission_readiness.py`: overall PASS; includes topomap and connectivity checks.
- `python scripts\80_audit_citation_claim_coverage.py`: overall PASS; invalid citations 0; overclaim warnings 0.
- `python scripts\81_audit_numeric_claim_source_trace.py`: status counts PASS:20.
- `python -m pytest tests\test_mne_topomap_coordinates.py tests\test_mne_wpli_connectivity.py -q`: 2 passed.
- `python scripts\50_assemble_submission_package.py`: packaged 261 files.
- Bundled Python `scripts\52_audit_submission_artifacts.py`: package integrity PASS, missing package paths 0, zip entries 261.

## Deferred Author-Metadata Items

The following are intentionally deferred based on the updated author objective:

- Ethics approval and informed-consent wording.
- Trial or study registration statement.
- Concurrent rehabilitation protocol and tACS device/electrode details.
- EEG acquisition hardware and upstream raw-preprocessing protocol.
- Repository DOI/accession, data licence, code licence, and controlled-access route.
- Final author list, affiliations, funding, competing interests, and target journal reference style.

These fields are documented in `docs/author_quick_response_request_zh.md`, `docs/author_minimal_completion_pack.md`, and `docs/author_submission_metadata_validation_report.md`.

## Interpretation

The paper's main scientific content is complete for the currently available project evidence: text, results, statistics, tables, figures, MNE topomap/connectivity outputs, citations, source-data workbook, and package integrity all have passing audits. The manuscript should still be treated as a high-quality working submission draft rather than a final upload file until the deferred author/institutional fields are supplied and inserted.
