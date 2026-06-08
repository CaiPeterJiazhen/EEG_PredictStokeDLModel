# Objective Completion Audit

Date: 2026-06-02

This audit checks the active thread objective against the current worktree. It does not redefine completion as "files exist". A requirement is considered complete only when current-state files, command output, or generated artifacts directly support it.

## Verification Commands Run

```text
python -m pytest tests\test_mne_topomap_coordinates.py tests\test_mne_wpli_connectivity.py tests\test_final_submission_gate.py -q
python -m py_compile scripts\45_make_mne_explainability_topomaps.py scripts\46_make_mne_wpli_connectivity.py scripts\45_make_nature_manuscript_figures.py scripts\83_make_revised_initial_manuscript_figures.py scripts\67_run_final_submission_gate.py scripts\80_audit_citation_claim_coverage.py scripts\58_verify_reference_metadata.py
python scripts\80_audit_citation_claim_coverage.py
python scripts\67_run_final_submission_gate.py
```

Observed results:

- Tests: `5 passed`.
- Py-compile: passed with no output.
- Citation coverage: `overall_status=PASS`, `claim_requirements=14`, `invalid_citations=0`, `overclaim_warnings=0`.
- Final submission gate: `overall_status=BLOCKED_BY_AUTHOR_METADATA`, `status_counts=BLOCKED:2,PASS:7,WARN:1`.

## Requirement-by-Requirement Status

| Objective item | Current evidence | Status | Remaining action |
|---|---|---|---|
| Use nature-reader to understand Lin 2022 and what each section/figure should contain | `docs/lin_2022_nature_reader/paper.md`; `docs/lin_2022_nature_reader/lin_section_figure_blueprint.md`; `docs/lin_2022_nature_reader/source_map.json`; 151 source blocks; 7 extracted image assets; 9 rendered page assets | Complete as a structural, source-grounded reader | None for structural use. A full paragraph-level bilingual translation could be made later, but the current objective was section/figure understanding for manuscript writing. |
| Use nature-writing to write manuscript text based on Lin and the initial draft | `docs/manuscript_residual_aware_ssl_cnn_nature_polished.md`; `docs/manuscript_revised_initial_draft.md`; `docs/manuscript_revised_initial_draft_zh.md`; `docs/lin_2022_nature_reader/lin_section_figure_blueprint.md` | Complete for draft manuscript text | Final author metadata must be inserted before upload-ready final text. |
| Use nature-polishing to polish academic language | `docs/manuscript_residual_aware_ssl_cnn_nature_polished.md`; clean variants; `docs/manuscript_integrity_audit.md` has 78 PASS, 0 WARN, 0 FAIL | Complete for polished draft | Final journal-specific style can only be fixed after target journal is confirmed. |
| Use nature-citation to complete references | `docs/citation_claim_coverage_audit.md`; `docs/reference_metadata_audit.md`; `docs/citation_artifacts/selected_references.ris`; `docs/citation_artifacts/manuscript_citation_map.md` | Complete for current manuscript claims | Rerun citation coverage if new claims are added after author metadata insertion. |
| Use nature-data to complete data availability and statistics | `docs/data_availability_and_fair_audit.md`; `outputs/manuscript_package/ResidualAware_SSL_CNN_Source_Data.xlsx`; `results/statistics/*.csv`; `results/tables/source_data_dictionary.csv`; source-data workbook has 36 sheets | Complete for derived data/statistical package | Repository DOI, licence, public/restricted scope and controlled-access route require author/institutional confirmation. |
| Complete required statistical analysis from project data | `docs/statistical_validation_summary.md`; `docs/manuscript_integrity_audit.md`; `results/statistics/model_metric_confidence_intervals.csv`; `model_pairwise_comparisons.csv`; `model_permutation_tests.csv`; final gate validation PASS | Complete for current available dataset | No external validation exists; manuscript correctly preserves this limitation. |
| Use nature-figure to complete required figures | `results/figures/nature/figure_manifest.csv`; Figure 1-4 and supplementary figures with PNG/SVG/PDF/TIFF exports; `docs/manuscript_figure_table_map.md`; artifact QA figure raster PASS | Complete for current manuscript figure package | If final target journal requires fully editable vector panels for raster-embedded explainability panels, regenerate those specific panels. |
| Draw topomap/connectivity using MNE-Python or existing generated figures | `results/figures/explainability/mne_topomaps/mne_topomap_manifest.csv`; `results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_manifest.csv`; tests `test_mne_topomap_coordinates.py` and `test_mne_wpli_connectivity.py` passed | Complete | None unless new attribution tables are generated. |
| Use all project data, ask for missing data if needed | Project-derived data are integrated into tables, figures, source workbook and audits; author metadata validation identifies missing non-inferable items | Partially complete | Author must supply ethics, consent, study-design, EEG acquisition/preprocessing, safety, repository, code and declaration fields. |
| Final target: high-completeness, clear logic, complete figures/tables, academic-paper standard, high-impact/CAS Q1 level | Draft package, figures, source data, audits, DOCX, SI and submission zip exist; final gate has 7 PASS, 1 WARN, 2 BLOCKED, 0 FAIL | Not fully complete | Final submission is blocked by author metadata. Also, true CAS Q1-level readiness depends on target-journal fit and author-confirmed protocol transparency. |

## Current Deliverables

| Artifact | Path |
|---|---|
| Lin 2022 structural reader | `docs/lin_2022_nature_reader/paper.md` |
| Lin-to-manuscript blueprint | `docs/lin_2022_nature_reader/lin_section_figure_blueprint.md` |
| Nature polished manuscript | `docs/manuscript_residual_aware_ssl_cnn_nature_polished.md` |
| Nature clean placeholder manuscript | `docs/manuscript_residual_aware_ssl_cnn_nature_clean_placeholder.md` |
| Main Nature DOCX | `output/doc/ResidualAware_SSL_CNN_Nature_Manuscript.docx` |
| Clean Nature DOCX | `output/doc/ResidualAware_SSL_CNN_Nature_Clean_Placeholder.docx` |
| JNE structured manuscript | `output/doc/ResidualAware_SSL_CNN_JNE_Structured_Manuscript.docx` |
| Supplementary Information DOCX | `output/doc/ResidualAware_SSL_CNN_Supplementary_Information.docx` |
| Source-data workbook | `outputs/manuscript_package/ResidualAware_SSL_CNN_Source_Data.xlsx` |
| Submission package | `outputs/submission_package_20260601` |
| Submission zip | `outputs/ResidualAware_SSL_CNN_submission_package_20260601.zip` |
| Author minimal completion pack | `docs/author_minimal_completion_pack.md` |
| Author quick response request | `docs/author_quick_response_request_zh.md` |
| Final gate report | `docs/final_submission_gate_report.md` |

## Blocking Author Metadata

The final gate is not ready for submission because strict metadata validation reports:

- Fields checked: 19.
- Complete fields: 1.
- Incomplete blocking or blocking-if-applicable fields: 12.
- Incomplete high-priority fields: 6.

Blocking fields:

1. `ethics_approval`
2. `informed_consent`
3. `trial_or_study_registration`
4. `study_site_dates_design`
5. `eligibility_stroke_timing`
6. `tacs_device_electrodes`
7. `concurrent_rehabilitation`
8. `tacs_safety_adverse_events`
9. `eeg_hardware_reference_impedance`
10. `raw_eeg_preprocessing`
11. `data_repository_doi_scope`
12. `code_repository_license`

High-priority fields:

1. `target_journal_reference_style`
2. `author_list_affiliations`
3. `author_contributions`
4. `fma_assessors_timing`
5. `resting_state_instructions`
6. `funding_competing_acknowledgements`

## Completion Conclusion

The local manuscript-production work is complete for the evidence currently available in the project. The full active goal is not yet complete because final submission readiness depends on author-approved human-participant, protocol, device, safety, repository, code, authorship and declaration metadata that cannot be inferred safely from the repository.

