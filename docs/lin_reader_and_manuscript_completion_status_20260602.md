# Lin Reader And Manuscript Completion Status

Date: 2026-06-02

## Completed In This Pass

### Lin 2022 nature-reader output

Created a source-grounded structural reader for the Lin 2022 reference paper:

- `docs/lin_2022_nature_reader/paper.md`
- `docs/lin_2022_nature_reader/lin_section_figure_blueprint.md`
- `docs/lin_2022_nature_reader/translation_notes.md`
- `docs/lin_2022_nature_reader/source_map.json`
- `docs/lin_2022_nature_reader/image_manifest.json`
- `docs/lin_2022_nature_reader/assets/`

The reader extracts Lin's section functions, figure/table placement logic, and direct implications for the residual-aware EEG/tACS manuscript. It also extracts page-level PNG assets and embedded figure assets from the PDF.

### Manuscript and submission package already available

The current manuscript package already contains:

- Nature-style polished manuscript: `docs/manuscript_residual_aware_ssl_cnn_nature_polished.md`
- Clean Nature placeholder manuscript: `docs/manuscript_residual_aware_ssl_cnn_nature_clean_placeholder.md`
- Revised initial Chinese/English drafts: `docs/manuscript_revised_initial_draft_zh.md`; `docs/manuscript_revised_initial_draft.md`
- JNE structured variant: `docs/manuscript_residual_aware_ssl_cnn_jne_structured.md`
- Main DOCX manuscript: `output/doc/ResidualAware_SSL_CNN_Nature_Manuscript.docx`
- Supplementary Information DOCX: `output/doc/ResidualAware_SSL_CNN_Supplementary_Information.docx`
- Source-data workbook: `outputs/manuscript_package/ResidualAware_SSL_CNN_Source_Data.xlsx`
- Submission package: `outputs/submission_package_20260601`
- Submission package zip: `outputs/ResidualAware_SSL_CNN_submission_package_20260601.zip`

### Figure package

Main manuscript figure map:

- Figure 1: `results/figures/nature/figure1_study_design_model.png`
- Figure 2: `results/figures/nature/figure2_performance_calibration.png`
- Figure 3: `results/figures/nature/figure3_robustness_ablation.png`
- Figure 4: `results/figures/nature/figure4_explainability_neurophysiology.png`
- Figure manifest: `results/figures/nature/figure_manifest.csv`
- Nature figure contact sheet: `results/figures/nature/nature_figures_contact_sheet.png`

MNE explainability outputs:

- PSD topomaps: `results/figures/explainability/mne_topomaps/`
- WPLI connectivity maps: `results/figures/explainability/mne_wpli_connectivity/`
- WPLI contact sheet: `results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_contact_sheet.png`

## Verification Run

| Check | Result |
|---|---|
| Lin reader files and JSON | PASS: required files exist; `source_map.json` parses; 7 extracted image assets present; 9 page assets present |
| MNE plotting script syntax | PASS: `scripts\45_make_mne_explainability_topomaps.py`, `scripts\46_make_mne_wpli_connectivity.py`, and `scripts\83_make_revised_initial_manuscript_figures.py` compile |
| MNE tests | PASS: `python -m pytest tests\test_mne_topomap_coordinates.py tests\test_mne_wpli_connectivity.py -q` returned `2 passed` |
| Citation claim coverage | PASS: `python scripts\80_audit_citation_claim_coverage.py` reports 14 claim requirements, 0 invalid citations, 0 overclaim warnings |
| Reference metadata | PASS: `python scripts\58_verify_reference_metadata.py` regenerated `docs/reference_metadata_audit.md` and `results/tables/reference_metadata_audit.csv` |
| Final submission gate | BLOCKED_BY_AUTHOR_METADATA: 7 PASS, 1 WARN, 2 BLOCKED, 0 FAIL |
| Figure contact sheets | PASS: Nature, revised-initial, and MNE WPLI contact sheets are present and nonblank |

## Remaining Blocking Author Inputs

These fields cannot be inferred safely from project data and must be supplied by the author before final journal upload:

1. Ethics approval institution, approval number, consent wording, and whether consent covers EEG/tACS/clinical data sharing.
2. Trial or study registration number, or an author-approved statement that the study was not registered if applicable.
3. Recruitment dates, inclusion/exclusion criteria, stroke subtype rules, and timing from stroke to EEG/tACS.
4. Whether standardized conventional rehabilitation was delivered concurrently with tACS.
5. EEG acquisition details: amplifier, acquisition software, cap/montage, online reference, ground, impedance threshold, raw filtering, notch filtering, re-reference, ICA/artifact rejection, bad-channel handling, and export rules before `.set/.fdt`.
6. FMA-UE assessor qualification, blinding status, exact assessment timing, and scale/scoring reference.
7. Data/code repository name, DOI/accession, version, licence, public data scope, restricted data scope, access committee/contact, and request-review conditions.
8. Final target journal and exact reference style.

## Scientific Boundaries To Preserve

- The supervised cohort is n=19 and has no external validation.
- The cohort-median residual threshold is a modelling endpoint, not an externally validated clinical cut-off.
- Clinical-only models are strong in this dataset; EEG incremental value is not proven.
- Residual-aware auxiliary supervision is better supported than an independent SSL performance gain.
- Explainability results are model-dependent and hypothesis-generating, not causal EEG biomarkers.
- Segment-level or seed-level rows must never be treated as independent patients.

