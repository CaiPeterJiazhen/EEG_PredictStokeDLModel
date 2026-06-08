# Manuscript Figure And Table Map

This map links the polished manuscript sections to generated figures, manuscript tables, and source data files.

## Main text placement

| Placement | Item | Manuscript section | Source |
|---|---|---|---|
| After Introduction or early Methods overview | Figure 1 | Participant flow, study design, and residual-aware SSL-CNN architecture | `results/figures/nature/figure1_study_design_model.png`; `results/tables/participant_flow_safety_source_notes.csv` |
| Results, Cohort and outcome definition | Table 1 | Cohort characteristics | `results/tables/table1_cohort_characteristics.csv` |
| Results, Residual-aware SSL-CNN performance | Figure 2 | Main performance, uncertainty, calibration | `results/figures/nature/figure2_performance_calibration.png` |
| Results, Residual-aware SSL-CNN performance | Table 2 | Main model performance | `results/tables/table2_main_model_performance.csv` |
| Results, Ablation and robustness analyses | Figure 3 | Robustness, ablation, threshold sensitivity | `results/figures/nature/figure3_robustness_ablation.png` |
| Results, Ablation and robustness analyses | Table 3 | Ablation analysis | `results/tables/table3_ablation.csv` |
| Results, Model explanation and EEG biomarker localization | Figure 4 | Explainability and neurophysiological interpretation | `results/figures/nature/figure4_explainability_neurophysiology.png` |
| Results or Supplementary Information | Table 4 | Explainability biomarker table | `results/tables/table4_explainability_biomarkers.csv` |
| Figure 4 source visualization | MNE topomap contact sheet | PSD attribution and WPLI node-importance topomaps rendered with MNE-Python | `results/figures/explainability/mne_topomaps/mne_topomap_contact_sheet.png`; `results/figures/explainability/mne_topomaps/mne_topomap_manifest.csv` |
| Supplementary Information | Supplementary Figure 1 | Repeated-error subject analysis | `results/figures/nature/supplementary_error_subjects.png` |
| Supplementary Information | Supplementary Figure 2 | MNE-rendered WPLI connectivity attribution maps | `results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_contact_sheet.png` |
| Supplementary Information | Supplementary Figure 3 | Performance precision and validation-boundary audit | `results/figures/nature/supplementary_performance_precision.png` |
| Supplementary Information | Supplementary Figure 4 | Exploratory clinical baseline and EEG incremental-value audit | `results/figures/nature/supplementary_clinical_incremental_value.png` |

## Supplementary Information Document

The compiled supplementary document is:

`output/doc/ResidualAware_SSL_CNN_Supplementary_Information.docx`

It contains:

- Supplementary Figures 1-4.
- Supplementary Tables 1-11, covering confidence intervals, paired comparisons, permutation tests, clinical incremental comparisons, seed stability, non-identifying EEG recording metadata, participant-flow/safety source-note categories, the conservative PROBAST/TRIPOD+AI risk audit, performance precision boundaries for the 19-patient LOSO analysis, claim-strength alignment for central manuscript statements, and an AI model reporting card.

## Export files

Each generated figure has `.png`, `.svg`, `.pdf`, and `.tiff` exports. The complete export manifest is:

`results/figures/nature/figure_manifest.csv`

The visual QA contact sheet is:

`results/figures/nature/nature_figures_contact_sheet.png`

The MNE WPLI connectivity supplementary contact sheet is:

`results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_contact_sheet.png`

The MNE topomap contact sheet is:

`results/figures/explainability/mne_topomaps/mne_topomap_contact_sheet.png`

The performance precision supplementary figure is:

`results/figures/nature/supplementary_performance_precision.png`

The exploratory clinical incremental-value supplementary figure is:

`results/figures/nature/supplementary_clinical_incremental_value.png`

## Submission notes

Use `.tiff` for journals that require high-resolution raster upload, `.pdf` or `.svg` for vector-preserving review files, and `.png` for manuscript drafting. Figure 4 includes embedded raster panels from earlier explainability outputs; if the target journal requests fully editable vector panels, regenerate those source panels directly from the explanation tables before final submission.
