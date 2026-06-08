# Figure And Table Submission Readiness Audit

This audit applies a Nature-style figure contract to the current manuscript figures and main tables. It checks figure logic, source-data traceability, caption presence, export coverage, raster nonblankness, and review-risk boundaries without changing any figure content.

Overall status: **PASS**

## Figure Contract Summary

| Figure | Archetype | Core conclusion | Evidence chain | Caption status | Export/visual status | Source status | Review risk | Status |
|---|---|---|---|---|---|---|---|---|
| Figure 1 | schematic-led composite | The study uses a locked participant-flow and residual-aware EEG modelling workflow. | participant flow counts -> label definition -> modality/state branches -> classification inference | present in Nature and JNE clean | PASS png 3742x2527, min RGB std 44.2; tiff 3742x2527, min RGB std 44.2; svg/pdf/tiff file sizes ok | all present | Author confirmation is still needed for final ethics, recruitment dates, intervention device details, and safety wording. | PASS |
| Figure 2 | quantitative grid | The final residual-aware SSL-CNN has internally validated ranking and calibration advantages over EEG-only comparators. | locked LOSO predictions -> bootstrap intervals -> paired comparisons -> calibration plot | present in Nature and JNE clean | PASS png 3820x2627, min RGB std 60.1; tiff 3820x2627, min RGB std 60.1; svg/pdf/tiff file sizes ok | all present | Clinical-only baseline remains strong; the figure should not be used to claim independent EEG incremental utility. | PASS |
| Figure 3 | quantitative grid | Residual-aware auxiliary supervision is the most consistent training signal in this cohort. | ablation table -> seed stability -> threshold sensitivity -> feature-family comparisons | present in Nature and JNE clean | PASS png 3934x2627, min RGB std 60.1; tiff 3934x2627, min RGB std 60.1; svg/pdf/tiff file sizes ok | all present | Self-supervised pretraining alone should remain framed as non-definitive in this cohort. | PASS |
| Figure 4 | asymmetric mixed-modality figure | Model explanations localize state- and frequency-dependent PSD and WPLI patterns as hypothesis-generating EEG biomarkers. | integrated gradients -> occlusion -> topomaps -> WPLI network summaries | present in Nature and JNE clean | PASS png 3562x2529, min RGB std 32.6; tiff 3562x2529, min RGB std 32.6; svg/pdf/tiff file sizes ok | all present | Interpret as model-dependent associations, not independently validated neural mechanisms. | PASS |
| Supplementary Figure 1 | quantitative grid | Repeated-error subjects identify review targets for future cohorts rather than label corrections. | subject-level error summaries -> model/seed recurrence -> post-hoc review boundary | present in Nature and JNE clean | PASS png 3560x1069, min RGB std 35.2; tiff 3560x1069, min RGB std 35.2; svg/pdf/tiff file sizes ok | all present | Post-hoc exploratory figure; do not revise labels based on this analysis. | PASS |
| Supplementary Figure 2 | asymmetric mixed-modality figure | WPLI attribution maps summarize top signed connectivity edges across states and frequency bands. | edge attribution table -> MNE scalp layout -> 12 state-band panels -> contact sheet | present in Nature and JNE clean | PASS contact sheet and 12 MNE rows with PNG/SVG/PDF | all present | Exploratory visualization; individual connectivity exports provide vector/PDF versions but the contact sheet is PNG-only. | PASS |
| Supplementary Figure 3 | quantitative grid | The small 19-patient LOSO validation has coarse metric resolution and imprecise paired-comparison evidence. | bootstrap intervals -> paired differences -> one-case sensitivity/specificity movement | present in Nature and JNE clean | PASS png 3962x2408, min RGB std 56.3; tiff 3962x2408, min RGB std 56.3; svg/pdf/tiff file sizes ok | all present | Use to constrain claims and avoid over-interpreting superiority. | PASS |
| Supplementary Figure 4 | quantitative grid | Clinical-only variables are strong and EEG-plus-clinical candidates do not show stable incremental gain. | clinical-only baseline -> EEG-plus-clinical paired bootstrap differences -> directional candidate screen | present in Nature and JNE clean | PASS png 3684x2761, min RGB std 50.9; tiff 3684x2761, min RGB std 50.9; svg/pdf/tiff file sizes ok | all present | Use to constrain EEG incremental-value claims; the exploratory clinical comparison is not a definitive clinical model selection analysis. | PASS |

## Main Table Source Trace

| Table | Role | Caption status | Source file | Source status | Status |
|---|---|---|---|---|---|
| Table 1 | cohort characteristics | present in Nature and JNE clean | `results/tables/table1_cohort_characteristics.csv` | present | PASS |
| Table 2 | main performance | present in Nature and JNE clean | `results/tables/table2_main_model_performance.csv` | present | PASS |
| Table 3 | ablation analysis | present in Nature and JNE clean | `results/tables/table3_ablation.csv` | present | PASS |
| Table 4 | explainability biomarkers | present in Nature and JNE clean | `results/tables/table4_explainability_biomarkers.csv` | present | PASS |

## Connectivity Export Detail

- MNE connectivity state-band rows: 12.
- Rows with PNG/SVG/PDF exports present: 12.
- Contact sheet present: True.

## Topomap Export Detail

- MNE topomap rows: 16.
- Rows with PNG/SVG exports present: 16.
- Contact sheet status: PASS topomap contact sheet and 16 MNE rows with PNG/SVG.

## Interpretation

- Figures 1-4 are ready as manuscript-level visual arguments provided the unresolved author metadata are not inserted into figure text without confirmation.
- Figure 4 source visualizations include MNE-Python topomap exports and a non-blank topomap contact sheet.
- Supplementary Figure 2 is appropriate as a contact-sheet overview; the underlying 12 MNE panels preserve separate PNG/SVG/PDF exports for editorial requests.
- Figure 4 and Supplementary Figure 2 must remain explicitly hypothesis-generating because the explanations are model-dependent and the supervised cohort is small.
- Supplementary Figure 4 documents the clinical-only baseline boundary and should be retained if the manuscript discusses EEG incremental value.
- The figure package should be regenerated after any author-confirmed changes to cohort flow, intervention details, or safety reporting.
