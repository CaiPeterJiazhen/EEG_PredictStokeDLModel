# Prediction Validation And Leakage Integrity Audit

This audit records machine-checkable safeguards for the small-sample EEG prediction analyses. It focuses on the validation unit, leakage-sensitive preprocessing claims, uncertainty estimation, and seed/segment independence. It does not re-train models or verify author-supplied clinical protocol fields.

## Summary

- PASS: 21
- WARN: 0
- FAIL: 0

## Interpretation

- Current manuscript-facing model metrics consistently report the supervised validation unit as 19 subjects.
- SelectK feature-selection rows explicitly state that selection occurred inside LOSO training folds.
- Confidence intervals, paired comparisons, and permutation tests use subject-level summaries rather than segment-level rows.
- Seed-stability rows are model-level summaries and should not be interpreted as extra patient observations.
- Exploratory clinical/incremental rows remain explicitly separated from the manuscript-facing primary EEG rows.

## Checks

| Category | Check | Artifact | Observed | Expected | Status | Interpretation |
|---|---|---|---|---|---|---|
| primary_results | all_table2_rows_subject_level_n19 | table2_main_model_performance.csv | 19: 39 | all rows n_subjects = 19 | PASS | Table 2 stores patient-level metrics for the supervised cohort. |
| primary_results | inference_type_single_loso_or_seed_summary | table2_main_model_performance.csv | 10-seed ensemble: 1; 10-seed mean; classification-head inference: 1; LOSO: 23; single LOSO prediction per subject: 14 | inference type identifies single LOSO prediction or seed ensemble/mean | PASS | Inference labels should make clear that rows are not segment-level observations. |
| leakage_safeguard | selectk_declared_inside_loso_folds | table2_main_model_performance.csv | 7 SelectK rows; True: 7 | all SelectK rows say inside LOSO train folds | PASS | Feature selection must be fold-local to avoid test-fold leakage. |
| primary_results | paper_locked_manuscript_facing_rows_have_no_legacy_warning | paper_locked_model_performance.csv | 23 total warning rows; 0 non-exploratory warning rows | 0 non-exploratory legacy-warning rows | PASS | Legacy warnings are acceptable for clearly exploratory clinical/incremental support rows, but not for manuscript-facing primary EEG rows. |
| primary_results | paper_locked_class_counts | paper_locked_model_performance.csv | n_subjects=19: 39; n_positive=10: 39; n_negative=9: 39 | n_subjects=19, n_positive=10, n_negative=9 | PASS | All locked model rows should use the same supervised patient cohort and class split. |
| supplementary_results | supplementary_rows_with_n_use_subject_count_19 | supplementary_all_metrics.csv | 98 rows report n_subjects; non-19 rows=6; allowed threshold-sensitivity rows=6 | all non-19 rows are explicit exclude-margin threshold-sensitivity analyses | PASS | Non-19 rows are acceptable only when they explicitly exclude near-threshold subjects for sensitivity analysis. |
| supplementary_results | supplementary_row_types_explicit | supplementary_all_metrics.csv | ensemble10: 1; max: 4; mean: 4; min: 4; missing: 5; reported: 1; seedmean10: 3; std: 4 | row types distinguish reported/seedmean/ensemble/ablation material where present | PASS | Explicit row types reduce the risk of treating seed-level summaries as independent patients. |
| uncertainty | ci_rows_subject_level_n19 | model_metric_confidence_intervals.csv | 19: 3 | all rows n_subjects = 19 | PASS | Confidence intervals are reported for subject-level predictions. |
| calibration | calibration_columns_present | model_metric_confidence_intervals.csv | brier_score, calibration_intercept, calibration_slope, ece | brier_score, calibration_intercept, calibration_slope, ece | PASS | Calibration reporting supports Brier-score interpretation beyond discrimination. |
| paired_comparison | paired_comparisons_subject_level_n19 | model_pairwise_comparisons.csv | 19: 24 | all paired rows n_subjects = 19 | PASS | Paired comparisons should resample or compare subjects, not segments. |
| paired_comparison | bootstrap_rows_use_5000_resamples | model_pairwise_comparisons.csv | 5000.0: 21 | all bootstrap rows n_bootstrap = 5000 | PASS | The manuscript states paired bootstrap comparisons used 5,000 subject-level resamples. |
| paired_comparison | mcnemar_rows_have_discordant_counts | model_pairwise_comparisons.csv | 3 McNemar rows; n_discordant=0.0: 1; 4.0: 2 | McNemar rows include paired discordant counts | PASS | McNemar tests should be based on paired hard predictions. |
| permutation | permutation_rows_use_5000_valid_permutations | model_permutation_tests.csv | n_permutations=5000: 15; n_valid=5000: 15 | all rows use 5000 valid permutations | PASS | Permutation testing should use subject-level label permutations with valid repeats. |
| clinical_incremental | clinical_incremental_bootstrap_count | clinical_incremental_paired_bootstrap_comparison.csv | 500: 18 | bootstrap count is explicitly reported and consistent across rows | PASS | Clinical incremental analyses are exploratory and must not be overinterpreted. |
| clinical_incremental | clinical_incremental_table_has_subject_count | clinical_incremental_paired_bootstrap_comparison.csv | 19: 18 | all rows n_subjects = 19 | PASS | Clinical incremental support rows should expose the same subject-count audit trail as the main model tables. |
| seed_stability | seed_stability_summary_rows_not_patient_rows | seed_stability_table.csv | 3 model-level seed summary rows | model-level summaries only | PASS | Seed stability rows summarize variability across random seeds and are not treated as independent subjects. |
| textual_safeguard | phrase_present__patient_level_loso | tripod_ai_reporting_checklist.md | True | True | PASS | Key safeguards should be explicit in reviewer-facing audit documents. |
| textual_safeguard | phrase_present__no_seed_segment_independence | tripod_ai_reporting_checklist.md | True | True | PASS | Key safeguards should be explicit in reviewer-facing audit documents. |
| textual_safeguard | phrase_present__no_segment_level_rows_or_seed_rows | statistical_validation_summary.md | True | True | PASS | Key safeguards should be explicit in reviewer-facing audit documents. |
| textual_safeguard | phrase_present__bootstrap_resampling_is_over_subjects | statistical_validation_summary.md | True | True | PASS | Key safeguards should be explicit in reviewer-facing audit documents. |

## Reviewer-Facing Use

This audit supports the manuscript statements that evaluation was patient-level, LOSO-based, and not inflated by seed-level or segment-level rows. It should be kept as an internal QA document or shared as part of a reproducibility package if requested by reviewers.
