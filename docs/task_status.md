# Task Status

Updated: 2026-06-01

## Manuscript Locked Results Package

- Main locked model table: `results/tables/paper_locked_model_performance.csv`
- Locked prediction index: `results/predictions/paper_locked_model_predictions.csv`
- Statistical validation: `results/statistics/model_metric_confidence_intervals.csv`, `model_pairwise_comparisons.csv`, `model_permutation_tests.csv`
- Calibration figure: `results/figures/paper/calibration_curves.png`
- Clinical/incremental baselines: `results/metrics/clinical_only_model_comparison.csv`, `eeg_clinical_incremental_model_comparison.csv`
- Core ablation summary: `results/metrics/core_ablation_10seed_summary.csv`
- Modality/state/band ablation: `results/metrics/modality_state_band_ablation.csv`
- Residual threshold sensitivity: `results/metrics/residual_threshold_sensitivity.csv`
- Error subject analysis: `results/tables/error_subject_clinical_eeg_summary.csv`
- Paper tables and figures: `results/tables/table1_cohort_characteristics.*`, `table2_main_model_performance.*`, `table3_ablation.*`, `table4_explainability_biomarkers.*`, and `results/figures/paper/figure*.png`

## Current Primary Result Interpretation

- The correct traditional ML baseline is the updated PSD+WPLI subject-level LOSO baseline, not the older deprecated 0.8421 ML number.
- Primary ML baseline rows include Logistic L1 no selector, Logistic L2 no selector, and SVM RBF SelectK=100.
- The final residual-aware SSL-CNN preserves the same accuracy as no-SSL CNN but improves ROC-AUC, PR-AUC, and Brier score in the locked subject-level table.
- Clinical-only baseline is strong in this small cohort; EEG incremental value must be reported conservatively and as internal pilot evidence.

## Long-running Items Not Recomputed Here

- Full 10 seeds x 19 folds explanation regeneration was not rerun to avoid overwriting existing long attribution outputs and requiring local checkpoints/GPU.
- `scripts/31_explain_residual_aware_ssl_cnn.py` has been fixed so a full rerun uses `path_config.output_root`, configurable residual threshold, and default full sanity checks.
- EEG+clinical exhaustive model matrix for RF/NB/KNN can be run locally with `--incremental-models logistic_l1 logistic_l2 svm_rbf random_forest gaussian_nb knn`; the default interactive run used Logistic L1/L2/SVM RBF for EEG+clinical.

## Verification

- Targeted tests for statistical validation, clinical baselines, threshold sensitivity, and explainability paths passed.
- Full test suite was run at the end of this task; see the final response for the exact result.
