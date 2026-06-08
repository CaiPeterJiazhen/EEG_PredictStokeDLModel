# Numeric Claim Source Trace Audit

Overall status: **PASS**

This audit maps the main manuscript's key numeric claims to source tables, statistics files, metadata audits, or deterministic feature-grid calculations. It is designed to catch stale manuscript numbers after statistical or package updates.

## Summary

- PASS: 20
- WARN: 0
- FAIL: 0
- Trace rows: 20

## Trace Table

| Check | Domain | Status | Source | Source value | Manuscript claim |
|---|---|---|---|---|---|
| N01 | participant_flow | PASS | results/tables/participant_flow_safety_source_notes.csv | M1=29; EEG-indexed=28; supervised=19; unsupervised=9; non-indexed=1 | M1 source workbook, EEG-indexed pool, and supervised cohort counts |
| N02 | participant_flow | PASS | results/tables/participant_flow_safety_source_notes.csv | proportional=10; poor=9 | Proportional-recovery versus poor-recovery label counts |
| N03 | cohort | PASS | results/tables/table1_cohort_characteristics.csv | age=64.74 (6.49); FMA_pre=40.47 (23.83); FMA_post=45.47 (23.23); Delta_FMA_obs=5.00 (4.07) | Supervised cohort descriptive statistics |
| N04 | cohort | PASS | results/tables/table1_cohort_characteristics.csv | female=11 (57.9%); left_affected=11 (57.9%) | Sex and affected-side counts |
| N05 | outcome_definition | PASS | docs/project_context.md; src/eeg_recovery/metadata/labels.py | median_residual=1.5; label=1 if residual<=1.5 else 0 | Residual threshold is the supervised-cohort median residual |
| N06 | eeg_metadata | PASS | results/tables/eeg_recording_metadata_audit.csv | records=38; subjects=19; duration_mean=188.4; duration_min=101.0; duration_max=247.8 | Supervised baseline EO/EC EEG file count and duration range |
| N07 | eeg_metadata | PASS | results/tables/eeg_recording_metadata_audit.csv; docs/project_context.md | srate_values=[250.0]; nbchan_values=[62]; trials_values=[1] | Sampling rate, retained channels, and continuous one-trial files |
| N08 | eeg_feature_dimensions | PASS | methods feature-grid formula and 62-channel upper triangle | PSD bins=90; WPLI edges=62*61/2=1891; bands=6 | PSD bins and WPLI edge dimensions |
| N09 | primary_performance | PASS | results/tables/table2_main_model_performance.csv | accuracy=0.842; balanced_accuracy=0.833; sensitivity=1.000; specificity=0.667; roc_auc=0.844; pr_auc=0.836; brier_score=0.126 | Final residual-aware SSL-CNN patient-level LOSO performance |
| N10 | primary_performance | PASS | results/tables/table2_main_model_performance.csv | accuracy=0.842; balanced_accuracy=0.833; roc_auc=0.811; pr_auc=0.808; brier_score=0.177 | No-SSL CNN reference performance |
| N11 | primary_performance | PASS | results/tables/table2_main_model_performance.csv | accuracy=0.737; balanced_accuracy=0.733; roc_auc=0.711; pr_auc=0.775; brier_score=0.208 | PSD+WPLI logistic-regression EEG baseline performance |
| N12 | paired_comparison | PASS | results/statistics/model_pairwise_comparisons.csv | roc_auc: diff=0.033, ci=-0.144 to 0.214, p=0.788; pr_auc: diff=0.028, ci=-0.170 to 0.224, p=0.824; brier_score: diff=-0.051, ci=-0.121 to 0.004, p=0.078 | Final model versus no-SSL CNN paired differences |
| N13 | paired_comparison | PASS | results/statistics/model_pairwise_comparisons.csv | accuracy: diff=0.105, ci=-0.105 to 0.316, p=0.423; roc_auc: diff=0.133, ci=-0.216 to 0.476, p=0.458; brier_score: diff=-0.082, ci=-0.189 to 0.037, p=0.169 | Final model versus logistic-regression EEG baseline paired differences |
| N14 | paired_comparison | PASS | results/statistics/model_pairwise_comparisons.csv | a_correct_b_wrong=1; a_wrong_b_correct=3; n_discordant=4; p=0.625 | McNemar hard-prediction comparison against logistic EEG baseline |
| N15 | clinical_baseline | PASS | results/tables/table2_main_model_performance.csv | roc_auc=0.911; pr_auc=0.899; brier_score=0.105 | Clinical-only logistic baseline ranking and calibration metrics |
| N16 | clinical_incremental | PASS | results/statistics/clinical_incremental_paired_bootstrap_comparison.csv | comparisons=18; min_p=0.016; max_p=1.000 | Adding EEG to baseline clinical variables did not provide stable incremental gain |
| N17 | ablation | PASS | results/tables/table3_ablation.csv | accuracy=0.895; balanced_accuracy=0.889; roc_auc=0.922; pr_auc=0.927; brier_score=0.110 | No-SSL CNN with residual-aware heads seed-mean ablation performance |
| N18 | ablation | PASS | results/tables/table3_ablation.csv | accuracy=0.842; balanced_accuracy=0.833; roc_auc=0.844; pr_auc=0.836; brier_score=0.126 | Final patient-level Barlow SSL plus residual-aware ablation row |
| N19 | ablation | PASS | results/tables/table3_ablation.csv | roc_auc=0.700; pr_auc=0.631 | Patient-level Barlow SSL without residual-aware heads |
| N30 | precision | PASS | results/tables/performance_precision_audit.csv | 0.053 (One changed hard prediction moves accuracy by 5.3 percentage points) | One changed hard prediction moves accuracy by 5.3 percentage points |

## Warnings

None.

## Interpretation

- The audited numeric claims are traceable to current source artifacts.
- This audit does not replace raw-data release, author-confirmed protocol metadata, or final journal copyediting.
- Rerun this audit after any change to cohort selection, labels, model metrics, figure legends, or statistical tables.
