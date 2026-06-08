# Manuscript Integrity Audit

This audit checks manuscript-source consistency for the current polished Nature manuscript and its submission variants. It covers citation numbering, reference identifiers, table and figure source paths, key numeric claims, manuscript-variant safeguards, and export availability. It does not verify author-supplied ethics, consent, EEG acquisition, raw preprocessing, or repository DOI fields.

## Summary

- PASS: 78
- WARN: 0
- FAIL: 0

## Checks

| Category | Artifact | Check | Observed | Expected | Status | Notes |
|---|---|---|---|---|---|---|
| citation | References | numbering_contiguous | 1-25 (25 entries) | 1-25 (25 entries) | PASS | Reference numbering should remain contiguous after citation edits. |
| citation | Main text | all_numeric_citations_have_reference_entries | none missing | none missing | PASS |  |
| citation | References | uncited_reference_entries | none | none preferred | PASS | Uncited references are not fatal for a draft but should be removed before journal upload. |
| citation | References | doi_or_url_present | all reference entries include DOI or URL | all reference entries include DOI or URL | PASS |  |
| table_source | results/tables/table1_cohort_characteristics.csv | table_source_exists | True | True | PASS |  |
| table_source | results/tables/table2_main_model_performance.csv | table_source_exists | True | True | PASS |  |
| table_source | results/tables/table3_ablation.csv | table_source_exists | True | True | PASS |  |
| table_source | results/tables/table4_explainability_biomarkers.csv | table_source_exists | True | True | PASS |  |
| figure_source | results/figures/nature/figure1_study_design_model.png | figure_file_exists | True | True | PASS |  |
| figure_export | results/figures/nature/figure1_study_design_model.png | journal_export_set | png, svg, pdf, tiff present | png, svg, pdf, tiff present | PASS |  |
| figure_source | results/figures/nature/figure2_performance_calibration.png | figure_file_exists | True | True | PASS |  |
| figure_export | results/figures/nature/figure2_performance_calibration.png | journal_export_set | png, svg, pdf, tiff present | png, svg, pdf, tiff present | PASS |  |
| figure_source | results/figures/nature/figure3_robustness_ablation.png | figure_file_exists | True | True | PASS |  |
| figure_export | results/figures/nature/figure3_robustness_ablation.png | journal_export_set | png, svg, pdf, tiff present | png, svg, pdf, tiff present | PASS |  |
| figure_source | results/figures/nature/figure4_explainability_neurophysiology.png | figure_file_exists | True | True | PASS |  |
| figure_export | results/figures/nature/figure4_explainability_neurophysiology.png | journal_export_set | png, svg, pdf, tiff present | png, svg, pdf, tiff present | PASS |  |
| figure_source | results/figures/nature/supplementary_error_subjects.png | figure_file_exists | True | True | PASS |  |
| figure_export | results/figures/nature/supplementary_error_subjects.png | journal_export_set | png, svg, pdf, tiff present | png, svg, pdf, tiff present | PASS |  |
| figure_source | results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_contact_sheet.png | figure_file_exists | True | True | PASS |  |
| figure_source | results/figures/nature/supplementary_performance_precision.png | figure_file_exists | True | True | PASS |  |
| figure_export | results/figures/nature/supplementary_performance_precision.png | journal_export_set | png, svg, pdf, tiff present | png, svg, pdf, tiff present | PASS |  |
| figure_source | results/figures/nature/supplementary_clinical_incremental_value.png | figure_file_exists | True | True | PASS |  |
| figure_export | results/figures/nature/supplementary_clinical_incremental_value.png | journal_export_set | png, svg, pdf, tiff present | png, svg, pdf, tiff present | PASS |  |
| figure_export | results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_manifest.csv | mne_connectivity_panel_exports | 12 rows; png=12, svg=12, pdf=12 | >=12 rows with png, svg, and pdf | PASS |  |
| cohort | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | supervised_n | found | 19 patients from source value 19 | PASS |  |
| cohort | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | all_patient_eeg_pool_n | found | 28 patients from source value 28 | PASS |  |
| cohort | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | age_mean_sd | found | Mean age was 64.7 years (SD 6.5) from source value 64.74 (6.49) | PASS |  |
| cohort | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | baseline_fma_mean_sd | found | Baseline FMA was 40.5 (SD 23.8) from source value 40.47 (23.83) | PASS |  |
| cohort | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | followup_fma_mean_sd | found | follow-up FMA was 45.5 (SD 23.2) from source value 45.47 (23.23) | PASS |  |
| cohort | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | observed_delta_mean_sd | found | observed FMA improvement was 5.0 points (SD 4.1) from source value 5.00 (4.07) | PASS |  |
| cohort | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | female_count | found | 11 patients were female from source value 11 | PASS |  |
| cohort | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | left_affected_count | found | 11 had left-sided affected upper limbs from source value 11 | PASS |  |
| cohort | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | label_distribution | found | 10 proportional-recovery and 9 poor-recovery from source value 10/9 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | final_residual_aware_ssl_cnn_accuracy | found | 0.842 from source value 0.842 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | final_residual_aware_ssl_cnn_balanced_accuracy | found | 0.833 from source value 0.833 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | final_residual_aware_ssl_cnn_roc_auc | found | 0.844 from source value 0.844 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | final_residual_aware_ssl_cnn_pr_auc | found | 0.836 from source value 0.836 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | final_residual_aware_ssl_cnn_brier_score | found | 0.126 from source value 0.126 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | no_ssl_cnn_accuracy | found | 0.842 from source value 0.842 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | no_ssl_cnn_balanced_accuracy | found | 0.833 from source value 0.833 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | no_ssl_cnn_roc_auc | found | 0.811 from source value 0.811 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | no_ssl_cnn_pr_auc | found | 0.808 from source value 0.808 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | no_ssl_cnn_brier_score | found | 0.177 from source value 0.177 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | logistic_eeg_baseline_accuracy | found | 0.737 from source value 0.737 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | logistic_eeg_baseline_balanced_accuracy | found | 0.733 from source value 0.733 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | logistic_eeg_baseline_roc_auc | found | 0.711 from source value 0.711 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | logistic_eeg_baseline_pr_auc | found | 0.775 from source value 0.775 | PASS |  |
| model_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | logistic_eeg_baseline_brier_score | found | 0.208 from source value 0.208 | PASS |  |
| clinical_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | clinical_only_logistic_l2_roc_auc | found | 0.911 from source value 0.911 | PASS |  |
| clinical_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | clinical_only_logistic_l2_pr_auc | found | 0.899 from source value 0.899 | PASS |  |
| clinical_metric | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | clinical_only_logistic_l2_brier_score | found | 0.105 from source value 0.105 | PASS |  |
| statistics | model_permutation_tests.csv | final_model_accuracy_permutation_p_reported | p = 0.004 | 0.004 rounded and reported | PASS |  |
| statistics | model_permutation_tests.csv | final_model_balanced_accuracy_permutation_p_reported | p = 0.003 | 0.003 rounded and reported | PASS |  |
| statistics | model_permutation_tests.csv | final_model_roc_auc_permutation_p_reported | p = 0.005 | 0.005 rounded and reported | PASS |  |
| statistics | model_permutation_tests.csv | final_model_pr_auc_permutation_p_reported | p = 0.015 | 0.015 rounded and reported | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | d_no_ssl_cnn_residual_heads_seedmean10_accuracy | found | 0.895 from source value 0.895 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | d_no_ssl_cnn_residual_heads_seedmean10_balanced_accuracy | found | 0.889 from source value 0.889 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | d_no_ssl_cnn_residual_heads_seedmean10_roc_auc | found | 0.922 from source value 0.922 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | d_no_ssl_cnn_residual_heads_seedmean10_pr_auc | found | 0.927 from source value 0.927 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | d_no_ssl_cnn_residual_heads_seedmean10_brier_score | found | 0.110 from source value 0.110 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | e_patient_barlow_ssl_residual_heads_seedmean10_accuracy | found | 0.842 from source value 0.842 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | e_patient_barlow_ssl_residual_heads_seedmean10_balanced_accuracy | found | 0.833 from source value 0.833 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | e_patient_barlow_ssl_residual_heads_seedmean10_roc_auc | found | 0.844 from source value 0.844 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | e_patient_barlow_ssl_residual_heads_seedmean10_pr_auc | found | 0.836 from source value 0.836 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | e_patient_barlow_ssl_residual_heads_seedmean10_brier_score | found | 0.126 from source value 0.126 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | c_patient_barlow_ssl_no_residual_heads_ensemble10_roc_auc | found | 0.700 from source value 0.700 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | c_patient_barlow_ssl_no_residual_heads_ensemble10_pr_auc | found | 0.631 from source value 0.631 | PASS |  |
| ablation | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | psd_only_row_roc_auc | found | 0.811 from source value 0.811 | PASS |  |
| manuscript_variant | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | old_overclaim_phrase_removed | False | False | PASS |  |
| manuscript_variant | docs/manuscript_residual_aware_ssl_cnn_nature_polished.md | ssl_pretraining_caveat_present | True | True | PASS |  |
| manuscript_variant | docs/manuscript_residual_aware_ssl_cnn_jne_structured.md | old_overclaim_phrase_removed | False | False | PASS |  |
| manuscript_variant | docs/manuscript_residual_aware_ssl_cnn_jne_structured.md | ssl_pretraining_caveat_present | True | True | PASS |  |
| manuscript_variant | docs/manuscript_residual_aware_ssl_cnn_nature_clean_placeholder.md | old_overclaim_phrase_removed | False | False | PASS |  |
| manuscript_variant | docs/manuscript_residual_aware_ssl_cnn_nature_clean_placeholder.md | ssl_pretraining_caveat_present | True | True | PASS |  |
| manuscript_variant | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md | old_overclaim_phrase_removed | False | False | PASS |  |
| manuscript_variant | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md | ssl_pretraining_caveat_present | True | True | PASS |  |
| manuscript_variant | docs/manuscript_residual_aware_ssl_cnn_nature_clean_placeholder.md | clean_variant_author_query_removed | False | False | PASS |  |
| manuscript_variant | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md | clean_variant_author_query_removed | False | False | PASS |  |

## Interpretation

- `FAIL` indicates a manuscript-source inconsistency that should be fixed before submission.
- `WARN` indicates a draft-level issue that may be acceptable temporarily but should be checked before journal upload.
- The author-query text in the working Nature and JNE manuscripts is intentional; clean placeholder variants are checked separately to ensure those lines are removed.
