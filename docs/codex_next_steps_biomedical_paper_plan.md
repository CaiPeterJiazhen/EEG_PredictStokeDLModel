# Codex Next Steps: Biomedical Paper Plan

Updated: 2026-06-01

## Immediate Manuscript Use

Use these files as the current paper results backbone:

- `docs/paper_locked_results_summary.md`
- `docs/statistical_validation_summary.md`
- `docs/clinical_incremental_value_results.md`
- `docs/core_ablation_results.md`
- `docs/modality_state_band_ablation_results.md`
- `docs/residual_threshold_sensitivity.md`
- `docs/error_subject_analysis.md`
- `docs/paper_results_narrative.md`

## Main Claim Boundary

The defensible main claim is:

> In a 19-patient internal LOSO pilot cohort, a CNN using baseline resting-state EEG PSD+WPLI outperformed the updated traditional ML PSD+WPLI baseline on accuracy and discrimination. Residual-aware SSL-CNN preserved accuracy while improving ROC-AUC, PR-AUC, Brier score/calibration, and stability-oriented evidence.

Do not claim clinical deployment, external generalization, causal biomarkers, or statistically definitive superiority.

## Local Long Runs

Run these only when GPU/checkpoints and time are available:

```powershell
python -B scripts\31_explain_residual_aware_ssl_cnn.py --config configs\paths.example.yaml --device cuda --residual-threshold 1.5 --max-sanity-samples 0
python -B scripts\39_train_clinical_and_incremental_baselines.py --incremental-models logistic_l1 logistic_l2 svm_rbf random_forest gaussian_nb knn --selector-options selectk100
python -B scripts\41_run_modality_state_band_ablation.py --models logistic_l1 logistic_l2 svm_rbf --feature-selection selectk100
```

## Writing Priorities

1. Convert `table1` and `table2` into Results text.
2. Report clinical-only strength before claiming EEG incremental value.
3. Use ablations and explainability as support/supplement unless full sanity checks are rerun.
4. Explicitly state that FMA_post, Delta_FMA_obs, Residual, and label-derived variables are never model inputs.
5. Frame threshold sensitivity as task-definition robustness, not model re-selection.
