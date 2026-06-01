# Paper Experiment Completion Checklist

Updated: 2026-06-01

- [x] Freeze main subject-level LOSO model performance table from prediction CSVs.
- [x] Recompute current PSD+WPLI ML baseline metrics from the updated no-selector and SelectK=100 predictions.
- [x] Add bootstrap CIs, exact binomial tests, label permutation tests, calibration metrics, paired bootstrap differences, and McNemar tests.
- [x] Train baseline-only clinical models using only age, sex, duration, affected_hand, FMA_pre, and MBI_pre.
- [x] Add EEG-only and EEG+clinical incremental-value tables with paired bootstrap comparisons.
- [x] Aggregate core CNN/SSL/residual-aware ablation results and list missing long-run commands.
- [x] Run modality/state/band tabular ablations under the same subject-level LOSO protocol.
- [x] Evaluate residual-label threshold sensitivity, including fold-local train-only median and near-threshold exclusions.
- [x] Fix explainability path and residual-threshold configurability in `scripts/31_explain_residual_aware_ssl_cnn.py`.
- [x] Generate error-subject clinical/EEG/outlier summary for sub05, sub14, sub09, and sub28.
- [x] Generate manuscript table and figure bundle from CSV sources.
- [x] Add tests for statistical validation, clinical incremental baselines, threshold sensitivity, and explainability paths.
- [ ] Rerun full 190-sample explainability with local checkpoints/GPU if updated attribution CSVs are required.
- [ ] Optionally rerun exhaustive EEG+clinical RF/NB/KNN incremental models; current default covers Logistic L1/L2/SVM RBF.
- [ ] Add external validation or clearly state that this is internal LOSO pilot evidence only.
