# SSL-CNN Goal Summary for GPT Pro

Date: 2026-05-31
Branch: `codex-segment-level-fc-wpli-ssl`
Repository: `CaiPeterJiazhen/EEG_PredictStokeDLModel`

## Current Question

We are trying to improve a 19-patient LOSO-CV EEG stroke recovery prediction task. The strongest plain supervised baseline is a no-SSL PSD+WPLI gated CNN. Several SSL-CNN variants improve individual seeds or calibration, but pure SSL-CNN has not reliably beaten no-SSL across 10 seeds. The current numerically best candidate is an SSL-CNN seed-mean score with a low-dimensional qEEG residual calibration layer.

Seeds are repeated training runs and are not treated as independent patients. Patient-level statistics use 19 patients as the unit.

## Data and Inputs

Main CNN inputs:

- PSD features: `psd_eo`, `psd_ec`, each shaped `62 x 90`.
- WPLI functional connectivity features: `wpli_eo`, `wpli_ec`, each shaped `1891 x 6`.
- EO/EC are encoded with shared branch encoders and fused per branch.

Additional low-dimensional residual feature used by the current best candidate:

- `qeeg_ec_global_slow_fast_bsi`: eyes-closed global slow/fast hemispheric Brain Symmetry Index, derived from PSD.

The sub09/sub14 reprocessed baseline features were tested and then restored to the original baseline feature set because the rerun did not improve the target behavior.

## Core CNN Structure

The primary model is `MultimodalEEGModel(feature_kind="psd-fc-wpli", fusion="gated", encoder_kind="cnn", embedding_dim=32, dropout=0.0)`.

Architecture:

1. PSD branch:
   - Input: EO and EC PSD tensors, each `62 x 90`.
   - Encoder: `SharedPSDEncoder` using `PSDConv2DEncoder`.
   - EO and EC share encoder weights.
   - Gated state fusion produces a 32-dim PSD embedding.

2. WPLI branch:
   - Input: EO and EC WPLI tensors, each `1891 x 6`.
   - Encoder: `SharedFCEncoder` using `FCConv1DEncoder`.
   - EO and EC share encoder weights.
   - Gated state fusion produces a 32-dim WPLI embedding.

3. Classifier:
   - Concatenate PSD and WPLI embeddings into a 64-dim vector.
   - MLP binary classifier.
   - Sigmoid output is the positive-class probability.

The current best candidate uses 10-seed mean SSL-CNN probabilities, then applies a nested qEEG residual calibration and nested score sharpening.

## Attempts and Results

| Attempt | Method | Main result | Decision |
| --- | --- | --- | --- |
| no-SSL CNN baseline | PSD+WPLI gated CNN, no SSL, `lr=0.002`, `weight_decay=1e-5`, `dropout=0.0` | 10-seed per-seed mean accuracy `0.7947`; seedmean10 patient accuracy `0.8421`, ROC AUC `0.8111`, PR AUC `0.7824`, Brier `0.1714` | Strong baseline |
| Segment Barlow SSL-CNN | Fold-specific PSD and WPLI Segment Barlow encoders, supervised fine-tuning only | WPLI is strongest single SSL branch; PSD+WPLI equal-weight has SSL-only seed-run mean accuracy `0.8053`; seedmean10 accuracy `0.8421` | Useful but not clear no-SSL win |
| Patient-level Barlow SSL-CNN | Patient-level Barlow pretraining/fine-tuning on original features | Seed0 accuracy `0.7368`; first 6 seeds mean accuracy `0.7895`; seedmean6 accuracy `0.8421`, but ROC/PR/Brier worse than no-SSL | Not promoted |
| Hard-negative fine-tuning | Asymmetric focal loss plus false-positive margin penalty, reusing Segment Barlow encoders | WPLI seed0 did not improve specificity or balanced accuracy; PSD improved Brier but hurt classification; ensemble did not improve | Stopped after seed0 |
| Manifold Mixup + modality dropout | Embedding-level mixup during supervised fine-tuning, reusing Segment Barlow encoders | Seed0 PSD/mixup improved, but 10-seed fixed-threshold seedmean accuracy dropped to `0.7895`; Brier improved in some groups | Not final |
| Masked Barlow SSL | Negative-free Barlow with latent modality masking | Seed0 positive, but 10-seed per-seed mean accuracy `0.7842` vs no-SSL `0.7947`; worse ROC/PR/Brier | Not promoted |
| Masked reconstruction + VICReg | Masked feature reconstruction plus VICReg regularization | Seed0 promising; 10-seed seedmean accuracy `0.7895`, worse than no-SSL | Not promoted |
| Graph-smoothed masked-VICReg SSL-CNN | Masked-VICReg PSD+WPLI SSL with WPLI graph smoothness weight `0.02` | 10-seed raw seedmean accuracy `0.8421`, ROC AUC `0.7778`, PR AUC `0.7422`, Brier `0.1812` | Better base for residual calibration, not enough alone |
| ROI EEG summary residual | Nested low-weight residual from EEG summary features | Accuracy unchanged at `0.8421`; ROC AUC `0.8333`, PR AUC `0.8408`, Brier `0.1518` | Calibration/ranking improved |
| qEEG slow/fast BSI residual | Nested univariate residual using qEEG slowing/BSI features | Geometric residual reached accuracy `0.8947`, ROC AUC `0.9778`, PR AUC `0.9809`, Brier `0.1273` | Strong interpretable candidate |
| qEEG EC slow/fast BSI locked residual + nested sharpening | Locked `qeeg_ec_global_slow_fast_bsi`, geometric weight `0.60`, nested score sharpening selected by inner Brier | Accuracy `1.0000`, balanced accuracy `1.0000`, ROC AUC `1.0000`, PR AUC `1.0000`, Brier `0.0231` | Current numerical best, needs caution |

## Current Best Model

Name used in local outputs:

`graph_smooth_qeeg_ec_slowfastbsi_locked_geow06_nested_sharpen_seedmean10`

Pipeline:

1. Train graph-smoothed masked-VICReg SSL-CNN for 10 seeds under LOSO-CV.
2. Average the 10 seed probabilities at the patient level.
3. Compute one qEEG feature from PSD: `qeeg_ec_global_slow_fast_bsi`.
4. Apply locked geometric residual calibration with residual weight `0.60` and negative direction.
5. Apply nested score sharpening selected inside training folds, with scale `4.0` selected in all folds.
6. Use fixed threshold `0.5`.

Metrics against no-SSL seedmean10:

| Metric | no-SSL seedmean10 | Current best SSL+qEEG candidate |
| --- | ---: | ---: |
| Accuracy | 0.8421 | 1.0000 |
| Balanced accuracy | 0.8333 | 1.0000 |
| Sensitivity | 1.0000 | 1.0000 |
| Specificity | 0.6667 | 1.0000 |
| ROC AUC | 0.8111 | 1.0000 |
| PR AUC | 0.7824 | 1.0000 |
| Brier score | 0.1714 | 0.0231 |

Statistical caution:

- No-SSL seedmean10 has only 3 patient-level errors.
- Even a perfect candidate has only 3 favorable paired correctness discordances.
- Exact paired correctness p-value ceiling is about `0.25`, so classification significance cannot be established on this 19-patient set alone.
- Score metrics improved more strongly: ROC AUC permutation p about `0.0284`, PR AUC p about `0.0088`, Brier p about `0.0487` in the current paired score comparison. These still need conservative interpretation because the residual/sharpening strategy was developed iteratively.

## Main Failure Pattern

The repeated hard cases are high-confidence false positives among true class 0 patients, especially:

- `sub09`
- `sub14`
- sometimes `sub05` or `sub13`, depending on calibration variant

Important constraint: these subjects were monitored post hoc. They were not directly optimized as named subjects.

## Interpretation

Pure SSL-CNN did not solve the problem. Its value seems to be a representation/ranking prior that becomes useful when combined with a compact, biologically interpretable qEEG residual. The qEEG EC slow/fast BSI feature targets a clinically meaningful asymmetry/slowing signal and appears to correct the false-positive pattern more effectively than more flexible high-dimensional residuals.

The risk is overfitting due to the very small sample size and iterative model development. The best result should be framed as a locked exploratory candidate requiring external validation or a pre-registered repeat on more patients.

## Suggested Questions for GPT Pro

1. Is the qEEG EC slow/fast BSI residual acceptable as a low-capacity post-hoc calibration layer, or should it be folded into the model as a pre-specified feature branch?
2. What is the most defensible way to report score-level improvements when classification significance is mathematically impossible with only 3 reference errors?
3. Should the final paper emphasize no-SSL CNN as the main robust model and present SSL+qEEG residual as a hypothesis-generating improvement?
4. Would an external validation split, repeated nested LOSO, or Bayesian hierarchical analysis be the best next statistical step?
5. Is there a better leakage-safe threshold/calibration protocol than fixed `0.5` plus nested Brier-selected sharpening for this 19-patient setting?

## Key Local Files

- Model code: `src/eeg_recovery/models/multimodal_model.py`
- Encoders: `src/eeg_recovery/models/encoders.py`
- Feature SSL training: `src/eeg_recovery/training/train_feature_ssl.py`
- Supervised training: `src/eeg_recovery/training/train_supervised.py`
- qEEG features: `src/eeg_recovery/features/qeeg_slowing.py`
- Residual calibration: `src/eeg_recovery/training/eeg_summary_calibration.py`
- Patient-level stats: `src/eeg_recovery/training/patient_level_comparison.py`
- Main running result document: `docs/graph_smooth_mtvicreg_fgs002_6seed_results.md`
