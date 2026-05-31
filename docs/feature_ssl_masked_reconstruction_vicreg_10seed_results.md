# Masked Reconstruction + VICReg Feature SSL Results

Date: 2026-05-30

## Objective

This experiment tested a negative-sample-free hybrid SSL objective for the PSD+WPLI CNN:

- masked feature reconstruction to learn robust feature-level representations
- VICReg invariance/variance/covariance regularization between two independently masked views
- no contrastive negatives
- same downstream supervised CNN fine-tuning protocol and unchanged PSD/WPLI inputs

The rationale was that pure masked reconstruction improved seed0 ROC/PR but hurt accuracy and Brier, so a VICReg regularizer might stabilize the representation.

## Implementation

Added `masked-reconstruction-vicreg` to feature-level SSL in:

- `src/eeg_recovery/training/train_feature_ssl.py`
- `scripts/07_train_feature_ssl_transfer.py`
- `tests/test_feature_ssl_transfer.py`

The SSL history records:

- `reconstruction_loss`
- `vicreg_invariance_loss`
- `vicreg_variance_loss`
- `vicreg_covariance_loss`

## Seed0

| Model | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-SSL CNN seed0 | 0.6316 | 0.6222 | 0.8000 | 0.4444 | 0.6667 | 0.6528 | 0.2760 |
| original Barlow SSL-CNN seed0 | 0.7368 | 0.7222 | 1.0000 | 0.4444 | 0.7444 | 0.6912 | 0.2007 |
| masked reconstruction seed0, mask 0.25 | 0.6842 | 0.6722 | 0.9000 | 0.4444 | 0.7778 | 0.8237 | 0.2254 |
| masked reconstruction seed0, mask 0.75 | 0.6842 | 0.6778 | 0.8000 | 0.5556 | 0.7889 | 0.8549 | 0.2329 |
| masked reconstruction + VICReg seed0, mask 0.50 | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.8000 | 0.7829 | 0.1846 |

Seed0 was promising: it improved balanced accuracy, specificity, ROC AUC, PR AUC, and Brier versus original Barlow seed0, while matching its accuracy.

Post-hoc hard-negative monitoring for the seed0 hybrid:

| Subject | True | Score | Pred |
|---|---:|---:|---:|
| sub09 | 0 | 0.6389 | 1 |
| sub14 | 0 | 0.0009 | 0 |

## 10-Seed

Seeds: `0, 1, 2, 3, 4, 5, 7, 13, 21, 42`

Seed-run mean:

| Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7158 | 0.7083 | 0.8500 | 0.5667 | 0.7478 | 0.7369 | 0.2304 |

Patient seedmean10 ensemble:

| Model | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-SSL CNN seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| masked reconstruction + VICReg seedmean10 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7444 | 0.6931 | 0.1862 |
| masked reconstruction + VICReg + low-weight no-ratio ROI residual | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.8000 | 0.7961 | 0.1670 |

Post-hoc hard-negative monitoring for seedmean10:

| Subject | True | Score | Pred |
|---|---:|---:|---:|
| sub09 | 0 | 0.8456 | 1 |
| sub14 | 0 | 0.8007 | 1 |

## Decision

Do not promote `masked-reconstruction-vicreg` to the main candidate. It looked promising on seed0 but failed to generalize across seeds.

The 10-seed result is worse than no-SSL CNN on accuracy, balanced accuracy, specificity, ROC AUC, PR AUC, and Brier. The low-weight ROI residual improves ranking and Brier for this SSL base, but it still does not restore classification performance to no-SSL levels.

Current best candidate remains:

`masked-VICReg SSL-CNN seedmean10 + constrained low-weight no-ratio ROI/BSI/WPLI graph residual`

That candidate preserves no-SSL seedmean10 classification and improves ROC AUC, PR AUC, and Brier numerically, but the patient-level paired bootstrap/permutation tests are not significant.
