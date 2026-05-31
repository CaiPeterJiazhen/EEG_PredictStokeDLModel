# Baseline Clinical Residual Calibration Pilot

Date: 2026-05-30

## Question

Test whether a low-capacity, interpretable residual model using baseline-only clinical covariates can improve the current best negative-sample-free SSL-CNN enough to separate it from the no-SSL CNN baseline.

This is not a pure EEG-only result. It is a multimodal baseline-covariate pilot. The comparison is only valid if the same clinical residual is also applied to no-SSL CNN.

## Literature Rationale

- Stroke recovery prediction commonly depends on baseline severity and clinical covariates, and EEG/FC biomarkers are usually evaluated in addition to clinical status rather than in isolation.
- Resting-state EEG PSD, BSI, and FC are plausible prognostic biomarkers, but FC evidence remains heterogeneous and often correlational.
- Negative-sample-free SSL remains relevant for biomedical signals because it can reduce reliance on labels without using contrastive negative pairs.

Sources queried with AnySearch:

- Functional connectivity drives stroke recovery: https://academic.oup.com/brain/article-pdf/145/4/1211/46768036/awab469.pdf
- EEG FC systematic review result: https://pmc.ncbi.nlm.nih.gov/articles/PMC13107850/
- Resting-state EEG with clinical measures for upper-limb recovery: https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2025.1577393/full
- Biomedical-signal SSL survey: https://ieeexplore.ieee.org/ielx7/6287639/6514899/10365170.pdf

## Leakage Control

Allowed clinical input columns come from `MODEL_INPUT_COLUMNS` in `src/eeg_recovery/metadata/labels.py`:

- age
- sex
- duration
- affected_hand
- FMA_pre
- MBI_pre

Excluded because they are post-treatment or label-derived:

- FMA_post
- MBI_post
- Delta_FMA_pred
- Delta_FMA_obs
- Residual
- label

The residual calibrator is nested patient-LOO. The held-out patient is not used for fitting the clinical logistic residual or selecting the residual weight.

## Implementation

Added:

- `src/eeg_recovery/features/clinical_summary.py`
- `tests/test_clinical_summary_features.py`

Extended:

- `scripts/26_calibrate_eeg_summary_residual.py`
  - new `--summary-source clinical`

Main summary CSV:

- `runs/baseline_rerun_20260529/results/metrics/baseline_clinical_residual_summary.csv`

## Results

All results are patient-level seed-mean predictions at fixed threshold 0.5.

| Model | Source | Objective | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no-SSL CNN | EEG only | none | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| masked-VICReg SSL-CNN + ROI residual | EEG only | aucpr | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8333 | 0.8489 | 0.1599 |
| no-SSL CNN + clinical residual | clinical | brier | 0.8947 | 0.8944 | 0.9000 | 0.8889 | 0.9111 | 0.8988 | 0.0945 |
| masked-VICReg SSL-CNN + ROI residual + clinical residual | clinical | brier | 0.8947 | 0.8944 | 0.9000 | 0.8889 | 0.8889 | 0.7980 | 0.0988 |

The best clinical-residual SSL version improves over its EEG-only SSL base in accuracy, balanced accuracy, specificity, and Brier. However, the same clinical residual improves no-SSL CNN at least as much, and no-SSL remains better on ROC AUC, PR AUC, and Brier.

Pairwise patient-level comparison for the brier-selected clinical residual:

- SSL clinical vs no-SSL clinical: same Acc/Bal Acc/Sens/Spec/F1.
- SSL clinical ROC AUC delta: -0.0222.
- SSL clinical PR AUC delta: -0.1008.
- SSL clinical Brier delta: +0.0043.

Clinical feature importance was identical for the SSL and no-SSL residual models because the same nested clinical model is fit against the same labels:

1. FMA_pre
2. MBI_pre
3. age
4. sex
5. duration
6. affected_hand

## Interpretation

This pilot is useful but does not satisfy the main goal. Baseline clinical covariates are strongly predictive and interpretable, but the gain is not SSL-specific. Adding them makes the overall model better, while no-SSL CNN still matches or exceeds SSL-CNN after the same covariates are added.

The next useful direction should not claim clinical residual as SSL evidence. It should either:

- improve the SSL representation itself while keeping no-SSL comparison fixed, or
- find EEG-derived additional features where SSL plus EEG residual beats no-SSL plus the same residual under nested patient-level evaluation.
