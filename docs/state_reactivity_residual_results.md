# EO/EC State-Reactivity Residual Pilot

Date: 2026-05-30

## Question

Test whether interpretable eyes-closed minus eyes-open reactivity features derived from the existing PSD and WPLI-FC summaries can help the negative-sample-free SSL-CNN beat no-SSL CNN.

This keeps the original PSD and WPLI-FC inputs unchanged. Reactivity features are an additional derived EEG summary source.

## Literature Rationale

AnySearch did not find strong stroke-specific evidence for classical EO/EC alpha reactivity as a motor-recovery biomarker. The stronger stroke EEG evidence remains:

- higher-frequency activity and interhemispheric balance relate to recovery,
- PSD/BSI/phase-synchrony style qEEG features are plausible prognostic markers,
- lesioned-vs-nonlesioned activity imbalance is clinically meaningful.

Sources queried:

- EEG with or without TMS as biomarkers for post-stroke recovery: https://www.frontiersin.org/articles/10.3389/fneur.2022.827866/pdf
- Biomarkers for prognostic functional recovery poststroke: https://www.frontiersin.org/articles/10.3389/fcell.2022.1062807/pdf
- Current implications of EEG and fNIRS for motor recovery after stroke: https://www.degruyter.com/document/doi/10.1515/mr-2024-0010/pdf

Given this, reactivity was treated as a low-cost exploratory feature, not a strong prior.

## Implementation

Added:

- `src/eeg_recovery/features/eeg_reactivity.py`
- `tests/test_eeg_reactivity_features.py`

Extended:

- `scripts/26_calibrate_eeg_summary_residual.py`
  - new `--summary-source reactivity`

Feature construction:

- Pair any `psd_eo_*` with matching `psd_ec_*`.
- Pair any `wpli_eo_*` with matching `wpli_ec_*`.
- Emit `EC - EO` and normalized fractional change.

This yields 624 interpretable derived features per subject from the existing EEG summary.

Summary CSV:

- `runs/baseline_rerun_20260529/results/metrics/state_reactivity_residual_summary.csv`

## Results

All results are patient-level seed-mean predictions at fixed threshold 0.5.

| Model | Source | Variant | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| no-SSL CNN | base | base | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| masked-VICReg SSL-CNN + ROI residual | base | base | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8333 | 0.8489 | 0.1599 |
| no-SSL CNN + reactivity residual | state reactivity | multivariate low-weight | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.8080 | 0.1603 |
| SSL-CNN + ROI residual + reactivity residual | state reactivity | multivariate low-weight | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.8155 | 0.1585 |
| no-SSL CNN + alpha/beta BSI reactivity | state reactivity | univariate | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.6667 | 0.5988 | 0.2016 |
| SSL-CNN + alpha/beta BSI reactivity | state reactivity | univariate | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7778 | 0.7494 | 0.1743 |

Patient-level comparison of SSL reactivity multivariate vs no-SSL base:

- accuracy delta: 0.0000
- balanced accuracy delta: 0.0000
- ROC AUC delta: 0.0000
- PR AUC delta: +0.0331, bootstrap CI includes 0
- Brier delta: -0.0129, bootstrap CI includes 0

Patient-level comparison of SSL reactivity multivariate vs no-SSL reactivity multivariate:

- PR AUC delta: +0.0075
- Brier delta: -0.0019
- both are very small and not significant.

## Interpretation

State reactivity is not a useful continuation path. The multivariate residual slightly improves Brier relative to the SSL base but lowers ROC/PR compared with the stronger masked-VICReg + ROI residual result. The constrained univariate alpha/beta BSI version is clearly worse.

This should be treated as a negative pilot. It remains useful code because it provides a clean, interpretable EO/EC derived feature source for future ablations.
