# Imaginary Coherence Residual Pilot

Date: 2026-05-30

## Question

Evaluate whether absolute imaginary coherence summaries from the existing FC files can add a useful, interpretable EEG connectivity signal to the current best negative-sample-free SSL-CNN.

This keeps PSD and WPLI-FC unchanged. `imaginary_coherence` is already saved in the FC `.npz` files, so this pilot does not recompute connectivity or alter the original inputs.

## Literature Rationale

AnySearch found a directly relevant stroke EEG paper:

- Coherent neural oscillations predict future motor and language improvement after stroke: https://academic.oup.com/brain/article-pdf/138/10/3048/13798362/awv200.pdf

That study used weighted node degree based on absolute imaginary coherence. Beta-band weighted node degree at ipsilesional motor cortex predicted later motor improvement, and contralesional theta-band degree related to recovery. This gives a strong prior for beta/theta motor-node imaginary coherence summaries.

Additional context:

- Relation Between EEG Measures and Upper Limb Motor Recovery in Stroke Patients: https://link.springer.com/content/pdf/10.1007/s10548-022-00915-y.pdf
- EEG/TMS biomarkers for post-stroke recovery: https://www.frontiersin.org/articles/10.3389/fneur.2022.827866/pdf

## Implementation

Added:

- `src/eeg_recovery/features/imaginary_coherence_summary.py`
- `tests/test_imaginary_coherence_summary.py`

Extended:

- `scripts/26_calibrate_eeg_summary_residual.py`
  - new `--summary-source imagcoh`

Feature construction:

- Use absolute imaginary coherence.
- Summarize EO, EC, and absolute EC-EO delta.
- Per band:
  - global abs mean/std
  - ipsilesional and contralesional intra-hemispheric abs mean
  - interhemispheric abs mean
  - signed asymmetry
  - global node strength
  - ipsilesional and contralesional strength
  - motor-area strength for C/FC/CP channels
  - motor strength signed asymmetry

Main summary CSV:

- `runs/baseline_rerun_20260529/results/metrics/imaginary_coherence_residual_summary.csv`

## Results

All results are patient-level seed-mean predictions at fixed threshold 0.5.

| Model | Source | Variant | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| no-SSL CNN | base | base | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| masked-VICReg SSL-CNN + ROI residual | base | base | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8333 | 0.8489 | 0.1599 |
| no-SSL CNN + imagcoh residual | imaginary coherence | multivariate low-weight | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.8211 | 0.1698 |
| SSL-CNN + ROI residual + imagcoh residual | imaginary coherence | multivariate low-weight | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7667 | 0.7411 | 0.1749 |
| no-SSL CNN + beta/theta motor imagcoh | imaginary coherence | univariate | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.7667 | 0.7892 | 0.1997 |
| SSL-CNN + beta/theta motor imagcoh | imaginary coherence | univariate | 0.6316 | 0.6222 | 0.8000 | 0.4444 | 0.6667 | 0.6706 | 0.2075 |

Patient-level paired comparison, SSL imagcoh residual vs no-SSL imagcoh residual:

- accuracy delta: 0.0000
- balanced accuracy delta: 0.0000
- ROC AUC delta: -0.0444
- PR AUC delta: -0.0800
- Brier delta: +0.0051

Patient-level paired comparison, SSL imagcoh residual vs SSL base:

- ROC AUC delta: -0.0667
- PR AUC delta: -0.1078
- Brier delta: +0.0150

## Interpretation

Despite strong literature motivation, imaginary coherence residuals are a negative pilot in this dataset. The full low-weight residual preserves threshold classification but degrades ranking and calibration for SSL. The beta/theta motor-node univariate residual is worse for both no-SSL and SSL, and especially poor for SSL.

This result should not be continued as a main path. The implementation remains useful for ablation and interpretability because it exposes lagged FC strength and motor-area node-strength features without changing the original PSD/WPLI inputs.
