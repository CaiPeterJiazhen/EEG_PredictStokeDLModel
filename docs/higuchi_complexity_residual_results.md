# Higuchi Complexity EEG Residual Pilot

Date: 2026-05-30

## Question

Evaluate whether raw-EEG Higuchi fractal dimension (FD) features can add an interpretable EEG-derived signal that helps the current best negative-sample-free SSL-CNN beat the no-SSL CNN baseline.

This pilot keeps PSD and WPLI-FC unchanged. Higuchi FD is computed from baseline EO/EC raw EEG as an additional interpretable residual feature source.

## Literature Rationale

AnySearch found a directly relevant stroke EEG paper:

- Fractal Dimension of EEG Activity Senses Neuronal Impairment in Acute Stroke: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0100199&type=printable

The paper reports that resting EEG Higuchi FD is lower after acute stroke, relates to clinical impairment, and FD inter-hemispheric asymmetry is associated with recovery prognosis. This supports a trial of global, ROI, and ipsilesional-vs-contralesional FD summaries.

Additional context:

- Reorganization of cerebral networks after stroke: https://academic.oup.com/brain/article-pdf/134/5/1264/797723/awr033.pdf
- EEG connectivity tutorial/review: https://www.mdpi.com/2306-5354/10/3/372/pdf?version=1679046232

## Implementation

Added:

- `src/eeg_recovery/features/complexity.py`
- `scripts/28_compute_complexity_features.py`
- `tests/test_eeg_complexity_features.py`

Extended:

- `scripts/26_calibrate_eeg_summary_residual.py`
  - new `--summary-source complexity`

Feature files:

- `runs/baseline_rerun_20260529/data/features/complexity/*_complexity.npz`

Summary CSV:

- `runs/baseline_rerun_20260529/results/metrics/higuchi_complexity_residual_summary.csv`

## Feature Design

For each subject and state, the script computes channel-wise Higuchi FD after affected-hand hemisphere alignment, then summarizes:

- global FD mean/std
- ipsilesional FD mean
- contralesional FD mean
- ipsi-contra signed asymmetry
- ROI FD mean/std for frontal, central, temporal, parietal, occipital
- EO, EC, and absolute EC-EO delta summaries

The resulting features are low-dimensional and interpretable.

## Results

All results are patient-level seed-mean predictions at fixed threshold 0.5.

| Model | Source | Variant | Objective | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no-SSL CNN | base | base | none | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| masked-VICReg SSL-CNN + ROI residual | base | base | aucpr | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8333 | 0.8489 | 0.1599 |
| no-SSL CNN + Higuchi residual | complexity | multivariate | aucpr/brier/balanced_accuracy | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.6667 | 0.5960 | 0.1750 |
| SSL-CNN + ROI residual + Higuchi residual | complexity | multivariate | aucpr/brier/balanced_accuracy | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7222 | 0.6427 | 0.1802 |
| no-SSL CNN + Higuchi univariate | complexity | univariate | aucpr | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| SSL-CNN + ROI residual + Higuchi univariate | complexity | univariate | aucpr | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8333 | 0.8489 | 0.1599 |

The multivariate complexity residual preserves the classification threshold outcome but substantially degrades ranking and calibration. The nested univariate search selects zero residual weight in all outer folds, effectively falling back to the base model.

## Interpretation

Higuchi FD is biologically plausible and interpretable, but in this 19-patient LOSO setting it did not add reliable predictive value beyond the current PSD/WPLI models. It should be treated as a negative pilot, not as evidence that SSL-CNN improves over CNN.

The useful part is methodological: the repo now has a clean path for adding raw-EEG complexity features without changing PSD/WPLI inputs, and the nested residual evaluation prevents held-out patient leakage.
