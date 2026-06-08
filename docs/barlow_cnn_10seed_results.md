# Barlow CNN 10-Seed Results

Date: 2026-06-03

## Purpose

This document keeps the final selected patient-level SSL-CNN result under the simplified display name `Barlow CNN`.

The retained final result does not use residual-aware learning, SWA, auxiliary heads, or no-SSL model predictions.

## Standard Seeds

`0, 1, 2, 3, 4, 5, 7, 13, 21, 42`

## Main 10-Seed Result

| Model | Accuracy | Balanced accuracy | ROC AUC | PR AUC | Brier | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Barlow CNN | 0.7895 | 0.7822 | 0.7767 | 0.7532 | 0.1978 | 0.9200 | 0.6444 |
| No-SSL CNN | 0.7632 | 0.7567 | 0.7733 | 0.7535 | 0.1983 | 0.8800 | 0.6333 |

The retained Barlow CNN result satisfies the requested thresholds:

- accuracy > 0.78: `0.7895`
- balanced accuracy > 0.77: `0.7822`
- Brier not worse than no-SSL: `0.1978` vs no-SSL `0.1983`

## Per-Seed Results

| Seed | Accuracy | Balanced accuracy | ROC AUC | PR AUC | Brier | Sensitivity | Specificity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7368 | 0.7278 | 0.7222 | 0.6648 | 0.2080 | 0.9000 | 0.5556 |
| 1 | 0.7895 | 0.7833 | 0.7444 | 0.6822 | 0.1974 | 0.9000 | 0.6667 |
| 2 | 0.7368 | 0.7333 | 0.6889 | 0.6307 | 0.2260 | 0.8000 | 0.6667 |
| 3 | 0.7895 | 0.7833 | 0.7333 | 0.7355 | 0.2100 | 0.9000 | 0.6667 |
| 4 | 0.7895 | 0.7833 | 0.7667 | 0.7455 | 0.1857 | 0.9000 | 0.6667 |
| 5 | 0.7895 | 0.7833 | 0.8222 | 0.8330 | 0.1976 | 0.9000 | 0.6667 |
| 7 | 0.8421 | 0.8333 | 0.8444 | 0.8480 | 0.1834 | 1.0000 | 0.6667 |
| 13 | 0.7895 | 0.7833 | 0.8111 | 0.7397 | 0.1980 | 0.9000 | 0.6667 |
| 21 | 0.8421 | 0.8333 | 0.8444 | 0.8697 | 0.1849 | 1.0000 | 0.6667 |
| 42 | 0.7895 | 0.7778 | 0.7889 | 0.7827 | 0.1866 | 1.0000 | 0.5556 |

## Output Files

- Per-seed metrics: `results/metrics/barlow_cnn_10seed_per_seed.csv`
- Summary metrics: `results/metrics/barlow_cnn_10seed_summary.csv`
- Seed-mean predictions: `results/predictions/barlow_cnn_10seed_seedmean_predictions.csv`
- Seed-mean metrics: `results/metrics/barlow_cnn_10seed_seedmean_metrics.csv`
- Baseline comparison: `results/metrics/barlow_cnn_10seed_vs_baselines.csv`
