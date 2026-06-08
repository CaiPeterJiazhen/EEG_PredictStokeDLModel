# Updated sub05/sub28 No-SSL CNN Same-Configuration 10-Seed Results

Date: 2026-06-03

## Purpose

This run repeats the same patient-level CNN configuration used for the selected Barlow CNN comparison, but without Barlow SSL pretraining.

The standard seed set is:

`0, 1, 2, 3, 4, 5, 7, 13, 21, 42`

## Configuration

- Feature kind: `psd-fc-wpli`
- Supervised architecture: `multimodal`
- Fusion: `gated`
- Encoder: `cnn`
- Pretraining: none
- Transfer mode: none
- Supervised epochs: `100`
- Patience: `100`
- Learning rate: `0.001`
- Weight decay: `0.0`
- Embedding dim: `32`
- Dropout: `0.0`
- Loss: `bce`

## Command Pattern

Each seed was run with:

```powershell
python -u -B scripts/05_train_supervised_loso.py --config configs/paths.example.yaml --device cuda --architecture multimodal --feature-kind psd-fc-wpli --fusion gated --encoder cnn --epochs 100 --patience 100 --lr 0.001 --weight-decay 0.0 --embedding-dim 32 --dropout 0.0 --seed <seed> --output-tag updated_sub05_sub28_no_ssl_exact6cfg_standard10_seed<seed>
```

## Patient-Level 10-Seed Result

| Row | Accuracy | Balanced accuracy | ROC AUC | PR AUC | Brier | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed-run mean | 0.7632 | 0.7567 | 0.7733 | 0.7535 | 0.1983 | 0.8800 | 0.6333 |
| Seed-run std | 0.1000 | 0.1001 | 0.0880 | 0.1017 | 0.0367 | 0.1033 | 0.1054 |
| Seed-run min | 0.5789 | 0.5722 | 0.6667 | 0.6149 | 0.1500 | 0.7000 | 0.4444 |
| Seed-run max | 0.8947 | 0.8889 | 0.8778 | 0.8967 | 0.2603 | 1.0000 | 0.7778 |
| 10-seed score mean, fixed 0.5 | 0.7895 | 0.7833 | 0.7889 | 0.7936 | 0.1861 | 0.9000 | 0.6667 |

## Per-Seed Accuracy

| Seed | Accuracy | Balanced accuracy | ROC AUC | PR AUC | Brier |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5789 | 0.5722 | 0.6778 | 0.7011 | 0.2603 |
| 1 | 0.6842 | 0.6778 | 0.6667 | 0.6149 | 0.2205 |
| 2 | 0.8947 | 0.8889 | 0.8778 | 0.8323 | 0.1550 |
| 3 | 0.6842 | 0.6778 | 0.6667 | 0.6306 | 0.2304 |
| 4 | 0.6842 | 0.6778 | 0.7222 | 0.7190 | 0.2315 |
| 5 | 0.7895 | 0.7833 | 0.8111 | 0.8218 | 0.1889 |
| 7 | 0.8421 | 0.8389 | 0.8778 | 0.8967 | 0.1500 |
| 13 | 0.8421 | 0.8333 | 0.7556 | 0.6661 | 0.1869 |
| 21 | 0.8421 | 0.8333 | 0.8778 | 0.8774 | 0.1634 |
| 42 | 0.7895 | 0.7833 | 0.8000 | 0.7747 | 0.1963 |

## Output Files

- Per-seed metrics: `results/metrics/updated_sub05_sub28_no_ssl_exact6cfg_standard10_per_seed.csv`
- Summary metrics: `results/metrics/updated_sub05_sub28_no_ssl_exact6cfg_standard10_summary.csv`
- 10-seed score-mean predictions: `results/predictions/updated_sub05_sub28_no_ssl_exact6cfg_standard10_seedmean_predictions.csv`
- 10-seed score-mean metrics: `results/metrics/updated_sub05_sub28_no_ssl_exact6cfg_standard10_seedmean_metrics.csv`
