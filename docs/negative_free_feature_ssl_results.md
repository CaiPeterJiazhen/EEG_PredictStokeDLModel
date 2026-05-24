# Negative-Free Feature SSL Results

## Goal

Implement feature-space self-supervised learning objectives that do not require negative samples, then transfer the pretrained encoder into the existing PSD + FC-wPLI gated CNN under 19-fold LOSO evaluation.

## Implemented Objectives

- `vicreg`: invariance + variance + covariance regularization on two augmented feature views.
- `barlow`: Barlow Twins cross-correlation objective on two augmented feature views.
- `byol`: online predictor and EMA target network with symmetric cosine prediction loss.

The previous `ntxent` option is retained for compatibility, but it uses batch-internal negatives and is not the negative-free option.

## Model And Data

- Input features: PSD `(62, 90)` + wPLI FC `(1891, 6)`.
- Model: multimodal gated CNN, finetuned after SSL pretraining.
- SSL data scope: `all-patient`.
- Evaluation: 19-fold LOSO on the supervised 19-patient list.
- Pretraining: 50 epochs, batch size 8, noise std 0.02, feature mask probability 0.05.
- Supervised training: 100 epochs.

## Commands

```powershell
python -B scripts\07_train_feature_ssl_transfer.py --device cuda --data-scopes all-patient --architecture multimodal --feature-kind psd-fc-wpli --fusion gated --encoder cnn --ssl-objective vicreg --pretrain-epochs 50 --pretrain-batch-size 8 --noise-std 0.02 --feature-mask-prob 0.05 --transfer-modes finetune --supervised-epochs 100 --seeds 0 1 2 3 7 13 --summary-name feature_vicreg_psd_fc_wpli_finetune_seed_stability.csv
```

```powershell
python -B scripts\07_train_feature_ssl_transfer.py --device cuda --data-scopes all-patient --architecture multimodal --feature-kind psd-fc-wpli --fusion gated --encoder cnn --ssl-objective byol barlow --pretrain-epochs 50 --pretrain-batch-size 8 --noise-std 0.02 --feature-mask-prob 0.05 --transfer-modes finetune --supervised-epochs 100 --seeds 2 7 13 --summary-name feature_noneg_byol_barlow_psd_fc_wpli_seed_scan.csv
```

## Results

| Objective | Seed | Accuracy | Balanced Accuracy | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| VICReg | 0 | 0.6842 | 0.6722 | 0.6333 |
| VICReg | 1 | 0.7368 | 0.7278 | 0.7667 |
| VICReg | 2 | 0.7368 | 0.7333 | 0.6778 |
| VICReg | 3 | 0.7368 | 0.7333 | 0.7889 |
| VICReg | 7 | 0.7368 | 0.7333 | 0.6778 |
| VICReg | 13 | 0.6842 | 0.6833 | 0.7222 |
| VICReg ensemble-6 | ensemble | 0.7895 | 0.7833 | 0.7222 |
| BYOL | 2 | 0.6842 | 0.6833 | 0.7111 |
| BYOL | 7 | 0.7368 | 0.7278 | 0.7556 |
| BYOL | 13 | 0.6842 | 0.6833 | 0.6667 |
| BYOL ensemble-3 | ensemble | 0.7895 | 0.7833 | 0.7444 |
| Barlow Twins | 2 | 0.7895 | 0.7833 | 0.7889 |
| Barlow Twins | 7 | 0.6842 | 0.6722 | 0.7222 |
| Barlow Twins | 13 | 0.8421 | 0.8333 | 0.8111 |
| Barlow Twins ensemble-3 | ensemble | 0.7895 | 0.7833 | 0.7778 |

Best negative-free SSL run:

- `barlow`, seed `13`: accuracy `0.8421`, balanced accuracy `0.8333`, ROC-AUC `0.8111`.

## Output Files

- `results/metrics/feature_vicreg_psd_fc_wpli_finetune_seed_stability.csv`
- `results/metrics/feature_noneg_byol_barlow_psd_fc_wpli_seed_scan.csv`
- `results/metrics/feature_barlow_psd_fc_wpli_more_seeds.csv`
- `results/metrics/feature_barlow_psd_fc_wpli_10seed_combined_summary.csv`
- `results/metrics/feature_barlow_proj32_mask001_10seed_combined_summary.csv`
- Per-run predictions are under `results/predictions/`.
- Per-run SSL histories are under `results/ssl/`.
- Per-run supervised loss histories and curves are under `results/training_logs/` and `results/figures/`.

## Barlow Twins 10-Seed Follow-Up

Additional Barlow Twins runs used seeds `0, 1, 3, 4, 5, 6, 8`, then were combined with the earlier seeds `2, 7, 13`.

| Seed | Accuracy | Balanced Accuracy | ROC-AUC |
| ---: | ---: | ---: | ---: |
| 0 | 0.6842 | 0.6722 | 0.6667 |
| 1 | 0.7368 | 0.7278 | 0.7222 |
| 2 | 0.7895 | 0.7833 | 0.7889 |
| 3 | 0.6316 | 0.6278 | 0.7556 |
| 4 | 0.7368 | 0.7278 | 0.7000 |
| 5 | 0.7895 | 0.7833 | 0.7111 |
| 6 | 0.6842 | 0.6722 | 0.7444 |
| 7 | 0.6842 | 0.6722 | 0.7222 |
| 8 | 0.6842 | 0.6778 | 0.6667 |
| 13 | 0.8421 | 0.8333 | 0.8111 |

10-seed single-model accuracy summary:

- Mean: `0.7263`
- Standard deviation: `0.0647`
- Minimum: `0.6316`
- Maximum: `0.8421`
- 10-seed probability ensemble accuracy: `0.7895`

## Barlow Twins Stability Tuning

The first stability tuning attempt changed the supervised fine-tuning stage to `supervised_lr=5e-4` and `dropout=0.1`. This did not help; the representative seed screen dropped below the default setup.

The best follow-up kept the original supervised fine-tuning settings and only changed the Barlow feature-space SSL setup:

- `projection_dim=32`
- `feature_mask_prob=0.01`
- `noise_std=0.02`
- `pretrain_epochs=50`
- `supervised_lr=1e-3`
- `dropout=0.0`

10-seed results for this tuned setting:

| Seed | Accuracy | Balanced Accuracy | ROC-AUC |
| ---: | ---: | ---: | ---: |
| 0 | 0.7368 | 0.7278 | 0.6778 |
| 1 | 0.7368 | 0.7278 | 0.7222 |
| 2 | 0.8421 | 0.8333 | 0.7889 |
| 3 | 0.6842 | 0.6778 | 0.6556 |
| 4 | 0.7368 | 0.7278 | 0.7444 |
| 5 | 0.7368 | 0.7278 | 0.7889 |
| 6 | 0.6842 | 0.6722 | 0.7556 |
| 7 | 0.7368 | 0.7333 | 0.8000 |
| 8 | 0.7368 | 0.7333 | 0.7889 |
| 13 | 0.8421 | 0.8333 | 0.7667 |

Compared with the previous 10-seed Barlow baseline:

| Setting | Mean Accuracy | Std | Min | Max | 10-Seed Ensemble |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default Barlow | 0.7263 | 0.0647 | 0.6316 | 0.8421 | 0.7895 |
| Barlow `proj32`, mask `0.01` | 0.7474 | 0.0544 | 0.6842 | 0.8421 | 0.7895 |

This tuning improved the average seed accuracy by `+0.0211`, raised the weakest seed by `+0.0526`, and reduced seed-to-seed variance.
