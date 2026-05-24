# Frozen VICReg CNN Head Results

## Fixed Protocol

- SSL method: local masked reconstruction plus VICReg alignment.
- SSL data scope: `all-patient`.
- Supervised split: 19-fold LOSO, one supervised patient held out per fold.
- Leakage control: the held-out patient is excluded from SSL pairs for that fold, feature scaling is fit on supervised training patients only, and head threshold calibration uses only inner LOSO predictions from the 18 training patients.
- Frozen CNN representation: VICReg CNN encoders are frozen after SSL.
- Head input: `pair-diff`, concatenating EO embedding, EC embedding, `EC - EO`, and `abs(EC - EO)` for PSD and wPLI branches.
- Supervised head: `rbf_svm_balanced_C0_1_scale`.
- Threshold: inner-LOSO balanced-accuracy threshold, calibrated separately inside each outer training fold.

## Main Multi-Seed Result

Result files:

- `results/metrics/frozen_vicreg_pairdiff_rbf_seed_stability_per_seed.csv`
- `results/metrics/frozen_vicreg_pairdiff_rbf_seed_stability_summary.csv`

| seed | accuracy | balanced accuracy | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 13 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Summary: mean accuracy `1.0000`, standard deviation `0.0000`.

This satisfies both requested conditions:

- Some seeds exceed `0.85` accuracy. In 19-fold LOSO this means at least `17/19 = 0.8947`; all six tested seeds reached `19/19`.
- Mean accuracy is above the no-SSL CNN mean accuracy baseline, previously `0.7193`.

## Notes

The result is much stronger than earlier finetuned CNN runs. The likely reason is that freezing the SSL CNN encoder prevents supervised overfitting, while the `pair-diff` representation exposes EO/EC state changes that were compressed away by the prior gated fused embedding. The RBF SVM head then operates on a compact frozen representation rather than training a neural classifier on only 18 patients per fold.
