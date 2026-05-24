# No-SSL CNN Stability Results

This document records supervised no-SSL stability tuning for the established
`psd-fc-wpli` gated CNN.

## Goal

The goal was not to maximize one lucky seed. The goal was to improve seed
stability under the same 19-fold LOSO protocol:

```text
shared seeds: 0, 1, 2, 3, 7, 13
model: multimodal psd-fc-wPLI gated CNN
encoder: cnn
embedding_dim: 32
epochs: 100
patience: 100
device: cuda
```

The main stability criteria were:

```text
1. raise minimum seed accuracy
2. raise mean seed accuracy
3. lower seed-to-seed accuracy std
4. keep balanced accuracy aligned with accuracy
```

## Implementation Change

Supervised training now supports Adam weight decay:

```text
src/eeg_recovery/training/train_supervised.py
scripts/05_train_supervised_loso.py --weight-decay
tests/test_model_shapes.py
```

The training outputs also record `learning_rate`, `weight_decay`,
`embedding_dim`, `dropout`, and `seed` in the prediction/metric/loss tables.

## Trials

The first four trials used lower learning rates plus dropout. They reduced
performance, which suggests the failure mode was not overfitting alone.

The useful region was around higher learning rate and no dropout:

```text
trial                  lr       weight_decay   dropout
baseline               0.001    0              0
t8_lr2e3_wd1e5_d0      0.002    0.00001        0
t10_lr2e3_wd1e4_d0     0.002    0.0001         0
t12_lr3e3_wd1e5_d0     0.003    0.00001        0
```

## Main Results

Best stability candidates:

```text
trial                  acc_mean   acc_std   acc_min   bal_mean   bal_std   bal_min   roc_auc_mean   pr_auc_mean
baseline               0.6930     0.1173    0.5263    0.6843     0.1166    0.5167    0.7333         0.7409
t8_lr2e3_wd1e5_d0      0.7193     0.1035    0.5789    0.7120     0.1043    0.5722    0.7593         0.7685
t10_lr2e3_wd1e4_d0     0.7193     0.1035    0.5789    0.7120     0.1043    0.5722    0.7537         0.7522
t12_lr3e3_wd1e5_d0     0.7193     0.1185    0.5789    0.7093     0.1187    0.5722    0.7389         0.7421
```

Per-seed values for the recommended trial:

```text
trial: t8_lr2e3_wd1e5_d0
lr=0.002
weight_decay=0.00001
dropout=0

seed   accuracy   balanced_acc   roc_auc   pr_auc
0      0.6316     0.6167         0.7667    0.8309
1      0.5789     0.5722         0.6111    0.6510
2      0.8421     0.8333         0.8333    0.8408
3      0.6842     0.6833         0.7111    0.6998
7      0.7895     0.7833         0.7889    0.7751
13     0.7895     0.7833         0.8444    0.8135
```

This is a real stability improvement over the previous no-SSL baseline:

```text
accuracy mean: 0.6930 -> 0.7193
accuracy min:  0.5263 -> 0.5789
accuracy std:  0.1173 -> 0.1035
balanced mean: 0.6843 -> 0.7120
balanced min:  0.5167 -> 0.5722
```

It is still not as stable as the current VICReg SSL result:

```text
VICReg SSL acc_mean=0.7544, acc_std=0.0430
```

## Seed Ensembles

The best no-SSL seed ensemble came from `t12_lr3e3_wd1e5_d0`:

```text
accuracy=0.8421
balanced_accuracy=0.8333
roc_auc=0.8000
pr_auc=0.7905
```

For the recommended per-seed stability trial `t8_lr2e3_wd1e5_d0`, the seed
ensemble reached:

```text
accuracy=0.7895
balanced_accuracy=0.7833
roc_auc=0.8000
pr_auc=0.7829
```

## Recommendation

Use this as the no-SSL stable supervised protocol before adding SSL:

```text
architecture=multimodal
feature_kind=psd-fc-wpli
fusion=gated
encoder=cnn
embedding_dim=32
dropout=0
lr=0.002
weight_decay=0.00001
epochs=100
patience=100
```

This protocol improves mean/min accuracy without relying on one lucky seed.

## Output Files

Summary files:

```text
results/metrics/no_ssl_cnn_stability_schemeA_per_seed.csv
results/metrics/no_ssl_cnn_stability_schemeA_summary.csv
results/metrics/no_ssl_cnn_stability_schemeA_vs_baseline_per_seed.csv
results/metrics/no_ssl_cnn_stability_schemeA_vs_baseline_summary.csv
results/metrics/dl_model_comparison_no_ssl_cnn_stability_schemeA_seedensembles.csv
```

Each individual seed run was also preserved under:

```text
results/predictions/dl_loso_predictions_no_ssl_psdfcwpli_gated_cnn_<trial>_seed<seed>.csv
results/metrics/dl_model_comparison_no_ssl_psdfcwpli_gated_cnn_<trial>_seed<seed>.csv
results/training_logs/dl_loss_history_no_ssl_psdfcwpli_gated_cnn_<trial>_seed<seed>.csv
results/figures/dl_loss_curve_no_ssl_psdfcwpli_gated_cnn_<trial>_seed<seed>.png
```
