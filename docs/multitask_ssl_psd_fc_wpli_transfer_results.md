# Multi-Task SSL Transfer Results For PSD+FC-wPLI CNN

This document records the multi-task SSL experiment requested after local
masked SSL tuning:

```text
local masked reconstruction + feature-space contrastive loss
```

The transfer target remains the established `psd-fc-wPLI` gated CNN.

## Implementation

Code:

```text
src/eeg_recovery/training/train_masked_ssl.py
scripts/09_train_masked_ssl_transfer.py
tests/test_masked_ssl_transfer.py
```

The new path is enabled by setting `--contrastive-weight` above zero. With the
default `--contrastive-weight 0.0`, the script preserves the previous local
masked SSL behavior. The feature-space alignment method is selected by
`--alignment-method`.

Multi-task pretraining uses one shared `psd-fc-wpli` gated CNN backbone per
LOSO fold:

1. PSD masked local reconstruction from pre-pooling Conv2D feature maps.
2. wPLI graph/edge masked local reconstruction from pre-pooling Conv1D edge
   feature maps.
3. Feature-space alignment on two augmented views of the multimodal EO/EC pair
   embedding.
4. Transfer the pretrained `branch_models.*` weights into supervised LOSO
   finetuning.

The reconstruction decoders and projection head are discarded before
supervised training.

Supported alignment methods:

```text
ntxent: contrastive NT-Xent loss, uses in-batch negatives.
vicreg: non-contrastive VICReg loss, uses invariance, variance, and covariance terms without negatives.
barlow: non-contrastive Barlow Twins loss, uses cross-correlation redundancy reduction without negatives.
byol: non-contrastive BYOL-style online/EMA-target prediction loss without negatives.
```

## Fixed Training Protocol

```text
SSL data scope: all-patient
strict LOSO SSL exclusion: enabled
SSL pairs per fold: 88
feature_kind: psd-fc-wpli
fusion: gated
encoder_kind: cnn
embedding_dim: 32
dropout: 0.0
PSD SSL epochs: 20
wPLI SSL epochs: 20
supervised epochs: 100
supervised patience: 100
seed: 2
device: cuda
CUDA device: NVIDIA TITAN Xp COLLECTORS EDITION
```

Mask parameters used the best local masked SSL setting:

```text
psd_channel_mask_prob=0.15
psd_frequency_mask_prob=0.15
psd_element_mask_prob=0.02
fc_node_mask_prob=0.02
fc_edge_mask_prob=0.05
fc_band_mask_prob=0.02
eo_ec_consistency_weight=0.0
contrastive_temperature=0.2
projection_dim=16
contrastive_noise_std=0.02
contrastive_feature_mask_prob=0.05
```

## Commands

Best multi-task run:

```powershell
python -B scripts\09_train_masked_ssl_transfer.py --config configs\paths.example.yaml --data-scope all-patient --device cuda --seed 2 --psd-ssl-epochs 20 --fc-ssl-epochs 20 --ssl-batch-size 8 --ssl-lr 0.001 --embedding-dim 32 --decoder-hidden-dim 128 --psd-channel-mask-prob 0.15 --psd-frequency-mask-prob 0.15 --psd-element-mask-prob 0.02 --fc-node-mask-prob 0.02 --fc-edge-mask-prob 0.05 --fc-band-mask-prob 0.02 --eo-ec-consistency-weight 0 --contrastive-weight 0.01 --contrastive-temperature 0.2 --projection-dim 16 --contrastive-noise-std 0.02 --contrastive-feature-mask-prob 0.05 --supervised-epochs 100 --patience 100 --supervised-lr 0.001 --dropout 0 --transfer-mode finetune --summary-name multitask_ssl_psd_fc_wpli_transfer_ctr0_01_summary.csv
```

## Results

Summary file:

```text
results/metrics/multitask_ssl_psd_fc_wpli_transfer_summary_all.csv
```

Combined NT-Xent, VICReg, Barlow Twins, and BYOL alignment summary:

```text
results/metrics/multitask_alignment_ssl_psd_fc_wpli_transfer_summary_all.csv
```

Multi-task weight scan:

```text
contrastive_weight   accuracy   balanced_acc   sensitivity   specificity   roc_auc   pr_auc
0.25                 0.5789     0.5722         0.7000        0.4444        0.6889    0.7385
0.05                 0.7368     0.7333         0.8000        0.6667        0.6667    0.6711
0.01                 0.8421     0.8333         1.0000        0.6667        0.7667    0.6828
```

Reference rows:

```text
method                         accuracy   balanced_acc   roc_auc   pr_auc
no SSL gated CNN seed2          0.8421     0.8333         0.8222    0.8158
feature contrastive SSL seed2   0.8421     0.8333         0.8000    0.7828
local masked SSL tuned seed2    0.7895     0.7833         0.8111    0.7957
multi-task SSL weight 0.01      0.8421     0.8333         0.7667    0.6828
```

## Output Files

Best multi-task run:

```text
results/metrics/multitask_ssl_psd_fc_wpli_transfer_ctr0_01_summary.csv
results/metrics/dl_model_comparison_multitask_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_fcnode0_02_fcedge0_05_fcband0_02_cons0_0_ctr0_01_temp0_2_noise0_02_mask0_05_bs8_sup100.csv
results/predictions/dl_loso_predictions_multitask_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_fcnode0_02_fcedge0_05_fcband0_02_cons0_0_ctr0_01_temp0_2_noise0_02_mask0_05_bs8_sup100.csv
results/ssl/feature_ssl_pretraining_history_multitask_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_fcnode0_02_fcedge0_05_fcband0_02_cons0_0_ctr0_01_temp0_2_noise0_02_mask0_05_bs8_sup100.csv
results/training_logs/dl_loss_history_multitask_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_fcnode0_02_fcedge0_05_fcband0_02_cons0_0_ctr0_01_temp0_2_noise0_02_mask0_05_bs8_sup100.csv
results/figures/dl_loss_curve_multitask_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_fcnode0_02_fcedge0_05_fcband0_02_cons0_0_ctr0_01_temp0_2_noise0_02_mask0_05_bs8_sup100.png
```

Additional weight-scan outputs use the same naming pattern with
`ctr0_05` and `ctr0_25`.

## VICReg Variant

VICReg was added as a no-negative-sample alternative to NT-Xent. It is enabled
through:

```powershell
python -B scripts\09_train_masked_ssl_transfer.py --config configs\paths.example.yaml --data-scope all-patient --device cuda --seed 2 --psd-ssl-epochs 20 --fc-ssl-epochs 20 --ssl-batch-size 8 --ssl-lr 0.001 --embedding-dim 32 --decoder-hidden-dim 128 --psd-channel-mask-prob 0.15 --psd-frequency-mask-prob 0.15 --psd-element-mask-prob 0.02 --fc-node-mask-prob 0.02 --fc-edge-mask-prob 0.05 --fc-band-mask-prob 0.02 --eo-ec-consistency-weight 0 --contrastive-weight 0.001 --alignment-method vicreg --projection-dim 16 --contrastive-noise-std 0.02 --contrastive-feature-mask-prob 0.05 --vicreg-invariance-weight 25 --vicreg-variance-weight 25 --vicreg-covariance-weight 1 --vicreg-variance-target 1 --supervised-epochs 100 --patience 100 --supervised-lr 0.001 --dropout 0 --transfer-mode finetune --summary-name multitask_vicreg_ssl_psd_fc_wpli_transfer_ctr0_001_summary.csv
```

The script uses shorter `mtvicreg_*` run names for this variant to stay inside
Windows path-length limits.

VICReg history files include these columns:

```text
alignment_method
alignment_loss
vicreg_invariance_loss
vicreg_variance_loss
vicreg_covariance_loss
vicreg_invariance_weight
vicreg_variance_weight
vicreg_covariance_weight
vicreg_variance_target
```

A full 20/20 epoch VICReg run was attempted before the short run-name fix. The
training loop completed all 19 LOSO folds, but result writing failed because
the first VICReg run name exceeded the Windows path length. No partial VICReg
result files were retained. The implementation now has a regression test that
keeps VICReg run names below the project-safe length.

After the short run-name fix, the full VICReg run completed:

```text
alignment_method   weight   accuracy   balanced_acc   sensitivity   specificity   roc_auc   pr_auc
vicreg             0.001    0.7895     0.7833         0.9000        0.6667        0.7667    0.7623
```

Average final SSL losses across the 19 LOSO folds:

```text
total_loss              1.0532
reconstruction_loss     1.0426
alignment_loss         10.5877
vicreg_invariance       0.0001
vicreg_variance         0.2681
vicreg_covariance       3.8823
```

VICReg output files:

```text
results/metrics/multitask_vicreg_ssl_psd_fc_wpli_transfer_ctr0_001_summary.csv
results/metrics/dl_model_comparison_mtvicreg_all-patient_psdfcwpli_gated_finetune_s2_p20_f20_fn0_02_fe0_05_fb0_02_w0_001_bs8_sup100.csv
results/predictions/dl_loso_predictions_mtvicreg_all-patient_psdfcwpli_gated_finetune_s2_p20_f20_fn0_02_fe0_05_fb0_02_w0_001_bs8_sup100.csv
results/ssl/feature_ssl_pretraining_history_mtvicreg_all-patient_psdfcwpli_gated_finetune_s2_p20_f20_fn0_02_fe0_05_fb0_02_w0_001_bs8_sup100.csv
results/training_logs/dl_loss_history_mtvicreg_all-patient_psdfcwpli_gated_finetune_s2_p20_f20_fn0_02_fe0_05_fb0_02_w0_001_bs8_sup100.csv
results/figures/dl_loss_curve_mtvicreg_all-patient_psdfcwpli_gated_finetune_s2_p20_f20_fn0_02_fe0_05_fb0_02_w0_001_bs8_sup100.png
```

## VICReg Tuning

Lightweight VICReg tuning tested three alignment weights and three component
weight settings:

```text
alignment weight: 0.0005, 0.001, 0.002
component set A: invariance=25, variance=25, covariance=1
component set B: invariance=25, variance=10, covariance=0.2
component set C: invariance=10, variance=10, covariance=0.1
```

Summary file:

```text
results/metrics/multitask_vicreg_ssl_tuning_summary_all.csv
```

Best VICReg tuning rows:

```text
weight   inv   var   cov   accuracy   balanced_acc   roc_auc   pr_auc
0.001    25    25    1     0.7895     0.7833         0.8333    0.8802
0.002    25    25    1     0.7895     0.7778         0.8222    0.8542
0.0005   10    10    0.1   0.7368     0.7333         0.7222    0.7274
```

The original VICReg component weights were best. Lowering variance/covariance
weights did not help.

## Other No-Negative Methods

Barlow Twins and BYOL were implemented as additional no-negative alignment
methods in the same local masked SSL framework.

Barlow Twins command:

```powershell
python -B scripts\09_train_masked_ssl_transfer.py --config configs\paths.example.yaml --data-scope all-patient --device cuda --seed 2 --psd-ssl-epochs 20 --fc-ssl-epochs 20 --ssl-batch-size 8 --ssl-lr 0.001 --embedding-dim 32 --decoder-hidden-dim 128 --psd-channel-mask-prob 0.15 --psd-frequency-mask-prob 0.15 --psd-element-mask-prob 0.02 --fc-node-mask-prob 0.02 --fc-edge-mask-prob 0.05 --fc-band-mask-prob 0.02 --eo-ec-consistency-weight 0 --contrastive-weight 0.01 --alignment-method barlow --projection-dim 16 --contrastive-noise-std 0.02 --contrastive-feature-mask-prob 0.05 --barlow-offdiag-weight 0.005 --supervised-epochs 100 --patience 100 --supervised-lr 0.001 --dropout 0 --transfer-mode finetune --summary-name multitask_barlow_ssl_psd_fc_wpli_transfer_w0_01_summary.csv
```

BYOL command:

```powershell
python -B scripts\09_train_masked_ssl_transfer.py --config configs\paths.example.yaml --data-scope all-patient --device cuda --seed 2 --psd-ssl-epochs 20 --fc-ssl-epochs 20 --ssl-batch-size 8 --ssl-lr 0.001 --embedding-dim 32 --decoder-hidden-dim 128 --psd-channel-mask-prob 0.15 --psd-frequency-mask-prob 0.15 --psd-element-mask-prob 0.02 --fc-node-mask-prob 0.02 --fc-edge-mask-prob 0.05 --fc-band-mask-prob 0.02 --eo-ec-consistency-weight 0 --contrastive-weight 0.01 --alignment-method byol --projection-dim 16 --contrastive-noise-std 0.02 --contrastive-feature-mask-prob 0.05 --byol-momentum 0.99 --byol-predictor-hidden-dim 64 --supervised-epochs 100 --patience 100 --supervised-lr 0.001 --dropout 0 --transfer-mode finetune --summary-name multitask_byol_ssl_psd_fc_wpli_transfer_w0_01_summary.csv
```

Results:

```text
method        weight   accuracy   balanced_acc   roc_auc   pr_auc
Barlow        0.01     0.7368     0.7333         0.7556    0.7392
BYOL          0.01     0.7895     0.7833         0.7333    0.6594
```

Barlow Twins average final SSL losses:

```text
total_loss              1.0449
reconstruction_loss     1.0429
alignment_loss          0.2043
barlow_on_diag          0.0000
barlow_off_diag        40.8520
```

BYOL average final SSL losses:

```text
total_loss              1.0439
reconstruction_loss     1.0437
alignment_loss          0.0205
byol_loss               0.0205
```

## Ensemble And Threshold Calibration

Post-hoc ensemble and threshold calibration were added for saved LOSO
prediction CSVs:

```text
src/eeg_recovery/training/ensemble_calibration.py
scripts/10_ensemble_ssl_predictions.py
tests/test_ensemble_calibration.py
```

The current run ensembles the best completed NT-Xent seed2 prediction with the
best completed VICReg seed2 prediction. Calibration is leave-one-subject-out on
the saved OOF predictions: when calibrating one subject, its own label is
excluded and only the other 18 subjects are used to select thresholds. The
weight+threshold variant also selects the two-model ensemble weight from the
same other-subject calibration set.

Command:

```powershell
python -B scripts\10_ensemble_ssl_predictions.py --config configs\paths.example.yaml --output-prefix ntxent_vicreg_seed2_ensemble
```

Output files:

```text
results/metrics/ensemble_metrics_ntxent_vicreg_seed2_ensemble.csv
results/predictions/ensemble_predictions_ntxent_vicreg_seed2_ensemble_fixed0_5.csv
results/predictions/ensemble_predictions_ntxent_vicreg_seed2_ensemble_oof_balanced_accuracy.csv
results/predictions/ensemble_predictions_ntxent_vicreg_seed2_ensemble_oof_weight_threshold_balanced_accuracy.csv
```

Results:

```text
model                          decision_rule                         accuracy   balanced_acc   sensitivity   specificity   roc_auc   pr_auc
ntxent_seed2                   fixed_threshold                       0.8421     0.8333         1.0000        0.6667        0.7667    0.6828
ntxent_seed2                   oof_balanced_accuracy                 0.8421     0.8333         1.0000        0.6667        0.7667    0.6828
vicreg_seed2                   fixed_threshold                       0.7895     0.7833         0.9000        0.6667        0.8333    0.8802
vicreg_seed2                   oof_balanced_accuracy                 0.5789     0.5833         0.5000        0.6667        0.8333    0.8802
ensemble_mean                  fixed_threshold                       0.7895     0.7833         0.9000        0.6667        0.7889    0.8267
ensemble_mean                  oof_balanced_accuracy                 0.7895     0.7833         0.9000        0.6667        0.7889    0.8267
ensemble_oof_weight_threshold  oof_weight_threshold_balanced_accuracy 0.6316    0.6333         0.6000        0.6667        0.7333    0.7200
```

The ensemble/calibration step did not improve over the single NT-Xent seed2
model. Equal averaging diluted NT-Xent's fixed-threshold classification, and
the stricter OOF weight+threshold calibration overreacted to the small
18-subject calibration folds. The best current classification result remains
the NT-Xent seed2 model at accuracy `0.8421` and balanced accuracy `0.8333`.

## NT-Xent Multi-Seed Stability

The best NT-Xent multi-task SSL setting was rerun on a shared seed set to test
whether SSL improves seed stability:

```text
shared seeds: 0, 1, 2, 3, 7, 13
SSL setting: local masked reconstruction + NT-Xent, contrastive_weight=0.01
comparison: no-SSL psd-fc-wPLI gated CNN with the same seeds
```

During the seed13 run, the original long NT-Xent run name exceeded the Windows
path length when writing the SSL history file. The training loop had completed,
but no seed13 output was retained from that failed write. NT-Xent multi-task
run names now use the shorter `mtntxent_*` pattern, matching the earlier
VICReg path-length fix.

Per-seed summary:

```text
seed   no_ssl_acc   ssl_acc   no_ssl_bal   ssl_bal   no_ssl_auc   ssl_auc   no_ssl_pr   ssl_pr
0      0.5263       0.6316    0.5167       0.6222    0.6556       0.6889    0.6768      0.6617
1      0.6316       0.7368    0.6222       0.7278    0.7000       0.6889    0.7253      0.6639
2      0.8421       0.8421    0.8333       0.8333    0.8222       0.7667    0.8158      0.6828
3      0.6316       0.6316    0.6278       0.6167    0.6444       0.8000    0.6142      0.8499
7      0.7368       0.7368    0.7278       0.7278    0.7778       0.7556    0.8311      0.7428
13     0.7895       0.7895    0.7778       0.7833    0.8000       0.7667    0.7819      0.6922
```

Aggregate stability:

```text
protocol              n   acc_mean   acc_std   bal_mean   bal_std   auc_mean   auc_std   pr_mean   pr_std
no-SSL gated CNN      6   0.6930     0.1173    0.6843     0.1166    0.7333     0.0767    0.7409    0.0846
NT-Xent multi-task    6   0.7281     0.0843    0.7185     0.0863    0.7444     0.0455    0.7155    0.0721
```

The same-seed comparison shows a meaningful stability gain for accuracy and
balanced accuracy: SSL improves the mean by about `+0.0351` and reduces the
accuracy standard deviation by about `0.0330`. It does not improve PR-AUC, and
it does not exceed the best single-seed accuracy of `0.8421`.

The 6-seed SSL probability ensemble reached:

```text
accuracy=0.7895
balanced_accuracy=0.7833
roc_auc=0.7444
pr_auc=0.7278
```

Output files:

```text
results/metrics/multitask_ntxent_ssl_seed_stability_per_seed.csv
results/metrics/multitask_ntxent_ssl_seed_stability_summary.csv
results/metrics/multitask_ntxent_ssl_seed_stability_delta_vs_no_ssl.csv
results/metrics/dl_model_comparison_multitask_ntxent_ssl_seedensemble_seed0_1_2_3_7_13.csv
results/predictions/dl_loso_predictions_multitask_ntxent_ssl_seedensemble_seed0_1_2_3_7_13.csv
```

## VICReg Multi-Seed Stability

The best VICReg setting was rerun on the same shared seed set:

```text
shared seeds: 0, 1, 2, 3, 7, 13
SSL setting: local masked reconstruction + VICReg
contrastive_weight=0.001
vicreg_invariance_weight=25
vicreg_variance_weight=25
vicreg_covariance_weight=1
```

Per-seed summary:

```text
seed   no_ssl_acc   ntxent_acc   vicreg_acc   no_ssl_bal   ntxent_bal   vicreg_bal   vicreg_auc   vicreg_pr
0      0.5263       0.6316       0.7368       0.5167       0.6222       0.7278       0.7111       0.6534
1      0.6316       0.7368       0.7368       0.6222       0.7278       0.7278       0.7778       0.8209
2      0.8421       0.8421       0.7895       0.8333       0.8333       0.7833       0.8333       0.8802
3      0.6316       0.6316       0.7895       0.6278       0.6167       0.7778       0.7667       0.7382
7      0.7368       0.7368       0.6842       0.7278       0.7278       0.6778       0.7889       0.7758
13     0.7895       0.7895       0.7895       0.7778       0.7833       0.7833       0.8556       0.8612
```

Aggregate stability:

```text
protocol              n   acc_mean   acc_std   bal_mean   bal_std   auc_mean   auc_std   pr_mean   pr_std
no-SSL gated CNN      6   0.6930     0.1173    0.6843     0.1166    0.7333     0.0767    0.7409    0.0846
NT-Xent multi-task    6   0.7281     0.0843    0.7185     0.0863    0.7444     0.0455    0.7155    0.0721
VICReg multi-task     6   0.7544     0.0430    0.7463     0.0427    0.7889     0.0512    0.7883    0.0845
```

VICReg is the most stable SSL method tested so far on this seed set. It has the
highest mean accuracy and balanced accuracy, and its accuracy standard
deviation is roughly half of NT-Xent and about one third of the no-SSL
baseline. VICReg also has the best ROC-AUC and PR-AUC means.

The 6-seed VICReg probability ensemble reached:

```text
accuracy=0.7895
balanced_accuracy=0.7833
roc_auc=0.8333
pr_auc=0.8553
```

The ensemble accuracy is not higher than the best single seed, but its ROC-AUC
and PR-AUC are substantially better than the NT-Xent ensemble.

Output files:

```text
results/metrics/multitask_ssl_seed_stability_no_ssl_ntxent_vicreg_per_seed.csv
results/metrics/multitask_ssl_seed_stability_no_ssl_ntxent_vicreg_summary.csv
results/metrics/multitask_vicreg_ssl_seed_stability_delta.csv
results/metrics/dl_model_comparison_multitask_vicreg_ssl_seedensemble_seed0_1_2_3_7_13.csv
results/predictions/dl_loso_predictions_multitask_vicreg_ssl_seedensemble_seed0_1_2_3_7_13.csv
```

## Interpretation

The multi-task objective is highly sensitive to contrastive weight. A strong
contrastive weight (`0.25`) harms supervised transfer. A moderate weight
(`0.05`) recovers part of the local masked SSL performance. A very light
contrastive weight (`0.01`) reaches the same accuracy as the best no-SSL and
feature-SSL seed2 runs.

This is useful but still not a robust SSL improvement. The best multi-task
accuracy is high, but ROC-AUC and PR-AUC are lower than the no-SSL seed2
reference. The current result should therefore be treated as a promising
single-seed configuration rather than evidence that SSL improves the model
reliably.

The practical recommendation is:

```text
Use contrastive_weight=0.01 if continuing multi-task SSL experiments.
Do not use 0.05 or 0.25 as default settings.
For VICReg, weight 0.001 with inv25/var25/cov1 is the best no-negative
alignment setting so far. It does not exceed NT-Xent weight 0.01 in accuracy,
but it gives the best ROC-AUC and PR-AUC among the multi-task SSL alignment
runs.
Barlow Twins and BYOL did not beat the tuned VICReg result in this first pass.
For robustness, the next useful check is multi-seed validation of NT-Xent
weight 0.01 and VICReg weight 0.001 with inv25/var25/cov1.
```
