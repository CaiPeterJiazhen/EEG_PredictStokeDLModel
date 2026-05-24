# Structured SSL Transfer Results For PSD+FC-wPLI CNN

This document records the first structured self-supervised learning trials for
the established `psd-fc-wpli` gated CNN.

## Purpose

The previous feature-space SSL sweep used the final PSD/wPLI feature tensors
directly. These trials test two more structured pretraining objectives:

- PSD branch: channel-by-frequency time-frequency image contrastive SSL.
- wPLI branch: graph-aware FC contrastive SSL with channel-node dropout and
  edge dropout.

The pretrained PSD and wPLI CNN encoder weights are transferred into the same
supervised `psd-fc-wpli` gated CNN documented in
`docs/best_psd_fc_wpli_gated_cnn_model.md`.

## Implementation

Code:

```text
src/eeg_recovery/training/train_structured_ssl.py
scripts/08_train_structured_ssl_transfer.py
tests/test_structured_ssl_transfer.py
```

For each LOSO fold:

1. Select the `all-patient` SSL pool.
2. Exclude the current LOSO test patient from the patient-containing SSL pool.
3. Build STFT log-power images shaped `windows x 62 x 90` for PSD-branch SSL.
4. Build wPLI graph views from 62-node upper-triangle FC edges.
5. Pretrain PSD and wPLI encoders with NT-Xent contrastive losses.
6. Initialize the supervised `psd-fc-wpli` gated CNN from the pretrained
   encoder weights.
7. Fine-tune with the same 19-patient LOSO supervised protocol.

Important alignment fix:

```text
TFR images now apply the same affected-hand hemisphere alignment as PSD and FC
feature extraction before STFT. Healthy records use the no-flip convention
equivalent to affected hand "right".
```

## Commands

Initial unaligned exploratory runs:

```powershell
python -B scripts\08_train_structured_ssl_transfer.py --data-scope all-patient --device cuda --seed 2 --psd-ssl-epochs 5 --fc-ssl-epochs 5 --ssl-batch-size 8 --embedding-dim 32 --projection-dim 16 --max-tfr-windows-per-record 1 --supervised-epochs 100 --patience 100 --summary-name structured_ssl_psd_fc_wpli_transfer_seed2_psd5_fc5_summary.csv
python -B scripts\08_train_structured_ssl_transfer.py --data-scope all-patient --device cuda --seed 2 --psd-ssl-epochs 10 --fc-ssl-epochs 10 --ssl-batch-size 8 --embedding-dim 32 --projection-dim 16 --max-tfr-windows-per-record 2 --supervised-epochs 100 --patience 100 --summary-name structured_ssl_psd_fc_wpli_transfer_seed2_psd10_fc10_summary.csv
```

Hemisphere-aligned TFR run:

```powershell
python -B scripts\08_train_structured_ssl_transfer.py --data-scope all-patient --device cuda --seed 2 --psd-ssl-epochs 10 --fc-ssl-epochs 10 --ssl-batch-size 8 --embedding-dim 32 --projection-dim 16 --max-tfr-windows-per-record 2 --supervised-epochs 100 --patience 100 --summary-name structured_ssl_psd_fc_wpli_transfer_aligned_seed2_psd10_fc10_summary.csv
```

## Results

Reference no-SSL gated CNN:

```text
accuracy=0.8421, balanced_accuracy=0.8333, roc_auc=0.8222, pr_auc=0.8158
```

Structured SSL trials:

```text
run                         tfr_aligned   ssl_epochs   tfr_windows   accuracy   balanced_acc   sensitivity   specificity   roc_auc   pr_auc
TFR+graph SSL 5/5           no            5/5          1             0.6316     0.6278         0.7000        0.5556        0.7111    0.7534
TFR+graph SSL 10/10         no            10/10        2             0.7368     0.7278         0.9000        0.5556        0.6889    0.6416
TFR-aligned+graph SSL 10/10 yes           10/10        2             0.6316     0.6333         0.6000        0.6667        0.7778    0.7965
```

The aligned 10/10 run covered all 19 LOSO test patients.

Aligned SSL pretraining losses decreased:

```text
psd_tfr:    first_loss=1.7302, last_loss=0.4587, min_loss=0.3950
wpli_graph: first_loss=1.3736, last_loss=0.3361, min_loss=0.2524
```

## Output Files

Aligned run:

```text
results/metrics/structured_ssl_psd_fc_wpli_transfer_aligned_seed2_psd10_fc10_summary.csv
results/metrics/dl_model_comparison_structured_ssl_all-patient_psd_tfr_aligned_wpli_graph_gated_cnn_seed2_psd10_fc10_bs8_sup100.csv
results/predictions/dl_loso_predictions_structured_ssl_all-patient_psd_tfr_aligned_wpli_graph_gated_cnn_seed2_psd10_fc10_bs8_sup100.csv
results/ssl/feature_ssl_pretraining_history_structured_ssl_all-patient_psd_tfr_aligned_wpli_graph_gated_cnn_seed2_psd10_fc10_bs8_sup100.csv
results/training_logs/dl_loss_history_structured_ssl_all-patient_psd_tfr_aligned_wpli_graph_gated_cnn_seed2_psd10_fc10_bs8_sup100.csv
results/figures/dl_loss_curve_structured_ssl_all-patient_psd_tfr_aligned_wpli_graph_gated_cnn_seed2_psd10_fc10_bs8_sup100.png
```

Unaligned exploratory runs were preserved with run names containing
`psd_tfr_wpli_graph`.

## Interpretation

This first structured SSL attempt does not improve the supervised CNN. The
loss curves show that both SSL objectives are learnable, and the pretrained
weights load into the supervised model, but transfer is not beneficial under
these settings.

The aligned TFR run improves ranking metrics compared with the unaligned 10/10
run (`roc_auc` 0.7778 vs 0.6889), but its fixed-threshold accuracy is lower.
That suggests the SSL initialization changes score calibration and fold-level
decision boundaries rather than producing a stronger classifier.

Likely reasons:

```text
The TFR SSL objective is still not identical to the downstream band-power PSD
objective; it transfers only low-level convolution filters.
The FC graph augmentation can be too destructive for a 19-subject supervised
target if the pretrained edge filters emphasize invariances that are not label
relevant.
The current transfer loads encoder weights only; it does not use a graph neural
network, ROI-level anatomical priors, or supervised calibration after SSL.
```

For reporting, keep the no-SSL `psd-fc-wpli` gated CNN as the current best
model unless later structured SSL tuning exceeds it under the same LOSO
protocol.
